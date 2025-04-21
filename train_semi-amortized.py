import argparse
from sysconfig import get_path
import time
from warnings import filters
import yaml
import sys
import os
import shutil

import torch
import numpy as np
import torch.nn.functional as F
from multiprocessing import Process
from torch.cuda.amp import autocast, GradScaler
import pytorch3d.transforms as ptf

from utils.mrc import save_mrc
from utils.rot_calc_error import global_alignment

from utils.losses import *

from dataset import RealDataset, StarfileDataLoader
from torch.utils.data import DataLoader

from pose_models import PoseModel
from multihypo_models import CryoMEM

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


from utils.graph_functions import *

def convert_rotmats_to_euler_degrees(rotmats):
    """Convert rotation matrices to Euler angles (ZYZ), convert to degrees, 
    and wrap angles to standard ranges: psi/phi [-180, 180], tilt [0, 180].

    Parameters:
    -------------    
    rotmats : numpy.ndarray or torch.Tensor of shape (n, 3, 3)
        Rotation matrices.

    Returns:
    --------
    numpy.ndarray
        Euler angles (psi, theta, phi) in degrees, shape (n, 3).
    """
    if isinstance(rotmats, np.ndarray):
        rotmats_t = torch.from_numpy(rotmats)
    else:
        rotmats_t = rotmats
    
    # Ensure tensor is on CPU for conversion if it came from GPU
    rotmats_t = rotmats_t.cpu()

    euler_rad = ptf.matrix_to_euler_angles(rotmats_t, convention="ZYZ")
    euler_deg = np.rad2deg(euler_rad.numpy())
    
    # Wrap angles: psi (idx 0) and phi (idx 2) to [-180, 180], tilt (idx 1) to [0, 180]
    euler_deg[:, 0] = ((euler_deg[:, 0] + 180) % 360) - 180
    euler_deg[:, 2] = ((euler_deg[:, 2] + 180) % 360) - 180
    euler_deg[:, 1] = np.clip(euler_deg[:, 1], 0, 180) # Tilt is intrinsically [0, 180] but clip for safety
    
    return euler_deg

def convert_rotmats_to_euler(rotmats):
    """Convert rotation matrices to Euler angles using the ZYZ convention.

    Parameters:
    -------------
    rotmats : numpy.ndarray or torch.Tensor of shape (n, 3, 3)
        Rotation matrices.

    Returns:
    --------
    numpy.ndarray
        Euler angles of shape (n, 3) as returned by pytorch3d.transforms.matrix_to_euler_angles.
    """
    if isinstance(rotmats, torch.Tensor):
        rotmats = rotmats.detach().cpu().numpy()
    return ptf.matrix_to_euler_angles(torch.tensor(rotmats), convention="ZYZ").numpy()

def dict2cuda(a_dict):
    """
    Loads a dictionary on GPU.

    Parameters
    ----------
    a_dict: Dictionary

    Returns
    -------
    tmp: Dictionary
    """
    tmp = {}
    for key, value in a_dict.items():
       if isinstance(value,torch.Tensor):
           tmp.update({key: value.cuda()})
       else:
           tmp.update({key: value})
    return tmp

def parse_args(config):
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--data_type', type=str, choices=['real', 'all_correct', 'all_random', 'random_membrane', 'random_membrane_w_wobble', 'correct_mem_w_wobble'], default=config.get('data_type', 'all_correct'), help='Specify whether to use real or synthetic data.')
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 32))
    parser.add_argument('--r', type=float, default=config.get('r', 1.))
    parser.add_argument('--uninvert_data', action='store_true', default=config.get('uninvert_data', False))
    parser.add_argument('--big_dataset', action='store_true', default=config.get('big_dataset', False), help='Use the larger dataset.')
    parser.add_argument('--max_particles', type=int, default=config.get('max_particles', None), help='Maximum number of particles to use from the dataset.')
    parser.add_argument('--partial_prior', action='store_true', default=config.get('partial_prior', False), help='Use the partial prior.')
    parser.add_argument('--set90dist', action='store_true', default=config.get('set90dist', True), help='Set the 90 force on tilt.')
    parser.add_argument('--snr', type=float, default=config.get('snr', 1.0), help='Signal-to-noise ratio for synthetic data.')

    # Amortized Regime
    parser.add_argument('--amortized_method', type=str, choices=['s2s2', 'in_membrane', 'in_membrane_and_tilt', 'split_reg', 'gt', 'none'], default=config.get('amortized_method', 'in_membrane'), help='Select the amortized inference method.')
    parser.add_argument('--epochs_amortized', type=int, default=config.get('epochs_amortized', 10))
    parser.add_argument('--heads', type=int, default=config.get('heads', 7))
    parser.add_argument('--encoder_lr', type=float, default=config.get('encoder_lr', 0.0001))
    parser.add_argument('--space', type=str, choices=['fourier', 'hartley'], default=config.get('space', 'hartley'))
    parser.add_argument('--decoder_lr', type=float, default=config.get('decoder_lr', 0.005))
    parser.add_argument('--in_membrane_lr', type=float, default=config.get('in_membrane_lr', 0))
    parser.add_argument('--tilt_lr', type=float, default=config.get('tilt_lr', 0))
    parser.add_argument('--inplane_lr', type=float, default=config.get('inplane_lr', 0))
    parser.add_argument('--s2s2_lr', type=float, default=config.get('s2s2_lr', 0))
    parser.add_argument('--use_softmin', action='store_true', default=config.get('use_softmin', False), help='Use softmin instead of argmax for head selection.')
    parser.add_argument('--init_temperature', type=float, default=config.get('init_temp', 0.7))
    parser.add_argument('--temperature_hold_epochs', type=int, default=config.get('init_temp_hold_epochs', 4))
    parser.add_argument('--end_temp', type=int, default=config.get('end_temp', 0.2))
    
    # Unamortized Regime
    parser.add_argument('--epochs_unamortized', type=int, default=config.get('epochs_unamortized', 10))
    parser.add_argument('--rot_lr', type=float, default=config.get('rot_lr', 0.05))
    parser.add_argument('--vol_iters', type=int, default=config.get('vol_iters', 1))
    parser.add_argument('--pose_iters', type=int, default=config.get('pose_iters', 5))
    parser.add_argument('--path_to_checkpoint', type=str, default=config.get('path_to_checkpoint', 'none'))
    parser.add_argument('--update_hyperparams', action='store_true', default=config.get('update_hyperparams', False), help='If set, update training hyperparameters from new config even when resuming from checkpoint')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # GPU-specific optimizations for A40
    torch.backends.cudnn.benchmark = True  # Enable cuDNN auto-tuner
    torch.backends.cuda.matmul.allow_tf32 = True  # Allow TF32 for faster matrix multiplications on A40
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--config', type=str)
    known_args, remaining_args = parser.parse_known_args()

    save_path = known_args.save_path

    with open(known_args.config, 'r') as file:
        config = yaml.safe_load(file)

    sys.argv = [sys.argv[0]] + remaining_args
    args = parse_args(config)
    with open(os.path.join(save_path, 'config.yaml'), 'w') as file:
        yaml.dump(vars(args), file)
    writer = SummaryWriter(os.path.join(save_path, 'tbd'))

    # Create directories=
    paths = ['reconst_volumes', 'orientation_distributions', 'orientation_distributions_aligned',
             'init_viz', 'pred_vs_gt', 'head_stats', 'error_histograms', 'ckpt']
    for path in paths:
        if not os.path.exists(path):
            os.makedirs(os.path.join(save_path, path), exist_ok=True)
        else:
            shutil.rmtree(os.path.join(save_path, path))
            os.makedirs(os.path.join(save_path, path))

    path_dict = {}
    for path in paths:
        path_dict[path] = os.path.join(save_path, path)
                    
    # Load data
    B = args.batch_size
    num_workers = 4
    img_sz = 128
    
    if args.snr == 1.0:
        snr_str = '_snr_1.0'
    elif args.snr == 0.1:
        snr_str = '_snr_0.1'

    # Initialize Dataset
    print("Initializing dataset...")
    if args.data_type == 'real':
        # Create the RealDataset instance first
        dataset = RealDataset(invert_data=~args.uninvert_data, max_particles=args.max_particles, big_dataset=args.big_dataset, partial_prior=args.partial_prior, set90dist=args.set90dist)
    elif args.data_type == 'all_random':
        dataset = StarfileDataLoader(img_sz, path_to_starfile='/fs01/datasets/empiar/vatpase_synth' + snr_str, input_starfile='all_random.star', invert_hand=False)
        print("Using all_random dataset")
    elif args.data_type == 'all_correct':
        dataset = StarfileDataLoader(img_sz, path_to_starfile='/fs01/datasets/empiar/vatpase_synth' + snr_str, input_starfile='all_correct.star', invert_hand=False)
        print("Using all_correct dataset")
    elif args.data_type == 'random_membrane':
        dataset = StarfileDataLoader(img_sz, path_to_starfile='/fs01/datasets/empiar/vatpase_synth' + snr_str, input_starfile='random_membrane.star', invert_hand=False)
        print("Using random_membrane dataset")
    elif args.data_type == 'random_membrane_w_wobble':
        dataset = StarfileDataLoader(img_sz, path_to_starfile='/fs01/datasets/empiar/vatpase_synth' + snr_str, input_starfile='random_membrane_w_wobble.star', invert_hand=False)
        print("Using random_membrane_w_wobble dataset")
    elif args.data_type == 'correct_mem_w_wobble':
        dataset = StarfileDataLoader(img_sz, path_to_starfile='/fs01/datasets/empiar/vatpase_synth' + snr_str, input_starfile='correct_mem_w_wobble.star', invert_hand=False)
        print("Using correct_mem_w_wobble dataset")
    else:
        raise ValueError(f"Unsupported data_type: {args.data_type}")

    # --- Create DataLoader --- 
    dataloader = DataLoader(
        dataset, 
        shuffle=True, 
        batch_size=B, 
        pin_memory=True,
        num_workers=num_workers, 
        drop_last=False,
        persistent_workers=True,  # Keep workers alive between iterations
        prefetch_factor=3         # Number of batches loaded in advance by each worker
    )

    # build the model
    heads = args.heads
    model = CryoMEM(
                    heads=heads, 
                    sidelen=img_sz, 
                    num_octaves=4,
                    hartley=args.space == 'hartley',
                    data_type=args.data_type,
                    amortized_method=args.amortized_method,
                    ctf_params=dataset.ctf_params,
                    real_file_1=dataset.extract_path,
                    real_file_2=dataset.passthrough_path
                    )

    resolution = model.ctf.apix
    print(f"Resolution: {resolution}")
    
    model.cuda()

    init_epoch = 0
    total_time = 0

    # pose amortized inference and reconstruction
    epochs_amortized = args.epochs_amortized
    if epochs_amortized > 0:
        # decoder optimizer
        decoder_params = [{'params': model.pred_map.parameters(), 'lr': args.decoder_lr}]
        decoder_optim = torch.optim.Adam(decoder_params)

        # encoder optimizer
        encoder_params = [{'params': model.cnn_encoder.parameters(), 'lr': args.encoder_lr}]
        if model.amortized_method not in ('gt', 'none'):
            if model.amortized_method == 's2s2':
                encoder_params.extend([
                    {'params': model.s2s2_regressor.parameters(), 'lr': args.s2s2_lr}
                ])
            else:
                encoder_params.extend([
                    {'params': model.inmembrane_regressor.parameters(), 'lr': args.in_membrane_lr}
                ])
                if model.amortized_method == 'in_membrane_and_tilt':
                    encoder_params.extend([
                        {'params': model.tilt_regressor.parameters(), 'lr': args.tilt_lr}
                    ])
                elif model.amortized_method == 'split_reg':
                    encoder_params.extend([
                        {'params': model.inplane_residual_regressor.parameters(), 'lr': args.inplane_lr},
                        {'params': model.tilt_residual_regressor.parameters(), 'lr': args.tilt_lr}
                    ])
        encoder_optim = torch.optim.Adam(encoder_params)

        r = args.r
        
        # # Run inference on the entire dataset in evaluation mode without gradient computations.
        # # Track predicted rotations over epochs for plotting trends
        model.eval()

        pred_euler_over_epochs = []
        global_aligned_pred_euler_over_epochs = []


        scaler = GradScaler()

        with torch.no_grad():
            # Collect predicted, ground truth, and initial rotations
            pred_rotmats = np.zeros((len(dataset), 3, 3))
            gt_rotmats = np.zeros((len(dataset), 3, 3))
            init_rotmats = np.zeros((len(dataset), 3, 3))

            # Initialize GPU dictionaries for storing rotations
            gt_rotmats_gpu = {}
            init_rotmats_gpu = {}

            for data in tqdm(dataloader, desc="Epoch -1 evaluation"):
                data_cuda = dict2cuda(data)
                idxs = data_cuda['idx'].detach().cpu().numpy()
                out_dict = model(data_cuda, r=r, amortized=True)
                
                # Compute head losses and determine the chosen rotation per sample
                head_loss = compute_amortized_loss(data_cuda, out_dict)
                batch_min_head_idx = torch.argmin(head_loss, dim=1).cpu().numpy()
                B = len(idxs)
                
                # Save raw predictions, ground truth, and initial rotations
                rotmats_batch = out_dict['rotmat'].detach().cpu().numpy().reshape(B, -1, 3, 3)

                # Store ground truth and initial rotations
                for i, idx in enumerate(idxs):
                    pred_rotmats[idx] = rotmats_batch[i, batch_min_head_idx[i]]
                    gt_rotmats_gpu[idx] = data_cuda['gt_rots'][i]
                    init_rotmats_gpu[idx] = data_cuda['init_rots'][i]
                    gt_rotmats[idx] = data_cuda['gt_rots'][i].detach().cpu().numpy()
                    init_rotmats[idx] = data_cuda['init_rots'][i].detach().cpu().numpy()
            
            # Save the -1th volume
            vol = model.pred_map.make_volume(r=r)
            filename = os.path.join(os.path.join(save_path, 'reconst_volumes'), f'ep_{-1}.mrc')
            save_mrc(filename, vol, voxel_size=resolution, header_origin=None)
            
            # Perform global alignment on all collected predictions at once
            global_aligned_pred_rotmats, _, _, _ = global_alignment(pred_rotmats, gt_rotmats)

            # Convert all rotation matrices to normalized Euler angles (degrees, ZYZ, wrapped)
            pred_euler = convert_rotmats_to_euler_degrees(pred_rotmats)
            global_aligned_euler = convert_rotmats_to_euler_degrees(global_aligned_pred_rotmats)
            gt_euler = convert_rotmats_to_euler_degrees(gt_rotmats)
            init_euler = convert_rotmats_to_euler_degrees(init_rotmats)
            
            # Store first epoch predictions
            pred_euler_over_epochs.append(pred_euler)
            global_aligned_pred_euler_over_epochs.append(global_aligned_euler)
            
            # Plot initial distributions
            plot_orientation_distribution(pred_euler, -1, path_dict['init_viz'], tag='pred')
            plot_orientation_distribution(global_aligned_euler, -1, path_dict['init_viz'], tag='aligned')
            plot_orientation_distribution(gt_euler, -1, path_dict['init_viz'], tag='gt')
            plot_orientation_distribution(init_euler, -1, path_dict['init_viz'], tag='initial')

            # Plot error histograms
            plot_error_histograms(global_aligned_euler, gt_euler, -1, path_dict['init_viz'], tag='aligned')
            # Compare initial predicted Euler angles to initial Euler angles from data
            plot_error_histograms(pred_euler, init_euler, -1, path_dict['init_viz'], tag='init_vs_pred')

            # Plot error scatter plots
            plot_pred_vs_gt_scatter(pred_euler, gt_euler, -1, path_dict['init_viz'], tag='unaligned')
            plot_pred_vs_gt_scatter(global_aligned_euler, gt_euler, -1, path_dict['init_viz'], tag='aligned')

            inplane_errs = np.abs(global_aligned_euler[:, 0] - gt_euler[:, 0])
            tilt_errs    = np.abs(global_aligned_euler[:, 1] - gt_euler[:, 1])
            inmem_errs   = np.abs(global_aligned_euler[:, 2] - gt_euler[:, 2])
            writer.add_scalar('RMS_Rotation_Error', np.sqrt(np.mean(inplane_errs**2)), -1)
            writer.add_scalar('RMS_Tilt_Error', np.sqrt(np.mean(tilt_errs**2)), -1)
            writer.add_scalar('RMS_Inmem_Error', np.sqrt(np.mean(inmem_errs**2)), -1)

        

        model.train()
        for iteration in range(epochs_amortized):
            t1 = time.time()
            model.train()
            avg_loss = 0
            num_batches = len(dataloader) # Get total number of batches
            if args.use_softmin:
                T = temperature_schedule(iteration, hold_epochs=args.temperature_hold_epochs,
                                        total_epochs=epochs_amortized,
                                        T_start=args.init_temperature, T_end=args.end_temp)
                writer.add_scalar('Temperature/SoftWTA', T, iteration)
            
            for batch, data in enumerate(dataloader):
                encoder_optim.zero_grad()
                decoder_optim.zero_grad()
                data_cuda = dict2cuda(data)
                
                with autocast():
                    out_dict = model(data_cuda, r=r, amortized=True, current_epoch=iteration)
                    if args.use_softmin:
                        loss, per_head_loss, _, _ = soft_head_loss(   # if you switched to the new version
                            fproj_pred=out_dict['pred_fproj'],
                            fproj_true=data_cuda['fproj'],
                            mask=out_dict['mask'],
                            temperature=T
                        )
                        scaler.scale(loss).backward()
                    else:
                        batch_head_loss = compute_amortized_loss(
                            data_cuda, out_dict
                        )
                        loss = batch_head_loss.min(dim=1)[0].mean()
                        scaler.scale(loss).backward()

                scaler.step(encoder_optim)
                scaler.step(decoder_optim)
                scaler.update()
                loss_val = loss.detach()  # Store value without gradient tracking
                writer.add_scalar('Loss/Amortized_Batch', loss_val, iteration * num_batches + batch)
                avg_loss += loss.item()
            t2 = time.time()
            avg_loss /= num_batches
            # Rename average loss tag for clarity
            writer.add_scalar('Loss/Amortized_Epoch_Avg', avg_loss, iteration)

            # evaluation
            with torch.no_grad():
                total_time += t2 - t1
                writer.add_scalar('time', total_time, iteration)
                model.eval()

                vol = model.pred_map.make_volume(r=r)
                filename = os.path.join(path_dict['reconst_volumes'], f'ep_{iteration}.mrc')
                save_mrc(filename, vol, voxel_size=resolution, header_origin=None)

                pred_rotmats = np.zeros((len(dataset), 3, 3))

                head_win_counts = np.zeros(heads, dtype=np.int64)
                total_weights = np.zeros(heads, dtype=np.float32)
                total_samples = 0

                for data in tqdm(dataloader, desc=f"Epoch {iteration} evaluation"):
                    data_cuda = dict2cuda(data)
                    idxs = data_cuda['idx'].detach().cpu().numpy()
                    out_dict = model(data_cuda, r=r, amortized=True, current_epoch=iteration)

                    B = len(idxs)
                    rotmats_batch = out_dict['rotmat'].reshape(B, -1, 3, 3)
                    
                    if args.use_softmin:
                        batch_indices = torch.arange(B, device=rotmats_batch.device)
                        # --- Softmin-based evaluation ---
                        loss, per_head_loss, weights, loss_per_sample = soft_head_loss(
                            fproj_pred=out_dict['pred_fproj'],
                            fproj_true=data_cuda['fproj'],
                            mask=out_dict['mask'],
                            temperature=T
                        )
                        weights_np = weights.detach().cpu().numpy()
                        winning_indices = np.argmax(weights_np, axis=1)

                        # Store rotation of softmax winner (argmax)
                        selected_rotmats = rotmats_batch[batch_indices, winning_indices]
                        pred_rotmats[idxs] = selected_rotmats.detach().cpu().numpy()

                        # Track win counts and total weights
                        for wi in winning_indices:
                            head_win_counts[wi] += 1
                        total_weights += weights_np.sum(axis=0)

                    else:
                        # --- WTA-based evaluation ---
                        # Get per-head loss matrix and indices directly from the function
                        batch_head_loss = compute_amortized_loss(data_cuda, out_dict)  # shape [B, M]

                        batch_min_head_idx = torch.argmin(batch_head_loss, dim=1).cpu().numpy()
                        B = len(idxs)
                        rotmats_batch = out_dict['rotmat'].detach().cpu().numpy().reshape(B, -1, 3, 3)

                        # Track WTA win counts
                        for wi in batch_min_head_idx:
                            head_win_counts[wi] += 1
                        total_weights += torch.ones_like(head_loss).sum(dim=0).cpu().numpy()

                        for i, idx in enumerate(idxs):
                            # ← use idx, not idxs
                            pred_rotmats[idx] = rotmats_batch[i, batch_min_head_idx[i]]
                    
                    total_samples += B

                # Perform global alignment on all collected predictions
                flat = pred_rotmats.reshape(len(dataset), -1)              # shape (N, 9)
                n_unique = np.unique(flat, axis=0).shape[0]
                print("unique rotations in pred_rotmats:", n_unique, "of", len(dataset))
                global_aligned_pred_rotmats, _, _, _ = global_alignment(pred_rotmats, gt_rotmats)
                
                # Convert rots to normalized euler angles for plotting
                pred_euler = convert_rotmats_to_euler_degrees(pred_rotmats)
                global_aligned_euler = convert_rotmats_to_euler_degrees(global_aligned_pred_rotmats)
                # gt_euler and init_euler are already converted before the loop
                
                # Store this epoch's predictions
                pred_euler_over_epochs.append(pred_euler.copy())
                global_aligned_pred_euler_over_epochs.append(global_aligned_euler.copy())
                
                # Plot orientation distribution using original predictions
                plot_orientation_distribution(pred_euler, iteration, path_dict['orientation_distributions'], tag='pred')
                plot_orientation_distribution(global_aligned_euler, iteration, path_dict['orientation_distributions'], tag='aligned')
                
                # Use globally aligned predictions for GT comparisons and original predictions for initial comparisons
                plot_error_histograms(global_aligned_euler, gt_euler, iteration, path_dict['error_histograms'], tag='gt')
                # Compare current predicted Euler angles to initial Euler angles from data
                init_euler_epoch = convert_rotmats_to_euler_degrees(init_rotmats) # Need to convert init rots for this epoch
                plot_error_histograms(pred_euler, init_euler_epoch, iteration, path_dict['error_histograms'], tag='init_vs_pred')
                plot_pred_vs_gt_scatter(pred_euler, gt_euler, iteration, path_dict['pred_vs_gt'], tag='unaligned')
                plot_pred_vs_gt_scatter(global_aligned_euler, gt_euler, iteration, path_dict['pred_vs_gt'], tag='aligned')

                # Head stats
                plot_head_win_frequency(head_win_counts, iteration, path_dict['head_stats'])

                # save model and optimizer states
                if iteration == epochs_amortized - 1:
                    checkpoint = { 
                        'epoch': iteration,
                        'total_time': total_time,
                        'model': model.state_dict(),
                        'optimizer': decoder_optim.state_dict()
                    }
                    torch.save(checkpoint, os.path.join(path_dict['ckpt'], f'ep_{iteration}.pth'))

        # Plot angle differences after amortized training using appropriate comparisons
        # gt_euler and init_euler were converted once before the loop
        plot_angle_differences(global_aligned_pred_euler_over_epochs, gt_euler, save_path, tag='gt')
        plot_angle_differences(pred_euler_over_epochs, init_euler, save_path, tag='initial')
        plot_phi_angle_differences(global_aligned_pred_euler_over_epochs, gt_euler, save_path, tag='gt')
        plot_phi_angle_differences(pred_euler_over_epochs, init_euler, save_path, tag='initial')

    else:
        print("No amortized training")
        assert args.path_to_checkpoint != "none"
        assert args.epochs_unamortized > 0
        init_ckpt_path = args.path_to_checkpoint
        checkpoint = torch.load(init_ckpt_path)
        model.load_state_dict(checkpoint['model'])
        init_epoch = checkpoint['epoch']
        total_time = checkpoint['total_time']

        # Using new hyperparameters from the config file for unamortized training
        # (optimizer states are not loaded, so values like learning rates, vol_iters, pose_iters,
        # etc. are updated from the new config file)
        print(f"Resuming unamortized training from epoch {init_epoch} using checkpoint: {init_ckpt_path}")
        parser = argparse.ArgumentParser()
        parser.add_argument('--save_path', type=str)
        parser.add_argument('--config', type=str)
        known_args, remaining_args = parser.parse_known_args()


    init_epoch = epochs_amortized

    epochs_unamortized = args.epochs_unamortized
    if epochs_unamortized > 0:
        r = args.r
        vol_iters = args.vol_iters
        pose_iters = args.pose_iters

        rotmats = torch.zeros((len(dataset), 3, 3)).cuda()
        # Initialize scaler before training loop
        scaler = GradScaler()
        model.eval()
        
        print("Using refined rotations from amortized training...")
        # Use the best poses from amortized predictions
        with torch.no_grad():
            for batch, data in tqdm(enumerate(dataloader), total=len(dataloader)):
                data_cuda = dict2cuda(data)
                idxs = data_cuda['idx']
                out_dict = model(data_cuda, r=r, amortized=True)
                fproj_input = data_cuda['fproj']
                fproj_pred = out_dict['pred_fproj']
                mask = out_dict['mask']
                
                # Compute losses and get best rotation indices
                batch_head_loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum((-1, -2))) / mask.sum((-1, -2))
                batch_min_head_idx = torch.argmin(batch_head_loss, dim=1)
                
                # Safely reshape and index the rotation matrices
                rotmats_batch = out_dict['rotmat']
                B, M = batch_head_loss.shape  # B is batch size, M is number of rotations
                rotmats_batch = rotmats_batch.view(B, M, 3, 3)
                
                # Use gather to safely select the best rotation for each sample
                batch_indices = torch.arange(B, device=rotmats_batch.device)
                selected_rotmats = rotmats_batch[batch_indices, batch_min_head_idx]
                
                # Safely assign to the output tensor
                rotmats[idxs] = selected_rotmats

        model.train()

        # Define optimizer learning rates based on args
        unamortized_decoder_lr = args.decoder_lr if hasattr(args, 'unamortized_decoder_lr') else args.decoder_lr
        unamortized_rot_lr = args.rot_lr if hasattr(args, 'unamortized_rot_lr') else args.rot_lr

        # decoder optimizer
        print(f"Unamortized decoder learning rate: {unamortized_decoder_lr}")
        decoder_params = [{'params': model.pred_map.parameters(), 'lr': unamortized_decoder_lr}]
        decoder_optim = torch.optim.Adam(decoder_params)

        # pose model
        print(f"Unamortized pose learning rate: {unamortized_rot_lr}")
        pose_model = PoseModel(n_data=len(dataset), rotations=rotmats, euler=True)
        pose_model.cuda()
        params = [{'params': list(filter(lambda p: p.requires_grad, pose_model.rots.parameters())), 
                   'lr': unamortized_rot_lr,
                   'betas': (0.9, 0.99)}]
        pose_optim = torch.optim.Adam(params)

        pred_euler_over_epochs_unamortized = []
        global_aligned_euler_over_epochs_unamortized = []
        # Also need ground truth and initial Euler angles (convert once)
        gt_euler = np.zeros((len(dataset), 3))
        gt_rotmats = np.zeros((len(dataset), 3, 3))
        init_euler = np.zeros((len(dataset), 3))
        init_rotmats = np.zeros((len(dataset), 3, 3))
        with torch.no_grad():
            for data in dataloader: # Need a pass to get all gt/init angles
                data_cuda = dict2cuda(data)
                idxs = data_cuda['idx'].cpu().numpy()
                gt_euler[idxs] = convert_rotmats_to_euler_degrees(data_cuda['gt_rots'])
                gt_rotmats[idxs] = data_cuda['gt_rots'].cpu().numpy()
                init_euler[idxs] = convert_rotmats_to_euler_degrees(data_cuda['init_rots'])
                init_rotmats[idxs] = data_cuda['init_rots'].cpu().numpy()

        # unamortized training
        global_step = init_epoch * len(dataloader) * (args.vol_iters + args.pose_iters) # Initialize global step counter
        for iteration in range(epochs_unamortized):
            print(f"Unamortized epoch {iteration}/{epochs_unamortized}")
            avg_loss = 0
            t1 = time.time()
            
            pred_rotmats_epoch = np.zeros((len(dataset), 3, 3))

            # Calculate total steps per epoch for TensorBoard logging
            num_batches = len(dataloader)
            total_steps_per_epoch = num_batches * (args.vol_iters + args.pose_iters)
            
            # Use enumerate to get batch index
            for batch_idx, data in tqdm(enumerate(dataloader), desc="Optimizing poses and volume", total=num_batches):
                data_cuda = dict2cuda(data)
                idxs = data_cuda['idx']
                idxs_np = idxs.cpu().numpy()

                # --- Pose optimization ---
                for param in model.pred_map.parameters():
                    param.requires_grad = False
                for param in pose_model.parameters():
                    param.requires_grad = True

                for pose_opt_idx in range(args.pose_iters):
                    current_step = global_step + batch_idx * (args.vol_iters + args.pose_iters) + pose_opt_idx
                    pose_optim.zero_grad()
                    decoder_optim.zero_grad()
                    pred_rotmat_batch = pose_model(idxs)
                    out_dict = model(data_cuda, amortized=False, pred_rotmat=pred_rotmat_batch, r=r)
                    
                    # Compute reconstruction loss
                    fproj_input = data_cuda['fproj']
                    fproj_pred = out_dict['pred_fproj']
                    mask = out_dict['mask']
                    batch_loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum((-1, -2))) / mask.sum((-1, -2))
                    loss = batch_loss.mean()
                    
                    # Backpropagate loss and update pose model
                    scaler.scale(loss).backward()
                    scaler.step(pose_optim)
                    scaler.update()
                    pose_model.update(idxs)

                    loss_val = loss.detach()  # Store value without gradient tracking
                    # Log pose optimization step loss
                    writer.add_scalar('Loss/Unamortized_Pose_Step', loss_val, current_step)
                    avg_loss += loss.item()

                # --- Volume optimization ---
                for param in model.pred_map.parameters():
                    param.requires_grad = True
                for param in pose_model.parameters():
                    param.requires_grad = False

                for vol_opt_idx in range(args.vol_iters):
                    current_step = global_step + batch_idx * (args.vol_iters + args.pose_iters) + args.pose_iters + vol_opt_idx
                    decoder_optim.zero_grad()
                    pred_rotmat = pose_model(torch.tensor(idxs).cuda())
                    out_dict = model(data_cuda, amortized=False, pred_rotmat=pred_rotmat, r=r)
                    
                    # Compute reconstruction loss
                    fproj_input = data_cuda['fproj']
                    fproj_pred = out_dict['pred_fproj']
                    mask = out_dict['mask']
                    batch_loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum((-1, -2))) / mask.sum((-1, -2))
                    loss = batch_loss.mean()
                    
                    # Backpropagate loss and update volume model
                    scaler.scale(loss).backward()
                    scaler.step(decoder_optim)
                    scaler.update()

                    # Log volume optimization step loss
                    loss_val = loss.detach()
                    writer.add_scalar('Loss/Unamortized_Volume_Step', loss_val, current_step)

                # Store the final rotation matrices for this batch *after* updates
                with torch.no_grad():
                    final_pred_rotmat_batch = pose_model(idxs)
                    pred_rotmats_epoch[idxs_np] = final_pred_rotmat_batch.detach().cpu().numpy()
                    # Also store GT and Init rots for this batch (needed for alignment)

                # Print loss every 4 batches (using the last calculated loss for simplicity)
                if batch_idx % 10 == 0:
                    print(f'Epoch {iteration + init_epoch}, Batch {batch_idx}/{num_batches}, last batch loss: {loss.item():.3f}')

            # --- Post-Epoch Processing & Plotting ---
            t2 = time.time()
            total_time += t2 - t1
            current_epoch = iteration + init_epoch
            global_step += len(dataloader) * (args.vol_iters + args.pose_iters) # Update global step after epoch

            writer.add_scalar('time', total_time, current_epoch)

            global_aligned_pred_rotmats_epoch, _, _, _ = global_alignment(pred_rotmats_epoch, gt_rotmats)

            pred_euler = convert_rotmats_to_euler_degrees(pred_rotmats_epoch)
            global_aligned_euler = convert_rotmats_to_euler_degrees(global_aligned_pred_rotmats_epoch)

            # Store this epoch's predictions
            pred_euler_over_epochs_unamortized.append(pred_euler.copy())
            global_aligned_euler_over_epochs_unamortized.append(global_aligned_euler.copy())
            
            # Plot orientation distribution using original predictions
            plot_orientation_distribution(pred_euler, current_epoch, path_dict['orientation_distributions'], tag='pred_unamortized')
            plot_orientation_distribution(global_aligned_euler, current_epoch, path_dict['orientation_distributions'], tag='aligned_unamortized')
            
            # Use globally aligned predictions for GT comparisons and original predictions for initial comparisons
            plot_error_histograms(global_aligned_euler, gt_euler, current_epoch, path_dict['error_histograms'], tag='gt_unamortized')
            # Compare current epoch's predicted Euler angles to initial Euler angles
            plot_error_histograms(pred_euler, init_euler, current_epoch, path_dict['error_histograms'], tag='init_vs_pred_unamortized')
            plot_pred_vs_gt_scatter(pred_euler, gt_euler, current_epoch, path_dict['pred_vs_gt'], tag='unaligned_unamortized')
            plot_pred_vs_gt_scatter(global_aligned_euler, gt_euler, current_epoch, path_dict['pred_vs_gt'], tag='aligned_unamortized')

            # Calculate and log RMS errors for the epoch
            inplane_errs = np.abs(global_aligned_euler[:, 0] - gt_euler[:, 0])
            tilt_errs    = np.abs(global_aligned_euler[:, 1] - gt_euler[:, 1])
            inmem_errs   = np.abs(global_aligned_euler[:, 2] - gt_euler[:, 2])
            # Wrap errors based on angle ranges: [-180, 180] for psi/phi, [0, 180] for tilt
            inplane_errs = np.minimum(inplane_errs, 360.0 - inplane_errs) 
            # Tilt error is absolute difference in [0, 180], no wrapping needed for abs error
            inmem_errs   = np.minimum(inmem_errs, 360.0 - inmem_errs)
            writer.add_scalar('RMS_Rotation_Error', np.sqrt(np.mean(inplane_errs**2)), current_epoch)
            writer.add_scalar('RMS_Tilt_Error', np.sqrt(np.mean(tilt_errs**2)), current_epoch)
            writer.add_scalar('RMS_Inmem_Error', np.sqrt(np.mean(inmem_errs**2)), current_epoch)

            avg_loss /= (len(dataloader) * args.pose_iters) # Normalize avg loss correctly by number of pose steps
            writer.add_scalar('Loss/Unamortized_Epoch_Avg', avg_loss, current_epoch)

            # Evaluation and Saving Volume/Checkpoint
            with torch.no_grad():
                model.eval() # Set model to eval mode for volume generation
                vol = model.pred_map.make_volume(r=r)
                filename = os.path.join(path_dict['reconst_volumes'], f'ep_{iteration + init_epoch}.mrc') # Use correct path variable
                save_mrc(filename, vol, voxel_size=resolution, header_origin=None)
                model.train() # Set back to train mode


        # --- Final Trend Plotting (after all unamortized epochs) ---
        if epochs_unamortized > 0:
            # gt_euler and init_euler are already computed and normalized
            plot_angle_differences(global_aligned_euler_over_epochs_unamortized, gt_euler, save_path, tag='gt_unamortized')
            plot_angle_differences(pred_euler_over_epochs_unamortized, init_euler, save_path, tag='initial_unamortized')
            plot_phi_angle_differences(global_aligned_euler_over_epochs_unamortized, gt_euler, save_path, tag='gt_unamortized')
            plot_phi_angle_differences(pred_euler_over_epochs_unamortized, init_euler, save_path, tag='initial_unamortized')

        writer.close()