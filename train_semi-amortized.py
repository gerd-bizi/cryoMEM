import argparse
import time
import yaml
import sys
import numpy as np
import torch
import os
import healpy as hp
import matplotlib.pyplot as plt

from utils.mrc import save_mrc
from utils.rot_calc_error import compute_rot_error_single

from dataset import RealDataset
from torch.utils.data import DataLoader

from pose_models import PoseModel
from multihypo_models import CryoSAPIENCE
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


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
    parser.add_argument('--data', type=str, choices=['hsp', 'spike', 'spliceosome', 'empiar10028'], default=config.get('data', 'hsp'))
    parser.add_argument('--batch_size', type=int, default=config.get('batch_size', 32))
    parser.add_argument('--r', type=float, default=config.get('r', 1.))
    parser.add_argument('--uninvert_data', action='store_true', default=config.get('uninvert_data', False))
    parser.add_argument('--prior', type=str, choices=['true', 'false'], default=config.get('prior', 'false'))

    # amortized regime
    parser.add_argument('--epochs_amortized', type=int, default=config.get('epochs_amortized', 10))
    parser.add_argument('--num_rotations', type=int, default=config.get('num_rotations', 7))
    parser.add_argument('--encoder_lr', type=float, default=config.get('encoder_lr', 0.0001))
    parser.add_argument('--space', type=str, choices=['fourier', 'hartley'], default=config.get('space', 'hartley'))
    parser.add_argument('--decoder_lr', type=float, default=config.get('decoder_lr', 0.005))
    
    # unamortized regime
    parser.add_argument('--epochs_unamortized', type=int, default=config.get('epochs_unamortized', 10))
    parser.add_argument('--unamortized_decoder_lr', type=float, default=config.get('unamortized_decoder_lr', 0.002))
    parser.add_argument('--rot_lr', type=float, default=config.get('rot_lr', 0.05))
    parser.add_argument('--vol_iters', type=int, default=config.get('vol_iters', 1))
    parser.add_argument('--pose_iters', type=int, default=config.get('pose_iters', 5))
    parser.add_argument('--path_to_checkpoint', type=str, default=config.get('path_to_checkpoint', 'none'))
    
    args = parser.parse_args()
    return args


def plot_orientation_distribution(rotmats, iteration, save_dir):
    if isinstance(rotmats, torch.Tensor):
        rotmats = rotmats.detach().cpu().numpy()
    
    print(f"Input rotation matrices shape: {rotmats.shape}")
    assert len(rotmats.shape) == 3 and rotmats.shape[1:] == (3, 3), f"Expected shape (N, 3, 3), got {rotmats.shape}"
    
    view_dirs = rotmats[:, :, 2]  # Z-axis directions
    print(f"Number of particles: {len(view_dirs)}")
    
    # Increase HEALPix resolution
    nside = 8
    npix = hp.nside2npix(nside)
    
    # Map view directions to HEALPix cells
    pix = hp.vec2pix(nside, view_dirs[:, 0], view_dirs[:, 1], view_dirs[:, 2])
    
    # Count rotations in each cell
    counts = np.array([np.sum(pix == i) for i in range(npix)])
    
    # Print some statistics to verify we're getting all rotations
    print(f"Total number of rotations: {len(rotmats)}")
    print(f"Number of non-zero cells: {np.sum(counts > 0)}")
    print(f"Max rotations in a cell: {np.max(counts)}")
    
    # Create visualization
    plt.figure(figsize=(15, 7))
    
    # Mollweide projection
    plt.subplot(121)
    hp.mollview(map=counts, title=f'Epoch {iteration}: Mollweide view', 
                cmap='viridis', hold=True, flip='geo')
    hp.graticule()
    
    # Cartesian projection
    plt.subplot(122)
    hp.cartview(map=counts, title=f'Epoch {iteration}: Cartesian view', 
                cmap='viridis', hold=True, flip='geo')
    hp.graticule()
    
    # Save plot
    save_file = os.path.join(save_dir, f'dist_ep_{iteration}.png')
    plt.savefig(save_file)
    plt.close()


if __name__ == "__main__":
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

    hartley = args.space == 'hartley'
    writer = SummaryWriter(os.path.join(save_path, 'tbd'))

    # load data
    B = args.batch_size
    num_workers = 8
    img_sz = 128
    if args.data == 'empiar10028':
        dataset = RealDataset(invert_data=~args.uninvert_data)
    else:
        all_resolutions = {'hsp': 1.5, '80S': 3.77, 'spliceosome': 4.33, 'spike': 2.13, 'J3365': 1.03 * 500 / 128}
        data = args.data
        resolution = all_resolutions[data]
        path_to_starfile = f'./synthetic_data/{data}'
        # strf = 'data.star'
        # dataset = StarfileDataLoader(img_sz, path_to_starfile, strf, invert_hand=False)
        dataset = RealDataset()
    dataloader = DataLoader(dataset, shuffle=True, batch_size=B, pin_memory=True,
                            num_workers=num_workers, drop_last=False)
    
    hartley = args.space == 'hartley'
    prior = args.prior == 'true'

    # build the model
    num_rotations = args.num_rotations
    model = CryoSAPIENCE(
                    num_rotations=num_rotations, 
                    sidelen=img_sz, 
                    num_octaves=4,
                    hartley=hartley,
                    experimental=True,
                    use_prior=prior)
    resolution = model.ctf.apix
    # import pdb; pdb.set_trace()
    model.cuda()

    epochs_amortized = args.epochs_amortized
    experimental = True

    # Create directories
    reconst_volume_paths = os.path.join(save_path, 'reconst_volumes')
    orientation_dist_paths = os.path.join(save_path, 'orientation_distributions')
    ckpt_path = os.path.join(save_path, 'ckpt')
    
    for path in [reconst_volume_paths, orientation_dist_paths, ckpt_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    # pose amortized inference and reconstruction
    if epochs_amortized > 0:
        # create paths
        reconst_volume_paths = os.path.join(save_path, 'reconst_volumes')
        if not os.path.exists(reconst_volume_paths):
            os.makedirs(reconst_volume_paths)
        ckpt_path = os.path.join(save_path, 'ckpt')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)

        # decoder optimizer
        decoder_params = [{'params': model.pred_map.parameters(), 'lr': args.decoder_lr}]
        decoder_optim = torch.optim.Adam(decoder_params)

        # encoder optimizer
        encoder_params = [
            {'params': model.cnn_encoder.parameters(), 'lr': args.encoder_lr},
            {'params': model.known_angle_regressor.parameters(), 'lr': args.encoder_lr},
            {'params': model.phi_regressor.parameters(), 'lr': args.encoder_lr}
        ]
        encoder_optim = torch.optim.Adam(encoder_params)

        # amortized training
        r = args.r
        total_time = 0
        for iteration in range(epochs_amortized):
            t1 = time.time()
            model.train()
            avg_loss = 0
            for batch, data in enumerate(dataloader):
                encoder_optim.zero_grad()
                decoder_optim.zero_grad()
                data_cuda = dict2cuda(data)
                out_dict = model(data_cuda, r=r, amortized=True)
                fproj_input = data_cuda['fproj'] # B x 1 x H x W
                fproj_pred = out_dict['pred_fproj'] # B x M x H x W
                mask = out_dict['mask']
                batch_head_loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum((-1, -2))) / mask.sum((-1, -2)) # B x M
                min_argmin = torch.min(batch_head_loss, 1)
                batch_min_head_loss = min_argmin[0] # B
                loss = batch_min_head_loss.mean()
                loss.backward()
                encoder_optim.step()
                decoder_optim.step()
                if batch % 100 == 0:
                    print(f'Epoch {iteration}, Batch {batch}, loss: {loss.item():.3f}')
                avg_loss += loss.item()
            t2 = time.time()
            avg_loss /= len(dataloader)
            writer.add_scalar('avg_loss', avg_loss, iteration)

            # evaluation
            with torch.no_grad():
                # store reconstructed volume
                total_time += t2 - t1
                writer.add_scalar('time', total_time, iteration)
                model.eval()
                vol = model.pred_map.make_volume(r=r)
                filename = os.path.join(reconst_volume_paths, f'ep_{iteration}.mrc')
                save_mrc(filename, vol, voxel_size=resolution, header_origin=None)
                
                # Collect rotations for orientation distribution plotting
                pred_rotmats = np.zeros((len(dataset), 3, 3))
                for data in tqdm(dataloader):
                    data_cuda = dict2cuda(data)
                    idxs = data_cuda['idx'].detach().cpu().numpy()
                    out_dict = model(data_cuda, r=r, amortized=True)
                    fproj_input = data_cuda['fproj']
                    fproj_pred = out_dict['pred_fproj']
                    mask = out_dict['mask']
                    
                    batch_head_loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum((-1, -2))) / mask.sum((-1, -2))
                    batch_min_head_idx = torch.argmin(batch_head_loss, dim=1).cpu().numpy()
                    
                    rotmats_batch = out_dict['rotmat'].detach().cpu().numpy()
                    batch_size = len(idxs)
                    rotmats_batch = rotmats_batch.reshape(batch_size, -1, 3, 3)
                    
                    for i, idx in enumerate(idxs):
                        pred_rotmats[idx] = rotmats_batch[i, batch_min_head_idx[i]]
                
                # Plot orientation distribution
                plot_orientation_distribution(pred_rotmats, iteration, orientation_dist_paths)

                # save model and optimizer states
                checkpoint = { 
                    'epoch': iteration,
                    'total_time': total_time,
                    'model': model.state_dict(),
                    'optimizer': decoder_optim.state_dict()}
                # torch.save(checkpoint, os.path.join(ckpt_path, f'ep_{iteration}.pth'))
    else:
        assert args.path_to_checkpoint != "none"
        assert args.epochs_unamortized > 0
        init_ckpt_path = args.path_to_checkpoint
        checkpoint = torch.load(init_ckpt_path)
        model.load_state_dict(checkpoint['model'])
        init_epoch = checkpoint['epoch']
        total_time = checkpoint['total_time']

    init_epoch = epochs_amortized

    epochs_unamortized = args.epochs_unamortized
    if epochs_unamortized > 0:
        r = args.r
        vol_iters = args.vol_iters
        pose_iters = args.pose_iters

        # create paths
        ckpt_path = os.path.join(save_path, 'ckpt')
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        reconst_volume_paths = os.path.join(save_path, 'reconst_volumes')
        if not os.path.exists(reconst_volume_paths):
            os.makedirs(reconst_volume_paths)

        # select the best pose based on reconstruction loss
        rotmats = torch.zeros((len(dataset), 3, 3)).cuda()
        model.eval()
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
                pred_rotmats = out_dict['rotmat']
                B, M = batch_head_loss.shape  # B is batch size, M is number of rotations
                pred_rotmats = pred_rotmats.view(B, M, 3, 3)
                
                # Use gather to safely select the best rotation for each sample
                batch_indices = torch.arange(B, device=pred_rotmats.device)
                selected_rotmats = pred_rotmats[batch_indices, batch_min_head_idx]
                
                # Safely assign to the output tensor
                rotmats[idxs] = selected_rotmats


            if (not experimental) and (epochs_amortized == 0): # add initial results if not directly continuing from amortized training
                # Collect rotations for plotting
                pred_rotmats = np.zeros((len(dataset), 3, 3))
                for data in tqdm(dataloader):
                    data_cuda = dict2cuda(data)
                    idxs = data_cuda['idx'].detach().cpu().numpy()
                    out_dict = model(data_cuda, r=r, amortized=True)
                    fproj_input = data_cuda['fproj']
                    fproj_pred = out_dict['pred_fproj']
                    mask = out_dict['mask']
                    head_loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum((-1, -2))) / mask.sum((-1, -2))
                    argmin = np.argmin(head_loss.detach().cpu().numpy(), axis=1)
                    pred_rotmats[idxs] = out_dict['rotmat'].detach().cpu().numpy()[np.arange(len(idxs)), argmin]
                
                # Plot orientation distribution
                plot_orientation_distribution(pred_rotmats, init_epoch, orientation_dist_paths)
                
                # Commented out GT-dependent code
                # if not experimental:
                #     gt_rotmats = ...  # Remove GT-related code
                #     rot_errors_dict = compute_rot_error_single(...)
                #     ...  # Remove metric logging
        model.train()

        # decoder optimizer
        decoder_params = [{'params': model.pred_map.parameters(), 'lr': args.unamortized_decoder_lr}]
        decoder_optim = torch.optim.Adam(decoder_params)

        # pose model
        pose_model = PoseModel(n_data=len(dataset), rotations=rotmats)
        pose_model.cuda()
        params = [{'params': list(filter(lambda p: p.requires_grad, pose_model.rots.parameters())), 
                       'lr': args.rot_lr, 'betas': (0.9, 0.9)}]
        pose_optim = torch.optim.Adam(params)

        # unamortized training
        for iteration in range(epochs_unamortized):
            avg_loss = 0
            t1 = time.time()
            for batch, data in enumerate(dataloader):
                data_cuda = dict2cuda(data)
                idxs = data_cuda['idx']

                # only optimize pose
                for param in model.pred_map.parameters():
                    param.requires_grad = False
                for param in pose_model.parameters():
                    param.requires_grad = True
                for _ in range(pose_iters):
                    decoder_optim.zero_grad()
                    pose_optim.zero_grad()
                    pred_rotmat = pose_model(idxs)
                    out_dict = model(data_cuda, amortized=False, pred_rotmat=pred_rotmat, r=r)
                    fproj_input = data_cuda['fproj']
                    fproj_pred = out_dict['pred_fproj']
                    mask = out_dict['mask']
                    batch_loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum((-1, -2))) / mask.sum((-1, -2))
                    loss = batch_loss.mean()
                    loss.backward()
                    pose_optim.step()
                    decoder_optim.zero_grad()
                pose_model.update(idxs)

                # only optimize volume
                for param in model.pred_map.parameters():
                    param.requires_grad = True
                for param in pose_model.parameters():
                    param.requires_grad = False
                for _ in range(vol_iters):
                    decoder_optim.zero_grad()
                    pose_optim.zero_grad()
                    pred_rotmat = pose_model(idxs)
                    out_dict = model(data_cuda, amortized=False, pred_rotmat=pred_rotmat, r=r)
                    fproj_input = data_cuda['fproj']
                    fproj_pred = out_dict['pred_fproj']
                    mask = out_dict['mask']
                    batch_loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum((-1, -2))) / mask.sum((-1, -2))
                    loss = batch_loss.mean()
                    loss.backward()
                    decoder_optim.step()
                    pose_optim.zero_grad()
                
                if batch % 100 == 0:
                    print(f'Epoch {iteration}, Batch {batch}, loss: {loss.item():.3f}')
                avg_loss += loss.item()
            t2 = time.time()
            avg_loss /= len(dataloader)
            writer.add_scalar('avg_loss', avg_loss, iteration + 1 + init_epoch)

            # evaluation
            with torch.no_grad():
                total_time += t2 - t1
                writer.add_scalar('time', total_time, iteration + 1 + init_epoch)
                # generate volume
                vol = model.pred_map.make_volume(r=r)
                filename = os.path.join(reconst_volume_paths, f'ep_{iteration + init_epoch}.mrc')
                save_mrc(filename, vol, voxel_size=resolution, header_origin=None)
                
                # Collect rotations for plotting
                pred_rotmats = np.zeros((len(dataset), 3, 3))
                for data in tqdm(dataloader):
                    data_cuda = dict2cuda(data)
                    idxs = data_cuda['idx'].detach().cpu().numpy()
                    out_dict = model(data_cuda, r=r, amortized=True)
                    fproj_input = data_cuda['fproj']
                    fproj_pred = out_dict['pred_fproj']
                    mask = out_dict['mask']
                    
                    # Move computations to CPU and ensure proper shapes
                    batch_head_loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum((-1, -2))) / mask.sum((-1, -2))
                    batch_min_head_idx = torch.argmin(batch_head_loss, dim=1).cpu().numpy()
                    
                    # Move rotations to CPU and reshape
                    rotmats_batch = out_dict['rotmat'].detach().cpu().numpy()
                    batch_size = len(idxs)
                    rotmats_batch = rotmats_batch.reshape(batch_size, -1, 3, 3)
                    
                    # Select best rotation for each sample
                    for i, idx in enumerate(idxs):
                        pred_rotmats[idx] = rotmats_batch[i, batch_min_head_idx[i]]
                
                # Plot orientation distribution
                plot_orientation_distribution(pred_rotmats, iteration + init_epoch, orientation_dist_paths)
                
                # Commented out GT-dependent code
                # if not experimental:
                #     gt_rotmats = ...  # Remove GT-related code
                #     rot_errors_dict = compute_rot_error_single(...)
                #     ...  # Remove metric logging

                # save model and optimizer states
                checkpoint = { 
                    'epoch': iteration + init_epoch + 1,
                    'total_time': total_time,
                    'model': model.state_dict(),
                    'pose_model': pose_model.state_dict(),
                    'pose_optimizer': pose_optim.state_dict(),
                    'decoder_optimizer': decoder_optim.state_dict()}
                # torch.save(checkpoint, os.path.join(ckpt_path, f'ep_{iteration + init_epoch + 1}.pth'))
    writer.close()