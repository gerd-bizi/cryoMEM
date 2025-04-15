import argparse
import time
import yaml
import sys
import numpy as np
import torch
import os
import healpy as hp
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils.mrc import save_mrc
from utils.rot_calc_error import compute_rot_error_single, global_alignment

from dataset import RealDataset, StarfileDataLoader
from torch.utils.data import DataLoader

from pose_models import PoseModel
from multihypo_models import CryoSAPIENCE
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pytorch3d.transforms


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
    
    # pre-unamortized regime (partial Euler angle optimization)
    parser.add_argument('--epochs_pre_unamortized', type=int, default=config.get('epochs_pre_unamortized', 0))
    parser.add_argument('--pre_unamortized_decoder_lr', type=float, default=config.get('pre_unamortized_decoder_lr', 0.003))
    parser.add_argument('--pre_unamortized_rot_lr', type=float, default=config.get('pre_unamortized_rot_lr', 0.02))
    parser.add_argument('--optimizable_angles', nargs='+', type=lambda x: x.lower() == 'true', 
                        default=config.get('optimizable_angles', [True, True, False]),
                        help='Which Euler angles to optimize in pre-unamortized regime (in-plane, tilt, in-membrane)')
    
    # unamortized regime
    parser.add_argument('--epochs_unamortized', type=int, default=config.get('epochs_unamortized', 10))
    parser.add_argument('--unamortized_decoder_lr', type=float, default=config.get('unamortized_decoder_lr', 0.002))
    parser.add_argument('--rot_lr', type=float, default=config.get('rot_lr', 0.05))
    parser.add_argument('--vol_iters', type=int, default=config.get('vol_iters', 1))
    parser.add_argument('--pose_iters', type=int, default=config.get('pose_iters', 5))
    parser.add_argument('--path_to_checkpoint', type=str, default=config.get('path_to_checkpoint', 'none'))
    
    parser.add_argument('--data_type', type=str, choices=['all_correct', 'all_random', 'random_membrane', 'random_membrane_w_wobble', 'correct_mem_w_wobble'], default=config.get('data_type', 'all_correct'), help='Specify whether to use real or synthetic data.')
    parser.add_argument('--amortized_method', type=str, choices=['s2s2', 'euler', 'split_reg', 'none'], default=config.get('amortized_method', 's2s2'), help='Select the amortized inference method.')
    parser.add_argument('--w1', type=float, default=config.get('w1', 0.0))
    parser.add_argument('--w2', type=float, default=config.get('w2', 0.0))
    parser.add_argument('--pose_lr', type=float, default=config.get('pose_lr', 0.01), help='Learning rate for pose optimization in unamortized regime')
    parser.add_argument('--update_hyperparams', action='store_true', default=config.get('update_hyperparams', False), help='If set, update training hyperparameters from new config even when resuming from checkpoint')

    
    args = parser.parse_args()
    return args


def plot_orientation_distribution(rotmats, iteration, save_dir):
    """
    Plots Cartesian projections for each pair of Euler angles derived from the input rotation matrices.
    Three graphs are produced (with the following comparisons):
      • rlnAngleTilt vs rlnAngleRot (tilt on x-axis)
      • rlnAngleRot vs rlnAnglePsi (x-axis is rlnAngleRot)
      • rlnAngleTilt vs rlnAnglePsi (tilt on x-axis)
    Each subplot includes properly labeled axes with the corresponding "rln..." names, a title, and a colorbar
    (using the red–blue 'RdBu' colormap) that shows the scale along with the min and max counts.
    """
    if isinstance(rotmats, torch.Tensor):
        rotmats = rotmats.detach().cpu().numpy()
    
    # Convert rotation matrices to Euler angles using a ZYZ convention.
    euler_angles = pytorch3d.transforms.matrix_to_euler_angles(torch.tensor(rotmats), convention="ZYZ").numpy()  # shape (N, 3)
    #psi tilt rot
    
    # Convert angles to degrees.
    angles_deg = np.degrees(euler_angles)
    # Remap the first and third angles to range -180 to 180, second (tilt) remains in [0, 180]
    angles_deg[:, 0] = ((angles_deg[:, 0] + 180) % 360) - 180
    angles_deg[:, 2] = ((angles_deg[:, 2] + 180) % 360) - 180
    angles_deg[:, 1] = np.clip(angles_deg[:, 1], 0, 180)
    
    # Define names corresponding to the angles from your STAR-file reading.
    angle_names = ["in-plane", "tilt", "in-membrane"]
    
    # Define desired pairs.
    pairs = [(0, 1), (2, 1), (0, 2)]
    
    # Set up the subplots.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (i, j) in zip(axs, pairs):
        x = angles_deg[:, i]
        y = angles_deg[:, j]
        # Set axis limits based on angle type:
        xlim = (0, 180) if angle_names[i] == "tilt" else (-180, 180)
        ylim = (0, 180) if angle_names[j] == "tilt" else (-180, 180)
        
        hist, xedges, yedges, im = ax.hist2d(x, y, bins=60, range=[xlim, ylim], cmap="RdBu")
        ax.set_xlabel(f"{angle_names[i]} (deg)")
        ax.set_ylabel(f"{angle_names[j]} (deg)")
        ax.set_title(f"{angle_names[i]} vs {angle_names[j]}")
        
        # Add a colorbar to each subplot.
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Count")
        # Overlay a small text box with the minimum and maximum histogram counts.
        ax.text(0.05, 0.95, f"min: {int(hist.min())}, max: {int(hist.max())}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    
    plt.suptitle(f"Orientation Distribution at Epoch {iteration}")
    save_file = os.path.join(save_dir, f'dist_cartesian_ep_{iteration}_rln.png')
    plt.savefig(save_file)
    plt.close()


def compute_amortized_loss(data_cuda, out_dict, psi_penalty_weight=0.0, theta_penalty_weight=0.0):
    """
    Computes the amortized loss as the average minimum head loss plus a penalty on psi and theta differences.

    Parameters:
        data_cuda: dictionary containing input tensors; must include key 'fproj' (B x 1 x H x W)
        out_dict: dictionary from the model containing keys 'pred_fproj', 'mask', and 'delta_angles'.
        psi_penalty_weight: weight for the psi penalty term.
        theta_penalty_weight: weight for the theta penalty term.

    Returns:
        loss: the computed loss value
    """
    # Reconstruction loss: Calculate the per-head loss
    fproj_input = data_cuda['fproj']  # shape: [B, 1, H, W]
    fproj_pred = out_dict['pred_fproj']  # shape: [B, M, H, W]
    mask = out_dict['mask']              # shape: [B, M, H, W]
    batch_head_loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum(dim=(-1, -2))) / (mask.sum(dim=(-1, -2)) + 1e-8)  # shape: [B, M]
    
    # # Regularization: Add L2 penalties per head
    # psi_diff = out_dict['init_psi_diff']  # expected shape: [B, M]
    # theta_diff = out_dict['init_theta_diff']  # expected shape: [B, M]
    # psi_penalty = psi_penalty_weight * psi_diff.pow(2)
    # theta_penalty = theta_penalty_weight * theta_diff.pow(2)
    
    # total_loss = batch_head_loss + psi_penalty + theta_penalty  # shape: [B, M]
    return batch_head_loss

# Update function signatures to take rotation matrices directly
def plot_angle_differences(pred_rotmats_over_epochs, gt_rotmats, save_dir, tag=''):
    """
    Plots average angle differences over epochs given a list of predicted rotation matrices.
    Computes errors directly from rotation matrices at plotting time.
    
    Parameters:
    -----------
    pred_rotmats_over_epochs: list of numpy arrays
        List of predicted rotation matrices for each epoch, shape (N, 3, 3) each
    gt_rotmats: numpy array
        Ground truth rotation matrices, shape (N, 3, 3)
    save_dir: str
        Directory to save the plot
    tag: str
        Additional tag for the filename
    """
    epochs = range(len(pred_rotmats_over_epochs))
    avg_psi_errors = []
    avg_theta_errors = []
    
    for epoch_rotmats in pred_rotmats_over_epochs:
        # Convert to euler angles using ZYZ convention
        pred_euler = pytorch3d.transforms.matrix_to_euler_angles(torch.tensor(epoch_rotmats), convention="ZYZ").numpy()
        gt_euler = pytorch3d.transforms.matrix_to_euler_angles(torch.tensor(gt_rotmats), convention="ZYZ").numpy()
        
        # Calculate angle differences (psi and theta)
        psi_diff = np.abs(np.degrees(pred_euler[:, 0]) - np.degrees(gt_euler[:, 0]))
        psi_diff = np.minimum(psi_diff, 360 - psi_diff)  # Account for periodic boundary
        
        theta_diff = np.abs(np.degrees(pred_euler[:, 1]) - np.degrees(gt_euler[:, 1]))
        theta_diff = np.minimum(theta_diff, 180 - theta_diff)  # Account for periodic boundary
        
        avg_psi_errors.append(np.mean(psi_diff))
        avg_theta_errors.append(np.mean(theta_diff))
    
    plt.figure()
    plt.plot(epochs, avg_psi_errors, marker='o', label='in-plane absolute error')
    plt.plot(epochs, avg_theta_errors, marker='o', label='tilt absolute error')
    plt.xlabel('Epoch')
    plt.ylabel('Absolute error (degrees)')
    plt.title(f'Angle differences over training ({tag})')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'angle_differences_{tag}.png'))
    plt.close()

def plot_phi_angle_differences(pred_rotmats_over_epochs, gt_rotmats, save_dir, tag=''):
    """
    Plots average phi (in-membrane) angle differences over epochs directly from rotation matrices.
    
    Parameters are same as plot_angle_differences.
    """
    epochs = range(len(pred_rotmats_over_epochs))
    avg_phi_errors = []
    
    for epoch_rotmats in pred_rotmats_over_epochs:
        # Convert to euler angles using ZYZ convention
        pred_euler = pytorch3d.transforms.matrix_to_euler_angles(torch.tensor(epoch_rotmats), convention="ZYZ").numpy()
        gt_euler = pytorch3d.transforms.matrix_to_euler_angles(torch.tensor(gt_rotmats), convention="ZYZ").numpy()
        
        # Calculate phi differences and account for periodicity
        phi_diff = np.abs(np.degrees(pred_euler[:, 2]) - np.degrees(gt_euler[:, 2]))
        phi_diff = np.minimum(phi_diff, 360 - phi_diff)
        
        avg_phi_errors.append(np.mean(phi_diff))
    
    plt.figure()
    plt.plot(epochs, avg_phi_errors, marker='o', label='in-membrane absolute error')
    plt.xlabel('Epoch')
    plt.ylabel('Absolute error (degrees)')
    plt.title(f'φ angle differences over training ({tag})')
    plt.legend()
    plt.savefig(os.path.join(save_dir, f'phi_angle_differences_{tag}.png'))
    plt.close()

def plot_error_histograms(pred_rotmats, gt_rotmats, epoch, save_dir, tag=''):
    """
    Plots histograms of the angle errors computed directly from rotation matrices.
    """
    # Convert to euler angles using ZYZ convention
    pred_euler = pytorch3d.transforms.matrix_to_euler_angles(torch.tensor(pred_rotmats), convention="ZYZ").numpy()
    gt_euler = pytorch3d.transforms.matrix_to_euler_angles(torch.tensor(gt_rotmats), convention="ZYZ").numpy()

    
    # Calculate angle differences
    psi_deg = np.degrees(pred_euler[:, 0]) - np.degrees(gt_euler[:, 0])
    theta_deg = np.degrees(pred_euler[:, 1]) - np.degrees(gt_euler[:, 1])
    phi_deg = np.degrees(pred_euler[:, 2]) - np.degrees(gt_euler[:, 2])
    
    # Adjust for periodic boundaries
    psi_deg = ((psi_deg + 180) % 360) - 180
    phi_deg = ((phi_deg + 180) % 360) - 180
    theta_deg = ((theta_deg + 90) % 180) - 90
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[2].hist(psi_deg, bins=30, color='red', alpha=0.7)
    axs[2].set_title(f'Epoch {epoch} rlnAnglePsi Error')
    axs[2].set_xlabel('Error (deg)')
    axs[2].set_ylabel('Frequency')    
    
    axs[1].hist(theta_deg, bins=30, color='green', alpha=0.7)
    axs[1].set_title(f'Epoch {epoch} rlnAngleTilt Error')
    axs[1].set_xlabel('Error (deg)')
    axs[1].set_ylabel('Frequency')

    axs[0].hist(phi_deg, bins=30, color='blue', alpha=0.7)
    axs[0].set_title(f'Epoch {epoch} rlnAngleRot Error')
    axs[0].set_xlabel('Error (deg)')
    axs[0].set_ylabel('Frequency')

    
    fig.suptitle(f'epoch_{epoch}_angle_errors', fontsize=16)
    
    if not os.path.exists(os.path.join(save_dir, 'angle_errors')):
        os.makedirs(os.path.join(save_dir, 'angle_errors'))
    plt.savefig(os.path.join(save_dir, 'angle_errors', f'epoch_{epoch}_angle_errors_{tag}.png'))
    plt.close()

def plot_error_scatter(pred_rotmats, gt_rotmats, epoch, save_dir, tag=''):
    """
    Generates scatter plots comparing the angle errors between angles directly from rotation matrices.
    """
    # Convert to euler angles using ZYZ convention
    pred_euler = pytorch3d.transforms.matrix_to_euler_angles(torch.tensor(pred_rotmats), convention="ZYZ").numpy()
    gt_euler = pytorch3d.transforms.matrix_to_euler_angles(torch.tensor(gt_rotmats), convention="ZYZ").numpy()
    
    # Calculate angle differences
    psi_deg = np.degrees(pred_euler[:, 0] - gt_euler[:, 0])
    theta_deg = np.degrees(pred_euler[:, 1] - gt_euler[:, 1])
    phi_deg = np.degrees(pred_euler[:, 2] - gt_euler[:, 2])
    
    # Adjust for periodic boundaries
    psi_deg = ((psi_deg + 180) % 360) - 180
    phi_deg = ((phi_deg + 180) % 360) - 180
    theta_deg = ((theta_deg + 90) % 180) - 90
    
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    axs[0].scatter(psi_deg, theta_deg, alpha=0.5)
    axs[0].set_xlabel('in-plane error (deg)')
    axs[0].set_ylabel('tilt error (deg)')
    axs[0].set_title('tilt vs in-plane Error Scatter')
    
    axs[1].scatter(psi_deg, phi_deg, alpha=0.5)
    axs[1].set_xlabel('in-plane Error (deg)')
    axs[1].set_ylabel('in-membrane Error (deg)')
    axs[1].set_title('in-membrane vs in-plane Error Scatter')
    
    axs[2].scatter(theta_deg, phi_deg, alpha=0.5)
    axs[2].set_xlabel('tilt Error (deg)')
    axs[2].set_ylabel('in-membrane Error (deg)')
    axs[2].set_title('in-membrane vs tilt Error Scatter')
    
    fig.suptitle(f'epoch_{epoch}_error_scatter', fontsize=16)
    
    if not os.path.exists(os.path.join(save_dir, 'error_scatter')):
        os.makedirs(os.path.join(save_dir, 'error_scatter'))
    plt.savefig(os.path.join(save_dir, 'error_scatter', f'epoch_{epoch}_error_scatter_{tag}.png'))
    plt.close()

def plot_pred_vs_gt_scatter(pred_rotmats, gt_rotmats, iteration, save_dir, tag=''):
    """
    Plots scatter plots comparing predicted and ground-truth Euler angles
    for each of the three angles: in-plane, tilt, and in-membrane.
    Uses a 2D histogram with a color gradient to show point density.
    
    Parameters
    ----------
    pred_rotmats : numpy.ndarray of shape (N, 3, 3)
        Predicted rotation matrices.
    gt_rotmats : numpy.ndarray of shape (N, 3, 3)
        Ground truth rotation matrices.
    iteration : int
        Current epoch/iteration number.
    save_dir : str
        Directory where the plot image should be saved.
    tag : str
        Additional tag for the output filename.
    """

    # Convert the rotation matrices to Euler angles using the ZYZ convention.
    pred_t = torch.tensor(pred_rotmats)
    gt_t = torch.tensor(gt_rotmats)
    pred_euler = pytorch3d.transforms.matrix_to_euler_angles(pred_t, convention="ZYZ").numpy()
    gt_euler = pytorch3d.transforms.matrix_to_euler_angles(gt_t, convention="ZYZ").numpy()

    # Convert angles to degrees.
    pred_deg = np.degrees(pred_euler)
    gt_deg = np.degrees(gt_euler)

    # Adjust the first and third angles to be in the range [-180, 180] and clip the second (tilt) to [0, 180].
    pred_deg[:, 0] = ((pred_deg[:, 0] + 180) % 360) - 180
    pred_deg[:, 2] = ((pred_deg[:, 2] + 180) % 360) - 180
    pred_deg[:, 1] = np.clip(pred_deg[:, 1], 0, 180)

    gt_deg[:, 0] = ((gt_deg[:, 0] + 180) % 360) - 180
    gt_deg[:, 2] = ((gt_deg[:, 2] + 180) % 360) - 180
    gt_deg[:, 1] = np.clip(gt_deg[:, 1], 0, 180)

    angle_names = ["in-plane", "tilt", "in-membrane"]

    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for i, ax in enumerate(axs):
        # Create 2D histogram with more bins for finer resolution
        if angle_names[i] == "tilt":
            xlim = (0, 180)
            ylim = (0, 180)
            bins = 72
        else:
            xlim = (-180, 180)
            ylim = (-180, 180)
            bins = 72
        
        hist, xedges, yedges, im = ax.hist2d(pred_deg[:, i], gt_deg[:, i], 
                                            range=[xlim, ylim], 
                                            bins=bins, cmap='viridis', norm=LogNorm())
        
        ax.set_xlabel(f'Predicted {angle_names[i]} (deg)')
        ax.set_ylabel(f'Ground Truth {angle_names[i]} (deg)')
        ax.set_title(f'Predicted vs Ground Truth {angle_names[i]}')
        
        # Add colorbar
        cb = fig.colorbar(im, ax=ax)
        cb.set_label('Count')
        
        # Add y=x reference line
        lims = ax.get_xlim()
        ax.plot(lims, lims, 'r--', linewidth=0.5)
        
        # Add text box with min/max counts
        ax.text(0.05, 0.95, f"min: {int(hist.min())}, max: {int(hist.max())}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Add tag to title if provided
    title = f"Predicted vs Ground Truth Euler Angles at Epoch {iteration}"
    if tag:
        title += f" ({tag})"
    fig.suptitle(title, fontsize=16)
    
    out_dir = os.path.join(save_dir, 'pred_vs_gt')
    os.makedirs(out_dir, exist_ok=True)
    
    # Fix filename construction to handle empty tags
    if tag:
        outfile = os.path.join(out_dir, f'pred_vs_gt_scatter_ep_{iteration}_{tag}.png')
    else:
        outfile = os.path.join(out_dir, f'pred_vs_gt_scatter_ep_{iteration}.png')
        
    plt.savefig(outfile, dpi=300)  # Increased DPI for better resolution
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

    # Create directories
    reconst_volume_paths = os.path.join(save_path, 'reconst_volumes')
    orientation_dist_paths = os.path.join(save_path, 'orientation_distributions')
    pred_vs_gt_path = os.path.join(save_path, 'pred_vs_gt')
    ckpt_path = os.path.join(save_path, 'ckpt')
    
    for path in [reconst_volume_paths, orientation_dist_paths, pred_vs_gt_path, ckpt_path]:
        if not os.path.exists(path):
            os.makedirs(path)
        elif path == pred_vs_gt_path:
            # Clean up any existing pred_vs_gt plots to avoid confusion
            for old_file in os.listdir(path):
                if old_file.startswith('pred_vs_gt_scatter_ep_'):
                    os.remove(os.path.join(path, old_file))
                    
    # load data
    B = args.batch_size
    num_workers = 8
    img_sz = 128

    # if args.data_type == 'real':
    #     resolution = 1.03 * 500 / 128
    #     dataset = RealDataset(invert_data=~args.uninvert_data)
    # else:
    #     resolution = 1.03 * 192 / 128
    #     dataset = StarfileDataLoader(img_sz, invert_hand=False)

    if args.data_type == 'all_random':
        dataset = StarfileDataLoader(img_sz, path_to_starfile='/fs01/datasets/empiar/vatpase_synth_restricted', input_starfile='all_random.star', invert_hand=False)
        print("Using all_random dataset")
    elif args.data_type == 'all_correct':
        dataset = StarfileDataLoader(img_sz, path_to_starfile='/fs01/datasets/empiar/vatpase_synth_restricted', input_starfile='all_correct.star', invert_hand=False)
        print("Using all_correct dataset")
    elif args.data_type == 'random_membrane':
        dataset = StarfileDataLoader(img_sz, path_to_starfile='/fs01/datasets/empiar/vatpase_synth_restricted', input_starfile='random_membrane.star', invert_hand=False)
        print("Using random_membrane dataset")
    elif args.data_type == 'random_membrane_w_wobble':
        dataset = StarfileDataLoader(img_sz, path_to_starfile='/fs01/datasets/empiar/vatpase_synth_restricted', input_starfile='random_membrane_w_wobble.star', invert_hand=False)
        print("Using random_membrane_w_wobble dataset")
    elif args.data_type == 'correct_mem_w_wobble':
        dataset = StarfileDataLoader(img_sz, path_to_starfile='/fs01/datasets/empiar/vatpase_synth_restricted', input_starfile='correct_mem_w_wobble.star', invert_hand=False)
        print("Using correct_mem_w_wobble dataset")
    elif args.data_type == 'real':
        dataset = RealDataset(invert_data=~args.uninvert_data)


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
                    vol_rep='explicit',
                    data_type=args.data_type,
                    amortized_method=args.amortized_method,
                    ctf_params=dataset.ctf_params)
    resolution = model.ctf.apix
    print(f"Resolution: {resolution}")
    # import pdb; pdb.set_trace()
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
        if model.amortized_method == 'split_reg' or model.amortized_method == 'none':
            encoder_params = [
                {'params': model.cnn_encoder.parameters(), 'lr': args.encoder_lr},
                # {'params': model.known_angle_regressor.parameters(), 'lr': args.encoder_lr},
                # {'params': model.residual_angle_regressor.parameters(), 'lr': args.encoder_lr},
                {'params': model.gated_residual_angle_regressor_rot.parameters(), 'lr': args.encoder_lr},
                {'params': model.gated_residual_angle_regressor_tilt.parameters(), 'lr': args.encoder_lr},
                {'params': model.phi_regressor.parameters(), 'lr': args.encoder_lr}
            ]
        elif model.amortized_method == 's2s2' or model.amortized_method == 'penalty':
            encoder_params = [
                {'params': model.cnn_encoder.parameters(), 'lr': args.encoder_lr},
                {'params': model.orientation_regressor.parameters(), 'lr': args.encoder_lr}
            ]
        elif model.amortized_method == 'euler' or model.amortized_method == 'penalty_euler':
            encoder_params = [
                {'params': model.cnn_encoder.parameters(), 'lr': args.encoder_lr},
                {'params': model.euler_regressor.parameters(), 'lr': args.encoder_lr}
            ]
        encoder_optim = torch.optim.Adam(encoder_params)

        # amortized training
        r = args.r
        
        # Run inference on the entire dataset in evaluation mode without gradient computations.
        model.eval()
        # Track predicted rotations over epochs for plotting trends
        pred_rotmats_over_epochs = []
        global_aligned_pred_rotmats_over_epochs = []
        
        # Evaluate the initial model (epoch 0)
        print("Evaluating initial model (epoch -1)...")
        # Collect predicted, ground truth, and initial rotations
        pred_rotmats = np.zeros((len(dataset), 3, 3))
        gt_rotmats = np.zeros((len(dataset), 3, 3))
        init_rotmats = np.zeros((len(dataset), 3, 3))
        
        for data in tqdm(dataloader, desc="Epoch -1 evaluation"):
            data_cuda = dict2cuda(data)
            idxs = data_cuda['idx'].detach().cpu().numpy()
            out_dict = model(data_cuda, r=r, amortized=True)
            
            # Compute head losses and determine the chosen rotation per sample
            head_loss = compute_amortized_loss(data_cuda, out_dict,
                                              psi_penalty_weight=args.w1,
                                              theta_penalty_weight=args.w2)
            batch_min_head_idx = torch.argmin(head_loss, dim=1).cpu().numpy()
            B = len(idxs)
            
            # Save raw predictions, ground truth, and initial rotations
            rotmats_batch = out_dict['rotmat'].detach().cpu().numpy().reshape(B, -1, 3, 3)
            
            for i, idx in enumerate(idxs):
                pred_rotmats[idx] = rotmats_batch[i, batch_min_head_idx[i]]
                gt_rotmats[idx] = data_cuda['gt_rots'][i].detach().cpu().numpy()
                init_rotmats[idx] = data_cuda['init_rots'][i].detach().cpu().numpy()
        
        # Perform global alignment on all collected predictions at once
        global_aligned_pred_rotmats, _, _, _ = global_alignment(pred_rotmats, gt_rotmats)
        
        # Store first epoch predictions
        pred_rotmats_over_epochs.append(pred_rotmats.copy())
        global_aligned_pred_rotmats_over_epochs.append(global_aligned_pred_rotmats.copy())
        
        # Plot initial distributions - use original predictions for orientation distribution
        plot_orientation_distribution(pred_rotmats, -1, orientation_dist_paths)
        
        # Use globally aligned predictions for GT comparisons and original predictions for initial comparisons
        plot_error_histograms(global_aligned_pred_rotmats, gt_rotmats, -1, save_path, tag='gt')
        plot_error_histograms(pred_rotmats, init_rotmats, -1, save_path, tag='initial')
        plot_error_scatter(global_aligned_pred_rotmats, gt_rotmats, -1, save_path, tag='gt')
        plot_error_scatter(pred_rotmats, init_rotmats, -1, save_path, tag='initial')
        plot_pred_vs_gt_scatter(pred_rotmats, gt_rotmats, -1, save_path, tag='unaligned')
        plot_pred_vs_gt_scatter(global_aligned_pred_rotmats, gt_rotmats, -1, save_path, tag='aligned')

        for iteration in range(epochs_amortized):
            t1 = time.time()
            model.train()
            avg_loss = 0
            for batch, data in enumerate(dataloader):
                encoder_optim.zero_grad()
                decoder_optim.zero_grad()
                data_cuda = dict2cuda(data)
                out_dict = model(data_cuda, r=r, amortized=True)
                head_loss = compute_amortized_loss(data_cuda, out_dict, psi_penalty_weight=args.w1, theta_penalty_weight=args.w2)

                loss = head_loss.min(dim=1)[0].mean()
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
                total_time += t2 - t1
                writer.add_scalar('time', total_time, iteration)
                model.eval()
                vol = model.pred_map.make_volume(r=r)
                filename = os.path.join(reconst_volume_paths, f'ep_{iteration}.mrc')
                save_mrc(filename, vol, voxel_size=resolution, header_origin=None)
                
                # Collect all predictions and ground truth after each epoch
                pred_rotmats = np.zeros((len(dataset), 3, 3))
                gt_rotmats = np.zeros((len(dataset), 3, 3))
                init_rotmats = np.zeros((len(dataset), 3, 3))
                
                for data in tqdm(dataloader, desc=f"Epoch {iteration} evaluation"):
                    data_cuda = dict2cuda(data)
                    idxs = data_cuda['idx'].detach().cpu().numpy()
                    out_dict = model(data_cuda, r=r, amortized=True)
                    
                    # Calculate head losses to select the best hypothesis per sample
                    head_loss = compute_amortized_loss(data_cuda, out_dict,
                                                      psi_penalty_weight=args.w1,
                                                      theta_penalty_weight=args.w2)
                    batch_min_head_idx = torch.argmin(head_loss, dim=1).cpu().numpy()
                    B = len(idxs)
                    
                    # Save all types of rotations from the multiple hypotheses
                    rotmats_batch = out_dict['rotmat'].detach().cpu().numpy().reshape(B, -1, 3, 3)
                    
                    for i, idx in enumerate(idxs):
                        pred_rotmats[idx] = rotmats_batch[i, batch_min_head_idx[i]]
                        gt_rotmats[idx] = data_cuda['gt_rots'][i].detach().cpu().numpy()
                        init_rotmats[idx] = data_cuda['init_rots'][i].detach().cpu().numpy()
                
                # Perform global alignment on all collected predictions
                global_aligned_pred_rotmats, _, _, _ = global_alignment(pred_rotmats, gt_rotmats)
                
                # Store this epoch's predictions
                pred_rotmats_over_epochs.append(pred_rotmats.copy())
                global_aligned_pred_rotmats_over_epochs.append(global_aligned_pred_rotmats.copy())
                
                # Plot orientation distribution using original predictions
                plot_orientation_distribution(pred_rotmats, iteration, orientation_dist_paths)
                
                # Use globally aligned predictions for GT comparisons and original predictions for initial comparisons
                plot_error_histograms(global_aligned_pred_rotmats, gt_rotmats, iteration, save_path, tag='gt')
                plot_error_histograms(pred_rotmats, init_rotmats, iteration, save_path, tag='initial')
                plot_error_scatter(global_aligned_pred_rotmats, gt_rotmats, iteration, save_path, tag='gt')
                plot_error_scatter(pred_rotmats, init_rotmats, iteration, save_path, tag='initial')
                plot_pred_vs_gt_scatter(pred_rotmats, gt_rotmats, iteration, save_path, tag='unaligned')
                plot_pred_vs_gt_scatter(global_aligned_pred_rotmats, gt_rotmats, iteration, save_path, tag='aligned')

                # save model and optimizer states
                checkpoint = { 
                    'epoch': iteration,
                    'total_time': total_time,
                    'model': model.state_dict(),
                    'optimizer': decoder_optim.state_dict()}
                if iteration == epochs_amortized - 1:
                    torch.save(checkpoint, os.path.join(ckpt_path, f'ep_{iteration}.pth'))

        # Plot angle differences after amortized training using appropriate comparisons
        plot_angle_differences(global_aligned_pred_rotmats_over_epochs, gt_rotmats, save_path, tag='gt')
        plot_angle_differences(pred_rotmats_over_epochs, init_rotmats, save_path, tag='initial')
        plot_phi_angle_differences(global_aligned_pred_rotmats_over_epochs, gt_rotmats, save_path, tag='gt')
        plot_phi_angle_differences(pred_rotmats_over_epochs, init_rotmats, save_path, tag='initial')

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

        # select the best pose based on reconstruction loss
        rotmats = torch.zeros((len(dataset), 3, 3)).cuda()
        model.eval()
        
        # Choose the source of initial rotations for unamortized training

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
                pred_rotmats = out_dict['rotmat']
                B, M = batch_head_loss.shape  # B is batch size, M is number of rotations
                pred_rotmats = pred_rotmats.view(B, M, 3, 3)
                
                # Use gather to safely select the best rotation for each sample
                batch_indices = torch.arange(B, device=pred_rotmats.device)
                selected_rotmats = pred_rotmats[batch_indices, batch_min_head_idx]
                
                # Safely assign to the output tensor
                rotmats[idxs] = selected_rotmats

        model.train()

        # decoder optimizer
        print(f"Unamortized decoder learning rate: {args.unamortized_decoder_lr}")
        decoder_params = [{'params': model.pred_map.parameters(), 'lr': args.unamortized_decoder_lr}]
        decoder_optim = torch.optim.Adam(decoder_params)

        # pose model
        print(f"Unamortized pose learning rate: {args.rot_lr}")
        pose_model = PoseModel(n_data=len(dataset), rotations=rotmats, euler=True)
        pose_model.cuda()
        # Optimization parameters for membrane proteins
        params = [{'params': list(filter(lambda p: p.requires_grad, pose_model.rots.parameters())), 
                'lr': args.rot_lr,
                'betas': (0.9, 0.99)}]  # Higher beta2 for adaptive learning
        pose_optim = torch.optim.Adam(params)

        # unamortized training
        for iteration in range(epochs_unamortized):
            print(f"Unamortized epoch {iteration+1}/{epochs_unamortized}")
            avg_loss = 0
            t1 = time.time()
            
            # Collect all types of rotations for plotting
            pred_rotmats = np.zeros((len(dataset), 3, 3))
            gt_rotmats = np.zeros((len(dataset), 3, 3))
            init_rotmats = np.zeros((len(dataset), 3, 3))
            
            # Perform alternating optimization of volume and poses
            for data in tqdm(dataloader, desc="Optimizing poses and volume"):
                data_cuda = dict2cuda(data)
                idxs = data_cuda['idx'].cpu().numpy()
                
                # Volume optimization (fixed poses)
                for _ in range(args.vol_iters):
                    decoder_optim.zero_grad()
                    pred_rotmat = pose_model(torch.tensor(idxs).cuda())
                    out_dict = model(data_cuda, amortized=False, pred_rotmat=pred_rotmat, r=r)
                    
                    # Compute reconstruction loss
                    fproj_input = data_cuda['fproj']
                    fproj_pred = out_dict['pred_fproj']
                    mask = out_dict['mask']
                    loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum((-1, -2))) / mask.sum((-1, -2))
                    loss = loss.mean()
                    avg_loss += loss.item()
                    loss.backward()
                    decoder_optim.step()
                
                # Pose optimization (fixed volume)
                for _ in range(args.pose_iters):
                    pose_optim.zero_grad()
                    pred_rotmat = pose_model(torch.tensor(idxs).cuda())
                    out_dict = model(data_cuda, amortized=False, pred_rotmat=pred_rotmat, r=r)
                    
                    # Compute reconstruction loss
                    fproj_input = data_cuda['fproj']
                    fproj_pred = out_dict['pred_fproj']
                    mask = out_dict['mask']
                    loss = ((torch.abs(fproj_pred - fproj_input) * mask).sum((-1, -2))) / mask.sum((-1, -2))
                    loss = loss.mean()
                    loss.backward()
                    pose_optim.step()
                
                # Apply the accumulated rotational updates
                pose_model.update(torch.tensor(idxs).cuda())
                
                # Store rotations for plotting
                pred_rotmats[idxs] = out_dict['rotmat'].detach().cpu().numpy()
                gt_rotmats[idxs] = data_cuda['gt_rots'].cpu().numpy()
                init_rotmats[idxs] = data_cuda['init_rots'].cpu().numpy()
            
            # Perform global alignment on all collected predictions
            global_aligned_pred_rotmats, _, _, _ = global_alignment(pred_rotmats, gt_rotmats)
            
            # Save this epoch's rotations if tracking over time
            if 'pred_rotmats_over_epochs_unamortized' not in locals():
                pred_rotmats_over_epochs_unamortized = []
                global_aligned_pred_rotmats_over_epochs_unamortized = []
            
            pred_rotmats_over_epochs_unamortized.append(pred_rotmats.copy())
            global_aligned_pred_rotmats_over_epochs_unamortized.append(global_aligned_pred_rotmats.copy())
            
            # Plot orientation distribution using original predictions
            plot_orientation_distribution(pred_rotmats, iteration + init_epoch, orientation_dist_paths)
            
            # Use globally aligned predictions for GT comparisons and original predictions for initial comparisons
            plot_error_histograms(global_aligned_pred_rotmats, gt_rotmats, iteration + init_epoch, save_path, tag='gt')
            plot_error_histograms(pred_rotmats, init_rotmats, iteration + init_epoch, save_path, tag='initial')
            plot_error_scatter(global_aligned_pred_rotmats, gt_rotmats, iteration + init_epoch, save_path, tag='gt')
            plot_error_scatter(pred_rotmats, init_rotmats, iteration + init_epoch, save_path, tag='initial')
            plot_pred_vs_gt_scatter(global_aligned_pred_rotmats, gt_rotmats, iteration + init_epoch, save_path, tag='aligned')
            plot_pred_vs_gt_scatter(pred_rotmats, gt_rotmats, iteration + init_epoch, save_path, tag='unaligned')
            
            # If last epoch, plot trends over time using appropriate comparisons
            if iteration == epochs_unamortized - 1:
                plot_angle_differences(global_aligned_pred_rotmats_over_epochs_unamortized, gt_rotmats, save_path, tag='gt_unamortized')
                plot_angle_differences(pred_rotmats_over_epochs_unamortized, init_rotmats, save_path, tag='initial_unamortized')
                plot_phi_angle_differences(global_aligned_pred_rotmats_over_epochs_unamortized, gt_rotmats, save_path, tag='gt_unamortized')
                plot_phi_angle_differences(pred_rotmats_over_epochs_unamortized, init_rotmats, save_path, tag='initial_unamortized')

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