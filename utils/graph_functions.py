import argparse
import time
import yaml
import sys
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from utils.mrc import save_mrc
from utils.rot_calc_error import compute_rot_error_single, global_alignment

from dataset import RealDataset, StarfileDataLoader
from torch.utils.data import DataLoader

from pose_models import PoseModel
from multihypo_models import CryoMEM
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import pytorch3d.transforms

def plot_orientation_distribution(euler_angles, iteration, save_dir, tag=''):
    """
    Plots Cartesian projections for each pair of Euler angles derived from the input rotation matrices.
    Three graphs are produced (with the following comparisons):
      • in-plane vs tilt
      • in-membrane vs tilt
      • in-plane vs in-membrane
    Two versions are generated: one using a grid (2D histogram) and one using hexagonal binning, with reduced bin counts.
    The grid version is saved in a subdirectory 'grid' and the hexbin version in 'hex' under save_dir.
    """
    if isinstance(euler_angles, torch.Tensor):
        euler_angles = euler_angles.detach().cpu().numpy()
    
    # Define names corresponding to the angles
    angle_names = ["In-Plane", "Tilt", "In-Membrane"]
    # Define desired pairs: (in-plane vs tilt), (in-membrane vs tilt), (in-plane vs in-membrane)
    pairs = [(0, 1), (2, 1), (0, 2)]
    
    # Grid plot version with reduced bin count (40 bins)
    grid_dir = os.path.join(save_dir, "grid")
    os.makedirs(grid_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (i, j) in zip(axs, pairs):
        x = euler_angles[:, i]
        y = euler_angles[:, j]
        xlim = (0, 180) if angle_names[i] == "Tilt" else (-180, 180)
        ylim = (0, 180) if angle_names[j] == "Tilt" else (-180, 180)
        # Compute 2D histogram with 40 bins
        hist, xedges, yedges = np.histogram2d(x, y, bins=40, range=[xlim, ylim])
        min_nonzero = np.min(hist[hist > 0]) if np.any(hist > 0) else 1
        max_value = np.max(hist)
        masked_hist = np.ma.masked_where(hist == 0, hist)
        im = ax.imshow(masked_hist.T, origin='lower', extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                         aspect='auto', cmap="viridis", norm=LogNorm(vmin=min_nonzero, vmax=max_value))
        ax.set_xlabel(f"{angle_names[i]} (deg)")
        ax.set_ylabel(f"{angle_names[j]} (deg)")
        ax.set_title(f"{angle_names[i]} vs {angle_names[j]}")
        if angle_names[i] == "Tilt":
            ax.set_xticks(np.arange(0, 181, 30))
        else:
            ax.set_xticks(np.arange(-180, 181, 30))
        if angle_names[j] == "Tilt":
            ax.set_yticks(np.arange(0, 181, 30))
        else:
            ax.set_yticks(np.arange(-180, 181, 30))
        cb = fig.colorbar(im, ax=ax)
        cb.set_label("Count")
        ax.text(0.05, 0.95, f"min: {int(min_nonzero)}, max: {int(max_value)}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.suptitle(f"Orientation Distribution at Epoch {iteration} (Grid)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    grid_file = os.path.join(grid_dir, f'dual_cartesian_ep_{iteration}.png')
    plt.savefig(grid_file)
    plt.close()
    
    # Hexbin plot version
    hex_dir = os.path.join(save_dir, "hex")
    os.makedirs(hex_dir, exist_ok=True)
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for ax, (i, j) in zip(axs, pairs):
        x = euler_angles[:, i]
        y = euler_angles[:, j]
        xlim = (0, 180) if angle_names[i] == "Tilt" else (-180, 180)
        ylim = (0, 180) if angle_names[j] == "Tilt" else (-180, 180)
        h = ax.hexbin(x, y, gridsize=40, extent=[xlim[0], xlim[1], ylim[0], ylim[1]], cmap="viridis")
        counts = h.get_array()
        min_nonzero = np.min(counts[counts > 0]) if counts.size > 0 and np.any(counts > 0) else 1
        max_value = np.max(counts)
        h.set_norm(LogNorm(vmin=min_nonzero, vmax=max_value))
        ax.set_xlabel(f"{angle_names[i]} (deg)")
        ax.set_ylabel(f"{angle_names[j]} (deg)")
        ax.set_title(f"{angle_names[i]} vs {angle_names[j]}")
        if angle_names[i] == "Tilt":
            ax.set_xticks(np.arange(0, 181, 30))
        else:
            ax.set_xticks(np.arange(-180, 181, 30))
        if angle_names[j] == "Tilt":
            ax.set_yticks(np.arange(0, 181, 30))
        else:
            ax.set_yticks(np.arange(-180, 181, 30))
        cb = fig.colorbar(h, ax=ax)
        cb.set_label("Count")
        ax.text(0.05, 0.95, f"min: {int(min_nonzero)}, max: {int(max_value)}",
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.suptitle(f"Orientation Distribution at Epoch {iteration} (Hexbin)")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    hex_file = os.path.join(hex_dir, f'dual_cartesian_ep_{iteration}_{tag}.png')
    plt.savefig(hex_file)
    plt.close()


# Update function signatures to take rotation matrices directly
def plot_angle_differences(euler_angles, euler_angles_gt, save_dir, tag=''):
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
    epochs = range(len(euler_angles))
    avg_psi_errors = []
    avg_theta_errors = []
    
    for i in range(len(euler_angles)):
        # Calculate angle differences (psi and theta), assuming input is already degrees
        current_epoch_angles = euler_angles[i] # Get the array for the current epoch
        psi_diff = np.abs(current_epoch_angles[:, 0] - euler_angles_gt[:, 0])
        psi_diff = np.minimum(psi_diff, 360 - psi_diff)  # Account for periodic boundary
        
        theta_diff = np.abs(current_epoch_angles[:, 1] - euler_angles_gt[:, 1])
        theta_diff = np.minimum(theta_diff, 180 - theta_diff)  # Account for periodic boundary
        
        avg_psi_errors.append(np.mean(psi_diff))
        avg_theta_errors.append(np.mean(theta_diff))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_psi_errors, marker='o', label='in-plane absolute error')
    plt.plot(epochs, avg_theta_errors, marker='o', label='tilt absolute error')
    plt.xlabel('Epoch')
    plt.ylabel('Absolute error (degrees)')
    plt.title(f'Angle differences over training ({tag})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'angle_differences_{tag}.png'))
    plt.close()

def plot_phi_angle_differences(pred_euler_over_epochs, gt_euler, save_dir, tag=''):
    """
    Plots average phi (in-membrane) angle differences over epochs given predicted and ground truth Euler angles.
    
    Parameters:
    -----------
    pred_euler_over_epochs : list of numpy arrays
        List of predicted Euler angles (ZYZ convention, degrees) for each epoch, shape (N, 3) each.
    gt_euler : numpy array
        Ground truth Euler angles (ZYZ convention, degrees), shape (N, 3).
    save_dir : str
        Directory to save the plot.
    tag : str
        Additional tag for the filename.
    """
    epochs = range(len(pred_euler_over_epochs))
    avg_phi_errors = []
    
    # Ensure gt_euler is in degrees
    gt_euler_deg = gt_euler # Assuming input is already degrees as per convention
    
    for epoch_euler in pred_euler_over_epochs:
        # Assume input epoch_euler is also in degrees
        pred_euler_deg = epoch_euler 
        
        # Calculate phi differences and account for periodicity
        # Phi is the third angle (index 2) in ZYZ
        phi_diff = np.abs(pred_euler_deg[:, 2] - gt_euler_deg[:, 2])
        phi_diff = np.minimum(phi_diff, 360 - phi_diff)
        
        avg_phi_errors.append(np.mean(phi_diff))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, avg_phi_errors, marker='o', label='in-membrane absolute error')
    plt.xlabel('Epoch')
    plt.ylabel('Absolute error (degrees)')
    plt.title(f'φ angle differences over training ({tag})')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'phi_angle_differences_{tag}.png'))
    plt.close()

def plot_error_histograms(pred_euler, gt_euler, epoch, save_dir, tag=''):
    """
    Plots histograms of the angle errors computed directly from Euler angles.
    
    Parameters:
    -----------
    pred_euler : numpy array
        Predicted Euler angles (ZYZ convention, degrees), shape (N, 3).
    gt_euler : numpy array
        Ground truth Euler angles (ZYZ convention, degrees), shape (N, 3).
    epoch : int
        Current epoch/iteration number.
    save_dir : str
        Directory where the plot image should be saved.
    tag : str
        Additional tag for the output filename.
    """
    # Assume input angles are already in degrees and follow ZYZ convention
    pred_deg = pred_euler
    gt_deg = gt_euler
    
    # Calculate angle differences in degrees
    # Indices: 0=Psi (in-plane), 1=Theta (tilt), 2=Phi (in-membrane)
    psi_deg_diff = pred_deg[:, 0] - gt_deg[:, 0]
    theta_deg_diff = pred_deg[:, 1] - gt_deg[:, 1]
    phi_deg_diff = pred_deg[:, 2] - gt_deg[:, 2]
    
    # Adjust for periodic boundaries
    # Psi and Phi: [-180, 180]
    psi_deg_diff = ((psi_deg_diff + 180) % 360) - 180
    phi_deg_diff = ((phi_deg_diff + 180) % 360) - 180
    # Theta: [0, 180] -> difference range is effectively [-180, 180], but Relion tilt error is often shown in [-90, 90]
    # Let's wrap theta difference to [-90, 90] for consistency with potential conventions
    theta_deg_diff = ((theta_deg_diff + 90) % 180) - 90 
    
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    
    # Relion convention: Rot (Phi), Tilt (Theta), Psi (Psi)
    # Plot Phi (in-membrane)
    axs[0].hist(phi_deg_diff, bins=30, color='blue', alpha=0.7)
    axs[0].set_title(f'Epoch {epoch} In-Membrane Error')
    axs[0].set_xlabel('Error (deg)')
    axs[0].set_ylabel('Frequency')
    axs[0].set_xticks(np.arange(-180, 181, 30))

    # Plot Theta (tilt)
    axs[1].hist(theta_deg_diff, bins=30, color='green', alpha=0.7)
    axs[1].set_title(f'Epoch {epoch} Tilt Error')
    axs[1].set_xlabel('Error (deg)')
    axs[1].set_ylabel('Frequency')
    axs[1].set_xticks(np.arange(-90, 91, 30))

    # Plot Psi (in-plane)
    axs[2].hist(psi_deg_diff, bins=30, color='red', alpha=0.7)
    axs[2].set_title(f'Epoch {epoch} In-Plane Error')
    axs[2].set_xlabel('Error (deg)')
    axs[2].set_ylabel('Frequency')   
    axs[2].set_xticks(np.arange(-180, 181, 30)) 
    
    fig.suptitle(f'epoch_{epoch}_angle_errors', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout but leave room for suptitle
    
    if not os.path.exists(os.path.join(save_dir, 'angle_errors')):
        os.makedirs(os.path.join(save_dir, 'angle_errors'))
    plt.savefig(os.path.join(save_dir, 'angle_errors', f'epoch_{epoch}_angle_errors_{tag}.png'))
    plt.close()

def plot_pred_vs_gt_scatter(pred_euler, gt_euler, iteration, save_dir, tag=''):
    """
    Plots scatter plots comparing predicted and ground-truth Euler angles
    for each of the three angles: in-plane, tilt, and in-membrane.
    Uses a 2D histogram with a color gradient to show point density.
    
    Parameters
    ----------
    pred_euler : numpy.ndarray of shape (N, 3)
        Predicted Euler angles (ZYZ convention, degrees).
    gt_euler : numpy.ndarray of shape (N, 3)
        Ground truth Euler angles (ZYZ convention, degrees).
    iteration : int
        Current epoch/iteration number.
    save_dir : str
        Directory where the plot image should be saved.
    tag : str
        Additional tag for the output filename.
    """
    # Assume input angles are already in degrees and follow ZYZ convention
    pred_deg = pred_euler
    gt_deg = gt_euler


    angle_names = ["in-plane (psi)", "tilt (theta)", "in-membrane (phi)"] # Updated names for clarity

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))
    for i, ax in enumerate(axs):
        # Create 2D histogram with more bins for finer resolution
        if angle_names[i] == "tilt (theta)":
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
        
        # Set proper tick intervals
        if angle_names[i] == "tilt (theta)":
            ax.set_xticks(np.arange(0, 181, 30))
            ax.set_yticks(np.arange(0, 181, 30))
        else:
            ax.set_xticks(np.arange(-180, 181, 30))
            ax.set_yticks(np.arange(-180, 181, 30))
        
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
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout but leave room for suptitle
    
    out_dir = os.path.join(save_dir, 'pred_vs_gt')
    os.makedirs(out_dir, exist_ok=True)
    
    # Fix filename construction to handle empty tags
    if tag:
        outfile = os.path.join(out_dir, f'pred_vs_gt_scatter_ep_{iteration}_{tag}.png')
    else:
        outfile = os.path.join(out_dir, f'pred_vs_gt_scatter_ep_{iteration}.png')
        
    plt.savefig(outfile, dpi=300)  # Increased DPI for better resolution
    plt.close()

def plot_head_win_frequency(head_win_counts, epoch, save_dir):
    """
    Plots a bar chart showing the number of times (frequency) each head was the winning head (i.e. had the minimum per-head loss) over an epoch.

    Parameters:
    -----------
    head_win_counts : numpy array of shape (M,)
        Frequency count for each of the M heads
    epoch : int
        Current epoch number
    save_dir : str
        Directory in which to save the plot
    """
    M = head_win_counts.shape[0]
    heads = list(range(M))
    plt.figure(figsize=(10, 6))
    plt.bar(heads, head_win_counts, color='skyblue')
    plt.xlabel('Head Index')
    plt.ylabel('Winning Frequency')
    plt.title(f'Per-Head Winning Frequency at Epoch {epoch}')
    plt.xticks(heads)
    plt.tight_layout()
    out_file = os.path.join(save_dir, f'head_win_frequency_epoch_{epoch}.png')
    plt.savefig(out_file, dpi=300)
    plt.close()

def export_euler_angles_csv(euler_angles, aligned_euler_angles, output_csv):
    """Export Euler angles (in-plane, tilt, in-membrane) derived from rotation matrices to a CSV file.

    The rotation matrices are converted to Euler angles using the ZYZ convention, then converted to degrees.
    The first and third angles are remapped to [-180, 180] and the second (tilt) is clamped to [0, 180].
    
    Assumes input Euler angles are already in degrees and follow the ZYZ convention.

    Parameters:
    -------------    
    euler_angles : numpy.ndarray
        Array of shape (N, 3) representing the unaligned Euler angles (psi, theta, phi) in degrees.
    aligned_euler_angles : numpy.ndarray or None
        Array of shape (N, 3) representing the globally aligned Euler angles (psi, theta, phi) in degrees, or None.
    output_csv : str
        Output file path for the CSV file.
    """
    # Assume input Euler angles are already in degrees and ZYZ convention.
    # The remapping/clipping is assumed to be done by the calling function or conversion process.
    angles_deg = euler_angles
    
    # Save to CSV using numpy.savetxt with a header for the three columns
    # Use the provided output path directly.
    np.savetxt(output_csv, angles_deg, delimiter=",", header="inplane,tilt,inmembrane", comments="")

    # Save aligned angles if provided, overwriting the previous file is the intended behavior based on original code.
    if aligned_euler_angles is not None:
        aligned_angles_deg = aligned_euler_angles
        np.savetxt(output_csv, aligned_angles_deg, delimiter=",", header="inplane,tilt,inmembrane", comments="")

