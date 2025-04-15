import torch
import numpy as np
from torch.utils.data import Dataset
import os
import mrcfile
from utils.ctf import primal_to_fourier_2D
from kornia.geometry.transform import translate
import lie_tools
import torchvision.transforms as transforms  # Import torchvision for resizing images
import pandas as pd
from torch.fft import fft2, ifft2, fftshift, ifftshift
import math
import starfile
import pytorch3d.transforms
import scipy.ndimage

def euler_angles2matrix(alpha, beta, gamma):
    """
    Converts euler angles in RELION convention to rotation matrix.

    Parameters
    ----------
    alpha: float / np.array
    beta: float / np.array
    gamma: float / np.array

    Returns
    -------
    A: np.array (3, 3)
    """
    # For RELION Euler angle convention
    ca = np.cos(alpha)
    cb = np.cos(beta)
    cg = np.cos(gamma)
    sa = np.sin(alpha)
    sb = np.sin(beta)
    sg = np.sin(gamma)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa

    A = np.zeros((3, 3))
    A[0, 0] = cg * cc - sg * sa
    A[0, 1] = -cg * cs - sg * ca
    A[0, 2] = cg * sb
    A[1, 0] = sg * cc + cg * sa
    A[1, 1] = -sg * cs + cg * ca
    A[1, 2] = sg * sb
    A[2, 0] = -sc
    A[2, 1] = ss
    A[2, 2] = cb
    return A

def fourier_crop(img_tensor):
        """
        Crops an image in the Fourier domain to 128x128.

        Parameters:
        -----------
        img_tensor: torch.Tensor
            The input image in spatial domain (1, H, W).
        
        Returns:
        --------
        cropped_img: torch.Tensor
            The cropped image in the spatial domain (1, 128, 128).
        """
        # Step 1: Perform 2D Fourier transform
        fft_img = fftshift(fft2(img_tensor))
        
        # Step 2: Crop the Fourier domain
        _, h, w = fft_img.shape
        crop_h, crop_w = 128, 128
        start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
        cropped_fft = fft_img[:, start_h:start_h + crop_h, start_w:start_w + crop_w]

        # Step 3: Inverse FFT to bring back to spatial domain
        cropped_img = ifft2(ifftshift(cropped_fft)).real
        
        return cropped_img
    
def create_radial_hann(size):
    """
    Creates a radial Hann window.
    
    Parameters:
    -----------
    size: int
        Size of the window (assumes square image)
        
    Returns:
    --------
    window: torch.Tensor
        2D radial Hann window
    """
    center = size // 2
    Y, X = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    R = torch.sqrt((X - center) ** 2 + (Y - center) ** 2)
    R = R / R.max()  # Normalize distances
    window = 0.5 * (1 + torch.cos(math.pi * R))
    return window

def center_crop(img_tensor):
    """
    Center crops an image to 128x128.
    """
    h, w = img_tensor.shape
    start_h, start_w = (h - 128) // 2, (w - 128) // 2
    return img_tensor[start_h:start_h + 128, start_w:start_w + 128]


class StarfileDataLoader(Dataset):
    def __init__(self, side_len, path_to_starfile='/fs01/datasets/empiar/vatpase_synth_noisy',
                 input_starfile='data.star', invert_hand=True, max_n_projs=None):
        """
        Initialization of a dataloader from starfile format.

        Parameters
        ----------
        config: namespace
        """
        super(StarfileDataLoader, self).__init__()


        self.path_to_starfile = path_to_starfile
        self.starfile = input_starfile
        self.df = starfile.open(os.path.join(self.path_to_starfile, self.starfile))
        self.correct_df = starfile.open(os.path.join(self.path_to_starfile, "all_correct.star"))
        self.sidelen_input = side_len
        self.vol_sidelen = side_len

        self.invert_hand = invert_hand
        self.invert_data = True

        idx_max = len(self.df['particles']) - 1
        if max_n_projs is not None:
            self.num_projs = max_n_projs
        else:
            self.num_projs = idx_max + 1
        self.idx_min = 0

        # Dictionary to store updated init_rots
        self.updated_init_rots = {}

        self.ctf_params = {
            "ctf_size": self.vol_sidelen,
            "kV": self.df['optics']['rlnVoltage'][0],
            "spherical_abberation": self.df['optics']['rlnSphericalAberration'][0],
            "amplitude_contrast": self.df['optics']['rlnAmplitudeContrast'][0],
            "resolution": self.df['optics']['rlnImagePixelSize'][0] * self.sidelen_input / self.vol_sidelen,
            "n_particles": idx_max + 1
        }

    def update_init_rots(self, indices, new_rots):
        """
        Update initial rotation matrices for specified indices.
        
        Parameters:
        -----------
        indices: torch.Tensor or numpy.ndarray
            Indices of data points to update
        new_rots: torch.Tensor or numpy.ndarray
            New rotation matrices to use as init_rots (shape: [n_indices, 3, 3])
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
            
        if isinstance(new_rots, torch.Tensor):
            new_rots = new_rots.cpu().numpy()
            
        # Update the rotation matrices for the specified indices
        for i, idx in enumerate(indices):
            self.updated_init_rots[idx] = torch.tensor(new_rots[i], dtype=torch.float)
            
    def reset_init_rots(self):
        """Reset all init_rots to their original values from the starfile"""
        self.updated_init_rots = {}

    def __len__(self):
        return self.num_projs

    def __getitem__(self, idx):
        """
        Initialization of a dataloader from starfile format.

        Parameters
        ----------
        idx: int

        Returns
        -------
        in_dict: Dictionary
        """
        particle = self.df['particles'].iloc[idx + self.idx_min]
        gt_particle = self.correct_df['particles'].iloc[idx + self.idx_min]
        
        # Load particle image from mrcs file
        imgname_raw = particle['rlnImageName']
        imgnamedf = particle['rlnImageName'].split('@')
        mrc_path = os.path.join(self.path_to_starfile, imgnamedf[1])
        pidx = int(imgnamedf[0]) - 1
        with mrcfile.mmap(mrc_path, mode='r', permissive=True) as mrc:
            proj = torch.from_numpy(mrc.data[pidx].copy()).float()
        
        if proj.shape[-1] != 128:
            proj = fourier_crop(proj)

        # # Apply in-plane rotation, reshape=True, and then crop to 128x128
        # proj = scipy.ndimage.rotate(proj, -particle['rlnAnglePsi'], reshape=True)
        # proj = center_crop(proj)
        proj = proj[None, :, :]  # add a dummy channel (for consistency w/ img fmt)

        # gt_particle['rlnAnglePsi'] -= particle['rlnAnglePsi']
        # particle['rlnAnglePsi'] -= particle['rlnAnglePsi']

        # except Exception:
        #     print(f"WARNING: Particle image {particle['rlnImageName']} invalid!\nSetting to zeros.")
        #     proj = torch.zeros(self.vol_sidelen, self.vol_sidelen)
        #     proj = proj[None, :, :]


        # # Read "GT" orientations USE PYTORCH 3D
        rotmat = torch.from_numpy(
            euler_angles2matrix(
                np.radians(-particle['rlnAngleRot']),
                np.radians(particle['rlnAngleTilt'])*(-1 if self.invert_hand else 1),
                np.radians(-particle['rlnAnglePsi'])
            )
        ).float()

        gt_rotmat = torch.from_numpy(
            euler_angles2matrix(
                np.radians(-gt_particle['rlnAngleRot']),
                np.radians(gt_particle['rlnAngleTilt'])*(-1 if self.invert_hand else 1),
                np.radians(-gt_particle['rlnAnglePsi'])
            )
        ).float()

        # rotmat = pytorch3d.transforms.euler_angles_to_matrix(
        #     torch.tensor([
        #         np.radians(-particle['rlnAnglePsi']),
        #         np.radians(particle['rlnAngleTilt']),
        #         np.radians(-particle['rlnAnglePsi'])
        #     ]),
        #     "ZYZ"
        # )

        # gt_rotmat = pytorch3d.transforms.euler_angles_to_matrix(
        #     torch.tensor([
        #         np.radians(-gt_particle['rlnAnglePsi']),
        #         np.radians(gt_particle['rlnAngleTilt']),
        #         np.radians(-gt_particle['rlnAngleRot'])
        #     ]),
        #     "ZYZ"
        # )

        shiftX = torch.from_numpy(np.array(particle['rlnOriginXAngst']))
        shiftY = torch.from_numpy(np.array(particle['rlnOriginYAngst']))
        shifts = torch.stack([shiftX, shiftY], dim=-1)

        fproj = primal_to_fourier_2D(proj)
        
        # # Use updated init_rots if available for this index
        # init_rots = self.updated_init_rots.get(idx, rotmat)
        
        in_dict = {'proj_input': proj,
                   'fproj': fproj,
                   'rots': rotmat,
                   'init_rots': rotmat,
                   'gt_rots': gt_rotmat,
                   'shifts': shifts,
                   'idx': torch.tensor(idx, dtype=torch.long),
                   }

        if self.ctf_params is not None:
            in_dict['defocusU'] = torch.from_numpy(np.array(particle['rlnDefocusU'] / 1e4, ndmin=2)).float()
            in_dict['defocusV'] = torch.from_numpy(np.array(particle['rlnDefocusV'] / 1e4, ndmin=2)).float()
            in_dict['angleAstigmatism'] = torch.from_numpy(np.radians(np.array(particle['rlnDefocusAngle'], ndmin=2))).float()
        return in_dict
    

class RealDataset(Dataset):
    # def __init__(self, invert_data=False): switched it to true for our data
    def __init__(self, invert_data=True):
        super(RealDataset, self).__init__()
        self.base_path = '/h/bizigerd/rotation_debugging/downsampled'  # Adjust this path to your dataset

        # Switch data dir to J3365 extract data, and re-fourier crop

        self.extract_path = '/h/bizigerd/rotation_debugging/angle_estimated_equatorial_particles_rotation_matrix.cs'
        self.passthrough_path = '/h/bizigerd/rotation_debugging/angle_estimated_equatorial_particles_passthrough_rotation_matrix.cs'
        
        # Load extracted particles (alignment data)
        extract = np.load(self.extract_path, allow_pickle=True)
        self.ts = extract['alignments3D/shift'].copy()
        self.pose = extract['alignments3D/pose'].copy()
        self.paths = extract['blob/path'].copy()
        self.ids = extract['blob/idx'].copy()

        # Decode paths
        paths_decoded = np.array([p.decode('utf-8') if isinstance(p, bytes) else p for p in self.paths])

        # Remove entries with unwanted substring in paths
        unwanted_substring = 'FoilHole_11266022_Data_10173115_10173117_20231012_030400_EER_patch_aligned_doseweighted_particles.mrc'
        indices_to_remove = np.where([unwanted_substring in path for path in paths_decoded])[0]
        mask = np.ones(len(self.paths), dtype=bool)
        mask[indices_to_remove] = False

        # Apply mask to all arrays
        self.paths = self.paths[mask]
        self.ts = self.ts[mask]
        self.pose = self.pose[mask]
        self.ids = self.ids[mask]

        # # Load passthrough particles (blob paths, indices, and CTF parameters)
        # passthrough = np.load(self.passthrough_path, allow_pickle=True)
        # self.ctf_df1 = passthrough['ctf/df1_A']
        # self.ctf_df2 = passthrough['ctf/df2_A']
        # self.ctf_angle = passthrough['ctf/df_angle_rad']
        # self.ctf_amp_contrast = passthrough['ctf/amp_contrast']
        # self.ctf_cs = passthrough['ctf/cs_mm']
        # self.ctf_accel_kv = passthrough['ctf/accel_kv']
        # self.ctf_phase_shift = passthrough['ctf/phase_shift_rad']
        # self.ctf_angle_astigmatism = passthrough['ctf/df_angle_rad']

        # self.ctf_df1 = self.ctf_df1[mask]
        # self.ctf_df2 = self.ctf_df2[mask]
        # self.ctf_angle = self.ctf_angle[mask]
        # self.ctf_amp_contrast = self.ctf_amp_contrast[mask]
        # self.ctf_cs = self.ctf_cs[mask]
        # self.ctf_accel_kv = self.ctf_accel_kv[mask]
        # self.ctf_phase_shift = self.ctf_phase_shift[mask]
        # self.ctf_angle_astigmatism = self.ctf_angle_astigmatism[mask]

        # # Set volume side length (vol_sidelen) and side_len input
        # self.vol_sidelen = 128
        # self.sidelen_input = 128

        # self.ctf_params = {
        #     "ctf_size": self.vol_sidelen,
        #     "kV": float(self.ctf_accel_kv[0]),
        #     "spherical_abberation": float(self.ctf_cs[0]),
        #     "amplitude_contrast": float(self.ctf_amp_contrast[0]),
        #     "resolution": float(self.ctf_df1[0] * self.sidelen_input / self.vol_sidelen),
        #     "n_particles": len(self.paths)
        # }

        self.n_data = len(self.paths)
        self.invert_data = True
        
        # Dictionary to store updated init_rots
        self.updated_init_rots = {}

    def update_init_rots(self, indices, new_rots):
        """
        Update initial rotation matrices for specified indices.
        
        Parameters:
        -----------
        indices: torch.Tensor or numpy.ndarray
            Indices of data points to update
        new_rots: torch.Tensor or numpy.ndarray
            New rotation matrices to use as init_rots (shape: [n_indices, 3, 3])
        """
        if isinstance(indices, torch.Tensor):
            indices = indices.cpu().numpy()
            
        if isinstance(new_rots, torch.Tensor):
            new_rots = new_rots.cpu().numpy()
            
        # Update the rotation matrices for the specified indices
        for i, idx in enumerate(indices):
            self.updated_init_rots[idx] = torch.tensor(new_rots[i], dtype=torch.float)
            
    def reset_init_rots(self):
        """Reset all init_rots to their original values"""
        self.updated_init_rots = {}

    def __len__(self):
        return self.n_data
    

    def __getitem__(self, i):
        idx = self.ids[i]
        address = self.paths[i].decode("utf-8")  # Use [i] instead of [idx]
        mrcfile_path = os.path.join('//datasets/empiar/', address)

        # print(f"__getitem__ called with index: {i}", flush=True)
        # print(f"Loading mrcfile_path: {mrcfile_path}, idx: {idx}", flush=True)
        
        with mrcfile.mmap(mrcfile_path, mode='r', permissive=True) as mrc:
            # print(f"mrc.data.shape: {mrc.data.shape}", flush=True)
            # Check if mrc is NoneType
            if mrc.data is None:
                print(f"mrc.data is None for {mrcfile_path}, idx: {idx}", flush=True)
                return None
            if mrc.data.ndim == 2:
                assert idx == 0
                proj = np.array(mrc.data.copy())
            else:
                proj = np.array(mrc.data[idx].copy())
            proj = proj[None, :, :]  # Shape becomes (1, H, W)
            old_D = proj.shape[-1]
            new_D = 128
            if old_D > new_D:
                proj = fourier_crop(torch.from_numpy(proj).float())
            else:
                proj = torch.from_numpy(proj).float()
            
            # # Apply radial Hann window
            # hann_window = self.create_radial_hann(128).to(proj.device)
            # proj = proj * hann_window[None, :, :]

        if self.invert_data:
            proj *= -1
        
        # Rotations
        aa = torch.from_numpy(self.pose[i])
        rot = lie_tools.expmap(aa[None])[0].T
        
        # Use updated init_rots if available for this index
        init_rot = self.updated_init_rots.get(i, rot)

        # Translation
        t = torch.from_numpy(self.ts[i])
        translated_proj = translate(proj[None], t[None]).squeeze(0)
        
        # Fourier transform
        fproj = primal_to_fourier_2D(translated_proj)

        # Create input dictionary with CTF parameters
        in_dict = {
            'proj_input': translated_proj,
            'fproj': fproj,
            'rots': rot,
            'init_rots': init_rot,
            'gt_rots': rot,
            'shifts': t,
            'idx': torch.tensor(i, dtype=torch.long)
        }
        
        return in_dict