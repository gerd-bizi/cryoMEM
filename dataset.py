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
from skimage.draw import polygon # Import polygon
from torchvision.transforms.functional import affine  # Add this import

def aa2quat(ax, theta=None):
    if theta is None:
        theta = np.linalg.norm(ax)
        if theta != 0:
            ax = ax / theta
    q = np.zeros(4, dtype=ax.dtype)
    q[0] = np.cos(theta / 2)
    q[1:] = ax * np.sin(theta / 2)
    return q

def quat2rot(q):
    n = np.sum(q**2)
    s = 0 if n == 0 else 2 / n
    wx = s * q[0] * q[1]
    wy = s * q[0] * q[2]
    wz = s * q[0] * q[3]
    xx = s * q[1] * q[1]
    xy = s * q[1] * q[2]
    xz = s * q[1] * q[3]
    yy = s * q[2] * q[2]
    yz = s * q[2] * q[3]
    zz = s * q[3] * q[3]
    r = np.array([[1 - (yy + zz), xy + wz,       xz - wy],
                  [xy - wz,       1 - (xx + zz), yz + wx],
                  [xz + wy,       yz - wx,       1 - (xx + yy)]], dtype=q.dtype)
    return r

def rot2euler(r):
    """Decompose rotation matrix into Euler angles"""
    # assert(isrotation(r))
    # Shoemake rotation matrix decomposition algorithm with same conventions as Relion.
    epsilon = np.finfo(np.double).eps
    abs_sb = np.sqrt(r[0, 2] ** 2 + r[1, 2] ** 2)
    if abs_sb > 16 * epsilon:
        gamma = np.arctan2(r[1, 2], -r[0, 2])
        alpha = np.arctan2(r[2, 1], r[2, 0])
        if np.abs(np.sin(gamma)) < epsilon:
            sign_sb = np.sign(-r[0, 2]) / np.cos(gamma)
        else:
            sign_sb = np.sign(r[1, 2]) if np.sin(gamma) > 0 else -np.sign(r[1, 2])
        beta = np.arctan2(sign_sb * abs_sb, r[2, 2])
    else:
        if np.sign(r[2, 2]) > 0:
            alpha = 0
            beta = 0
            gamma = np.arctan2(-r[1, 0], r[0, 0])
        else:
            alpha = 0
            beta = np.pi
            gamma = np.arctan2(r[1, 0], -r[0, 0])
    return alpha, beta, gamma


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
    
def create_radial_hann(size=128, outer_radius=60):
    center = size // 2
    Y, X = torch.meshgrid(torch.arange(size), torch.arange(size), indexing='ij')
    R = torch.sqrt((X - center)**2 + (Y - center)**2)
    R = R / outer_radius  # normalize w.r.t. cutoff
    hann = 0.5 * (1 + torch.cos(np.pi * torch.clip(R, 0, 1)))
    hann[R > 1] = 0
    return hann

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
        self.df = starfile.read(os.path.join(self.path_to_starfile, self.starfile))
        self.correct_df = starfile.read(os.path.join(self.path_to_starfile, "all_correct.star"))
        self.sidelen_input = side_len
        self.vol_sidelen = side_len

        self.extract_path = None
        self.passthrough_path = None

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
                np.deg2rad(-particle['rlnAngleRot']),
                np.deg2rad(particle['rlnAngleTilt'])*(-1 if self.invert_hand else 1),
                np.deg2rad(-particle['rlnAnglePsi'])
            )
        ).float()

        gt_rotmat = torch.from_numpy(
            euler_angles2matrix(
                np.deg2rad(-gt_particle['rlnAngleRot']),
                np.deg2rad(gt_particle['rlnAngleTilt'])*(-1 if self.invert_hand else 1),
                np.deg2rad(-gt_particle['rlnAnglePsi'])
            )
        ).float()

        # rotmat = pytorch3d.transforms.euler_angles_to_matrix(
        #     torch.tensor([
        #         np.deg2rad(-particle['rlnAnglePsi']),
        #         np.deg2rad(particle['rlnAngleTilt']),
        #         np.deg2rad(-particle['rlnAnglePsi'])
        #     ]),
        #     "ZYZ"
        # )

        # gt_rotmat = pytorch3d.transforms.euler_angles_to_matrix(
        #     torch.tensor([
        #         np.deg2rad(-gt_particle['rlnAnglePsi']),
        #         np.deg2rad(gt_particle['rlnAngleTilt']),
        #         np.deg2rad(-gt_particle['rlnAngleRot'])
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
            in_dict['angleAstigmatism'] = torch.from_numpy(np.deg2rad(np.array(particle['rlnDefocusAngle'], ndmin=2))).float()
        return in_dict
    
def custom_quadrilateral_mask(shape: tuple[int, int],
                              hlen: float,
                              vlen: float,
                              center: tuple[float, float] = [64, 64]
                              ) -> np.ndarray:
    """
    Create a binary mask with a 4‑vertex polygon centered in the image.

    The polygon's vertices (in row, col order) are:
      (cy, cx + hlen)
      (cy + vlen, cx)
      (cy, cx - hlen)
      (cy - vlen, cx)

    Everything inside → 1, outside → 0.

    Parameters
    ----------
    shape : (H, W)
        Height and width of the mask.
    hlen : float
        Horizontal half‑span (in pixels) from center to polygon vertex.
    vlen : float
        Vertical half‑span (in pixels) from center to polygon vertex.
    center : (row, col), optional
        Polygon center. Defaults to image center: (H/2, W/2).

    Returns
    -------
    mask : ndarray of uint8, shape=(H, W)
        Binary mask (1 inside polygon, 0 outside).
    """
    H, W = shape
    if center is None:
        cy, cx = (H / 2.0, W / 2.0)
    else:
        cy, cx = center

    # Define the four vertices (in row, col)
    verts = np.array([
        ( cy,      cx + hlen ),  # right
        ( cy + vlen, cx      ),  # down
        ( cy,      cx - hlen ),  # left
        ( cy - vlen, cx      ),  # up
    ])

    # Split into row / col sequences and rasterize
    rr, cc = polygon(verts[:, 0], verts[:, 1], shape)

    mask = np.zeros(shape, dtype=np.uint8)
    # Clip in case vertices lie slightly outside
    valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    mask[rr[valid], cc[valid]] = 1

    return mask

class RealDataset(Dataset):
    # def __init__(self, invert_data=False): switched it to true for our data
    def __init__(self, invert_data=True, max_particles=None, big_dataset=False, partial_prior=False, set90dist=True):
        super(RealDataset, self).__init__()
        # self.base_path = '/h/bizigerd/rotation_debugging/downsampled'  # Adjust this path to your dataset

        # Switch data dir to J3365 extract data, and re-fourier crop

        if big_dataset:
            # Larger dataset
            self.extract_path = '/h/bizigerd/rotation_debugging/angle_estimated_equatorial_particles_rotation_matrix_big.cs'
            self.passthrough_path = '/h/bizigerd/rotation_debugging/angle_estimated_equatorial_particles_passthrough_rotation_matrix_big.cs'
        else:
            # Smaller dataset
            self.extract_path = '/h/bizigerd/rotation_debugging/angle_estimated_equatorial_particles_rotation_matrix.cs'
            self.passthrough_path = '/h/bizigerd/rotation_debugging/angle_estimated_equatorial_particles_passthrough_rotation_matrix.cs'

        self.radial_hann = create_radial_hann(size=128, outer_radius=60)

        
        # Assuming the GT poses are in the extracted_particles.cs file
        self.gt_extract_path = '/datasets/empiar/J3365/extracted_particles.cs' 
        # If GT poses are in the passthrough file, use this instead:
        # self.gt_extract_path = '/datasets/empiar/J3365/J3365_passthrough_particles.cs'
        
        # Load extracted particles (alignment data)
        extract = np.load(self.extract_path, allow_pickle=True)
        gt_extract = np.load(self.gt_extract_path, allow_pickle=True) # Corrected loading path
        self.ts = extract['alignments3D/shift'].copy()
        self.pose = extract['alignments3D/pose'].copy()
        self.paths = extract['blob/path'].copy()
        self.ids = extract['blob/idx'].copy()

        # Build map from GT blob/idx to GT pose
        gt_ids = gt_extract['blob/idx']
        # Make sure the key for GT pose is correct, assuming 'alignments3D/pose'
        gt_poses = gt_extract['alignments3D/pose'] 
        self.gt_pose_map = {gt_ids[idx]: gt_poses[idx] for idx in range(len(gt_ids))}
        
        # Report statistics on GT pose map
        matching_ids = sum(1 for idx in self.ids if idx in self.gt_pose_map)
        print(f"GT pose mapping statistics:")
        print(f"  - Total particles in dataset: {len(self.ids)}")
        print(f"  - Total particles with GT poses: {matching_ids}")
        print(f"  - Percentage with GT poses: {100.0 * matching_ids / len(self.ids):.2f}%")
        
        del gt_extract, gt_ids, gt_poses # Free up memory

        passthrough = np.load(self.passthrough_path, allow_pickle=True)

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

        # Apply mask to CTF arrays as well
        self.ctf_df1 = passthrough['ctf/df1_A'][mask]
        self.ctf_df2 = passthrough['ctf/df2_A'][mask]
        self.ctf_angle = passthrough['ctf/df_angle_rad'][mask]
        self.ctf_amp_contrast = passthrough['ctf/amp_contrast'][mask]
        self.ctf_cs = passthrough['ctf/cs_mm'][mask]
        self.ctf_accel_kv = passthrough['ctf/accel_kv'][mask]
        self.ctf_phase_shift = passthrough['ctf/phase_shift_rad'][mask]
        self.ctf_angle_astigmatism = passthrough['ctf/df_angle_rad'][mask] # Already assigned above, re-assigning for clarity

        # Limit the number of particles if max_particles is specified
        current_num_particles = len(self.paths)
        if max_particles is not None and current_num_particles > max_particles:
            print(f"Limiting dataset from {current_num_particles} to {max_particles} particles.")
            limit = int(max_particles) # Ensure it's an integer
            self.paths = self.paths[:limit]
            self.ts = self.ts[:limit]
            self.pose = self.pose[:limit]
            self.ids = self.ids[:limit]
            self.ctf_df1 = self.ctf_df1[:limit]
            self.ctf_df2 = self.ctf_df2[:limit]
            self.ctf_angle = self.ctf_angle[:limit]
            self.ctf_amp_contrast = self.ctf_amp_contrast[:limit]
            self.ctf_cs = self.ctf_cs[:limit]
            self.ctf_accel_kv = self.ctf_accel_kv[:limit]
            self.ctf_phase_shift = self.ctf_phase_shift[:limit]
            self.ctf_angle_astigmatism = self.ctf_angle_astigmatism[:limit]
        elif max_particles is not None:
             print(f"Dataset contains {current_num_particles} particles, which is less than or equal to the specified limit of {max_particles}. Using all {current_num_particles} particles.")
        else:
             print(f"No particle limit specified. Using all {current_num_particles} filtered particles.")


        # # Load passthrough particles (blob paths, indices, and CTF parameters)
        # passthrough = np.load(self.passthrough_path, allow_pickle=True)
        # self.ctf_df1 = passthrough['ctf/df1_A'] # These were moved up and masked
        # self.ctf_df2 = passthrough['ctf/df2_A']
        # self.ctf_angle = passthrough['ctf/df_angle_rad']
        # self.ctf_amp_contrast = passthrough['ctf/amp_contrast']
        # self.ctf_cs = passthrough['ctf/cs_mm']
        # self.ctf_accel_kv = passthrough['ctf/accel_kv']
        # self.ctf_phase_shift = passthrough['ctf/phase_shift_rad']
        # self.ctf_angle_astigmatism = passthrough['ctf/df_angle_rad']


        # Set volume side length (vol_sidelen) and side_len input
        self.vol_sidelen = 128
        self.sidelen_input = 128

        # Set n_data *after* potential truncation
        self.n_data = len(self.paths)

        # Update ctf_params with potentially truncated data and correct count
        # Ensure there's at least one particle left before accessing index 0
        if self.n_data > 0:
            self.ctf_params = {
                "ctf_size": self.vol_sidelen,
                "kV": float(self.ctf_accel_kv[0]),
                "spherical_abberation": float(self.ctf_cs[0]),
                "amplitude_contrast": float(self.ctf_amp_contrast[0]),
                "resolution": float(self.ctf_df1[0] * self.sidelen_input / self.vol_sidelen), # Assuming resolution calculation uses df1? Check this logic.
                "n_particles": self.n_data
            }
        else:
             # Handle case with no particles left after filtering/truncation
             self.ctf_params = {
                "ctf_size": self.vol_sidelen,
                "kV": 0, # Default values or raise error?
                "spherical_abberation": 0,
                "amplitude_contrast": 0,
                "resolution": 0,
                "n_particles": 0
             }
             print("Warning: No particles remaining after filtering and/or truncation.")


        # self.n_data = len(self.paths) # Moved up
        self.invert_data = True
        self.partial_prior = partial_prior
        self.set90dist = set90dist
            
    def reset_init_rots(self):
        """Reset all init_rots to their original values"""
        self.updated_init_rots = {}

    def __len__(self):
        return self.n_data
    

    def __getitem__(self, i):
        idx = self.ids[i]
        address = self.paths[i].decode("utf-8")
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
                proj_np = np.array(mrc.data.copy())
            else:
                proj_np = np.array(mrc.data[idx].copy())
            old_D = proj_np.shape[-1]
            new_D = 128
            if old_D > new_D:
                # fourier_crop expects (..., H, W), so add and remove batch dim
                proj = fourier_crop(torch.from_numpy(proj_np[None, :, :]).float())[0, :, :]
                # Calculate the scaling factor for shifts
                shift_scale = float(new_D) / old_D
            else:
                proj = torch.from_numpy(proj_np).float()
                # No resizing, so scale is 1
                shift_scale = 1.0
            # proj shape is now (H, W), e.g., (128, 128)

        if self.invert_data:
            proj *= -1

        # Get the blob/idx for the current particle in the base dataset
        current_blob_idx = self.ids[i]

        # Look up the corresponding GT pose using the blob/idx
        pose_gt_np = self.gt_pose_map.get(current_blob_idx)

        if pose_gt_np is not None:
            # Calculate GT rotation matrix if pose was found
            aa_gt = torch.from_numpy(pose_gt_np)
            q_gt = aa2quat(aa_gt.numpy())
            r_gt = quat2rot(q_gt)
            euler_angles_gt = np.array(rot2euler(r_gt)) # alpha, beta, gamma (in radians)
            # Create GT matrix using all three GT angles (consistent convention)
            rotation_matrix_gt = torch.from_numpy(
                euler_angles2matrix(
                    -euler_angles_gt[0],
                    -euler_angles_gt[1], # tilt
                    0
                )
            ).float()
        else:
            # Handle case where particle ID from base set is not in GT set
            print(f"Warning: GT pose not found for blob/idx: {current_blob_idx}. Using identity matrix.")
            rotation_matrix_gt = torch.eye(3, dtype=torch.float)
        
        # Calculate Rotations (Matrix and Euler Angles)
        aa = torch.from_numpy(self.pose[i])
        q = aa2quat(aa.numpy())
        r = quat2rot(q)
        # Initial euler angles [alpha, beta, gamma] in radians from initial pose estimate
        euler_angles = np.array(rot2euler(r))

        if self.partial_prior:
            euler_angles[2] = ((euler_angles_gt[2] + np.deg2rad(np.random.uniform(-10, 10))) + np.pi) % (2 * np.pi) - np.pi
            euler_angles[1] = (-(euler_angles_gt[1] + np.deg2rad(np.random.uniform(-10, 10)))).clip(0, np.pi) # pray for no gimbal lock lol
        elif self.set90dist:
            euler_angles[1] = np.deg2rad(np.random.normal(90, 3.75))  # 3.75 is std dev to get 95% CI of [82.5, 97.5]
        else:
            euler_angles[1] = np.deg2rad(90) # pray for no gimbal lock lol

        rotation_matrix = torch.from_numpy(
                euler_angles2matrix(
                    -euler_angles[0],
                    euler_angles[1],
                    0
                )
        ).float()

        # Apply in-plane rotation (psi / gamma) to the image tensor
        # scipy.ndimage.rotate expects (H, W) numpy array
        proj_rotated_np = scipy.ndimage.rotate(proj.numpy(), -np.rad2deg(euler_angles[2]), reshape=False)
        proj_rotated = torch.from_numpy(proj_rotated_np) # Back to tensor (H, W)

        # Load original translation and scale it according to image resizing
        t_original = torch.from_numpy(self.ts[i])
        t = t_original * shift_scale  # Scale the shift
        
        # Translation - apply the SCALED shift to the in-plane rotated image
        # affine expects (..., C, H, W), so add channel dim -> (1, H, W)
        # Use proj_rotated here
        translated_proj = affine(proj_rotated[None, :, :], angle=0, translate=(t[0].item(), t[1].item()), scale=1.0, shear=[0., 0.])
        # translated_proj shape is (1, H, W)

        # mask = custom_quadrilateral_mask(shape=(128, 128), hlen=55, vlen=55)
        # masked_proj = translated_proj[0] * mask  # (128, 128), remove channel
        # translated_proj = masked_proj[None, :, :]  # re-add channel dim

        # Apply gaussian window to the image
        translated_proj = translated_proj[0] * self.radial_hann
        translated_proj = translated_proj[None, :, :]


        # translated_proj = (translated_proj + 0.0098) / 0.8714

        # Fourier transform
        fproj = primal_to_fourier_2D(translated_proj)

        # Create input dictionary with CTF parameters
        in_dict = {
            'proj_input': translated_proj, # Shape (1, H, W)
            'fproj': fproj,
            'rots': rotation_matrix, # The 3D rotation matrix
            'init_rots': rotation_matrix, # Initial 3D rotation matrix
            'gt_rots': rotation_matrix_gt, # GT 3D rotation matrix
            'shifts': t,
            'idx': torch.tensor(i, dtype=torch.long)
        }
        
        return in_dict