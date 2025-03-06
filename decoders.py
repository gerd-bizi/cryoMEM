import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from utils.ctf import fourier_to_primal_3D
from utils.nets import SIREN, FCBlock, FourierNet
from utils.nets import PositionalEncoding
from utils.frequency_marcher import FrequencyMarcher


class Explicit3D(nn.Module):
    def __init__(self, downsampled_sz, img_sz, hartley):
        super(Explicit3D, self).__init__()
        self.hartley = hartley
        self.D = downsampled_sz + 1
        self.img_sz = img_sz
        self.D2 = img_sz + 1
        
        
        # Original random initialization
        if hartley:
            self.fvol = torch.nn.Parameter(torch.randn(1, 2, self.D2, self.D2, self.D2))
        else:
            self.fvol = torch.nn.Parameter(torch.randn(1, 4, self.D2, self.D2, self.D2))
        lincoords = np.linspace(-1., 1., self.D2, endpoint=True)
        Z, Y, X = np.meshgrid(lincoords, lincoords, lincoords, indexing='ij')
        # X, Y, Z = np.meshgrid(lincoords, lincoords, lincoords, indexing='ij')
        # coords = np.stack([Y, X, Z], axis=-1)
        coords = np.stack([X, Y, Z], axis=-1)
        coords_3d = torch.tensor(coords).float()
        self.register_buffer('coords_3d', coords_3d)

        X, Y = np.meshgrid(lincoords, lincoords, indexing='ij')
        coords = np.stack([Y, X, np.zeros_like(X)], axis=-1)
        coords_2d = torch.tensor(coords).float()
        self.register_buffer('coords_2d', coords_2d)
    
    def get_mask(self, coords, radius):
        return torch.norm(coords, dim=-1) < radius
        
    def forward(self, rotmat, r=None):
        batch_sz = rotmat.shape[0]
        plane_coords = self.coords_2d.view(-1, 3)
        expanded_coords = plane_coords.repeat(batch_sz, 1, 1)
        rot_plane_coords = torch.bmm(expanded_coords, rotmat)

        fslice = F.grid_sample(self.fvol, rot_plane_coords[None, None], align_corners=True, mode='bilinear')
        fslice = fslice.reshape(-1, batch_sz, self.D2, self.D2)
        if self.hartley:
            hslice_exp, hslice_mantissa = fslice.unbind(0)
            hslice = hslice_mantissa * torch.exp(hslice_exp)
            flipped_hslice = torch.flip(hslice, dims=[-1, -2])
            fslice_real = (flipped_hslice + hslice) / 2
            fslice_imag = (flipped_hslice - hslice) / 2
        else:
            fslice_real_exp, fslice_real_mantissa, fslice_imag_exp, fslice_imag_mantissa = fslice.unbind(0)
            fslice_real = fslice_real_mantissa * torch.exp(fslice_real_exp)
            fslice_imag = fslice_imag_mantissa * torch.exp(fslice_imag_exp)
        pred_fproj = fslice_real + 1j * fslice_imag
        pred_fproj = pred_fproj[:, None, :self.img_sz, :self.img_sz]
        mask = torch.ones((self.img_sz, self.img_sz), device=pred_fproj.device)
        if r is not None:
            mask = self.get_mask(plane_coords, r).view(self.D2, self.D2)
            mask = mask[:self.img_sz, :self.img_sz]
            # print(mask.shape)

        output_dict = {'pred_fproj_prectf': pred_fproj,
                       'mask': mask}
        return output_dict

    def make_volume(self, r=None, real_radius=None):
        coords_3d = self.coords_3d.view(-1, 3)

        fvol = F.grid_sample(self.fvol, 
                             coords_3d[None, None, None], 
                             align_corners=True, mode='bilinear')
        fvol = fvol.reshape(-1, self.D2, self.D2, self.D2)
        if self.hartley:
            fvol_exp, fvol_mantissa = fvol.unbind(0)
            fvol = fvol_mantissa * torch.exp(fvol_exp)
            flipped_fvol = torch.flip(fvol, dims=[0, 1, 2])
            fvol_real = (flipped_fvol + fvol) / 2
            fvol_imag = (flipped_fvol - fvol) / 2
        else:
            fvol_real_exp, fvol_real_mantissa, fvol_imag_exp, fvol_imag_mantissa = fvol.unbind(0)
            fvol_real = fvol_real_mantissa * torch.exp(fvol_real_exp)
            fvol_imag = fvol_imag_mantissa * torch.exp(fvol_imag_exp)
        fvol = fvol_real + 1j * fvol_imag
        fvol = fvol[:self.img_sz, :self.img_sz, :self.img_sz]
            
        if r is not None:
            mask = self.get_mask(coords_3d, r)
            mask = mask.view(self.D2, self.D2, self.D2)[:self.img_sz, :self.img_sz, :self.img_sz]
            fvol = fvol * mask

        vol = fourier_to_primal_3D(fvol).real
        return vol
    

class VolumeBase(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def make_volume(self):
        ...


def shift_coords(coords, x_range, y_range, z_range, Nx, Ny, Nz, flip=False):
    """
    Shifts the coordinates and puts the DC component at (0, 0, 0).

    Parameters
    ----------
    coords: torch.tensor (..., 3)
    x_range: float
        (max_x - min_x) / 2
    y_range: float
        (max_y - min_y) / 2
    z_range: float
        (max_z - min_z) / 2
    Nx: int
    Ny: int
    Nz: int
    flip: bool

    Returns
    -------
    coords: torch.tensor (..., 3)
    """
    alpha = -1.
    if flip:  # "unshift" the coordinates.
        alpha = 1.

    if Nx % 2 == 0:
        x_shift = coords[..., 0] + alpha * x_range / (Nx - 1)
    else:
        x_shift = coords[..., 0]
    if Ny % 2 == 0:
        y_shift = coords[..., 1] + alpha * y_range / (Ny - 1)
    else:
        y_shift = coords[..., 1]
    if Nz % 2 == 0:
        z_shift = coords[..., 2] + alpha * z_range / (Nz - 1)
    else:
        z_shift = coords[..., 2]
    coords = torch.cat((x_shift.unsqueeze(-1),
                        y_shift.unsqueeze(-1),
                        z_shift.unsqueeze(-1)), dim=-1)
    return coords

class ImplicitFourierVolume(VolumeBase):
    def __init__(self, img_sz, params_implicit):
        """
        Implicit representation of the Fourier volume with a coordinate system
        matching the explicit model.
        """
        super(ImplicitFourierVolume, self).__init__()
        self.img_sz = img_sz
        self.is_chiral = False
        # Use the same number of grid points as the explicit model.
        lincoords = torch.linspace(-1., 1., self.img_sz)
        
        # Create the 2D plane coordinates.
        # Explicit model does:
        #   X, Y = np.meshgrid(lincoords, lincoords, indexing='ij')
        #   coords = np.stack([Y, X, np.zeros_like(X)], axis=-1)
        # We do the same:
        X, Y = torch.meshgrid(lincoords, lincoords, indexing='ij')
        coords_2d = torch.stack([Y, X, torch.zeros_like(X)], dim=-1)  # shape: (D2, D2, 3)
        self.register_buffer('plane_coords', coords_2d)
        
        # Create the 3D volume coordinates.
        # Explicit model does:
        #   Z, Y, X = np.meshgrid(lincoords, lincoords, lincoords, indexing='ij')
        #   coords = np.stack([X, Y, Z], axis=-1)
        # We replicate that:
        Z, Y, X = torch.meshgrid(lincoords, lincoords, lincoords, indexing='ij')
        coords_3d = torch.stack([X, Y, Z], dim=-1)  # shape: (D2, D2, D2, 3)
        self.register_buffer('coords_3d', coords_3d.reshape(-1, 3))
        
        # Create the Fourier network (or alternative) according to params_implicit.
        if params_implicit["type"] == 'siren':
            self.fvol = SIREN(in_features=3, out_features=2,
                              num_hidden_layers=4, hidden_features=256,
                              outermost_linear=True, w0=30)
            self.pe = None
        elif params_implicit["type"] == 'fouriernet':
            self.fvol = FourierNet(force_symmetry=params_implicit['force_symmetry'])
            self.pe = None
        elif params_implicit["type"] == 'relu_pe':
            num_encoding_fns = 6
            self.pe = PositionalEncoding(num_encoding_fns)
            self.fvol = FCBlock(in_features=3 * (1 + 2 * num_encoding_fns), 
                                features=[256, 256, 256, 256],
                                out_features=2, nonlinearity='relu', 
                                last_nonlinearity=None)
        else:
            raise NotImplementedError

    def forward(self, rotmat, r=None):
        """
        Given a rotation matrix, rotates the 2D plane coordinates, queries the Fourier
        network, and returns a dictionary with the complex Fourier projection and a mask.
        """
        batch_sz = rotmat.shape[0]
        # Flatten the plane coordinates from (D2, D2, 3) to (D2*D2, 3)
        plane_coords = self.plane_coords.view(-1, 3)
        
        # Repeat for each batch element and apply the rotation matrix.
        # rotmat has shape (batch_sz, 3, 3)
        rot_plane_coords = torch.bmm(plane_coords.unsqueeze(0).repeat(batch_sz, 1, 1), rotmat)
        
        if self.pe is not None:
            rot_plane_coords = self.pe(rot_plane_coords)
        
        # Outputs 2 channels (interpreted as real and imaginary exponents).
        fplane = self.fvol(rot_plane_coords)  # shape: (batch_sz, D2*D2, 2)
        fplane = fplane.reshape(batch_sz, self.img_sz, self.img_sz, 2)
        fplane = torch.view_as_complex(fplane)
        # Mimic explicit by selecting the first img_sz indices.
        fplane = fplane[:, None, :self.img_sz, :self.img_sz]
        
        # Compute the mask based on the original 2D plane coordinates.
        mask = torch.ones((self.img_sz, self.img_sz), device=fplane.device)
        if r is not None:
            mask = (torch.norm(plane_coords, dim=-1) < r).view(self.img_sz, self.img_sz)[:self.img_sz, :self.img_sz]
        
        return {'pred_fproj_prectf': fplane, 'mask': mask}

    def make_volume(self, resolution='full', r=None):
        """
        Generates a voxel-grid volume from the Fourier representation.
        """
        with torch.no_grad():
            if resolution == 'full':
                coords = self.coords_3d
                vol_sz = self.img_sz
            else:
                left = max(0, self.img_sz // 2 - resolution)
                right = min(self.img_sz, self.img_sz // 2 + resolution + 1)
                coords = self.coords_3d.reshape(self.img_sz, self.img_sz, self.img_sz, 3)[left:right, left:right, left:right, :].reshape(-1, 3)
                vol_sz = right - left

            if self.pe is not None:
                coords = self.pe(coords)

            exp_fvol = self.fvol(coords).reshape(vol_sz, vol_sz, vol_sz, 2)
            exp_fvol = torch.view_as_complex(exp_fvol)
            exp_fvol = torch.fft.ifftshift(exp_fvol, dim=(-3, -2, -1))
            exp_vol = torch.fft.fftshift(torch.fft.ifftn(exp_fvol, s=(vol_sz, vol_sz, vol_sz), dim=(-3, -2, -1)),
                                          dim=(-3, -2, -1))
            return exp_vol.real