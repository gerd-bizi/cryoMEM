import torch
import torch.nn as nn
import numpy as np
import pytorch3d.transforms
from encoders import CNNEncoderVGG16, GaussianPyramid
from utils.nets import FCBlock
from decoders import Explicit3D, ImplicitFourierVolume
from utils.ctf import CTFRelion
from utils.real_ctf import ExperimentalCTF


class CryoSAPIENCE(nn.Module):
    def __init__(self,  
                 num_rotations, 
                 ctf_params=None,
                 sidelen=128,
                 vol_rep='explicit',
                 num_octaves=4,
                 hartley=True,
                 experimental=False,
                 use_prior=False):
        super(CryoSAPIENCE, self).__init__()
        self.num_rotations = num_rotations
        self.sidelen = sidelen

        vol_rep = 'implicit'

        if vol_rep == 'explicit':
            self.pred_map = Explicit3D(downsampled_sz=sidelen, img_sz=sidelen, hartley=hartley)
        else:
            # Implicit Fourier Volume
            params_implicit = {"type": "fouriernet", "force_symmetry": False}
            self.pred_map = ImplicitFourierVolume(img_sz=sidelen, params_implicit=params_implicit)

        # Gaussian Pyramid
        self.gaussian_filters = GaussianPyramid(
                kernel_size=11,
                kernel_variance=0.01,
                num_octaves=num_octaves,
                octave_scaling=10
            )
        num_additional_channels = num_octaves

        # CNN encoder
        self.cnn_encoder = CNNEncoderVGG16(1 + num_additional_channels,
                                        batch_norm=True)
        cnn_encoder_out_shape = self.cnn_encoder.get_out_shape(sidelen, sidelen)
        latent_code_size = torch.prod(torch.tensor(cnn_encoder_out_shape))

        # Orientation regressor
        # self.latent_to_rot3d_fn = pytorch3d.transforms.rotation_6d_to_matrix
        # self.orientation_dims = 6
        # self.orientation_regressor = nn.ModuleList()
        # for _ in range(self.num_rotations):
        #     # We split the regressor in 2 to have access to the latent code
        #     self.orientation_regressor.append(FCBlock(
        #         in_features=latent_code_size,
        #         out_features=self.orientation_dims,
        #         features=[512, 256],
        #         nonlinearity='relu',
        #         last_nonlinearity=None,
        #         batch_norm=True,
        #         group_norm=0)
        #     )

        self.known_angle_regressor = FCBlock(
            in_features=latent_code_size,
            out_features=2,  # delta_theta, delta_psi
            features=[256, 128],
            nonlinearity='relu',
            last_nonlinearity='tanh',  # outputs in [-1, 1]
            batch_norm=True
        )
        self.angle_scale_factor = 1/18 * np.pi  # Store as a separate parameter

        self.phi_regressor = nn.ModuleList([
            FCBlock(
                in_features=latent_code_size,
                out_features=1,  # phi prediction
                features=[256, 128],
                nonlinearity='relu',
                last_nonlinearity=None,
                batch_norm=True
            ) for _ in range(self.num_rotations)
        ])


        if experimental:
            self.ctf = ExperimentalCTF()
        else:
            assert ctf_params is not None
            self.ctf = CTFRelion(size=ctf_params['ctf_size'], 
                                resolution=ctf_params['resolution'],
                                kV=ctf_params['kV'], 
                                valueNyquist=0.001, 
                                cs=ctf_params['spherical_abberation'],
                                amplitudeContrast=ctf_params['amplitude_contrast'], 
                                requires_grad=False,
                                num_particles=ctf_params['n_particles'], 
                                precompute=0,
                                flip_images=False)
        self.experimental = experimental
    
    def forward_amortized(self, in_dict, r):
        # # encoder
        # proj = in_dict['proj_input']
        # proj = self.gaussian_filters(proj)
        # latent_code = torch.flatten(self.cnn_encoder(proj), start_dim=1)
        # all_latent_code_prerot = []
        # for orientation_regressor in self.orientation_regressor:
        #     latent_code_prerot = orientation_regressor(latent_code)
        #     all_latent_code_prerot.append(latent_code_prerot)
        # all_latent_code_prerot = torch.stack(all_latent_code_prerot, dim=1)
        # pred_rotmat = self.latent_to_rot3d_fn(all_latent_code_prerot)
        # pred_rotmat = pred_rotmat.view(-1, 3, 3)

        # # decoder
        # out_dict = self.pred_map(pred_rotmat, r=r)
        # pred_fproj_prectf = out_dict['pred_fproj_prectf']
        # mask = out_dict['mask']
        # B, _, H, W = pred_fproj_prectf.shape
        # pred_fproj_prectf = pred_fproj_prectf.view(B // self.num_rotations, self.num_rotations, H, W)
        # pred_rotmat = pred_rotmat.view(B // self.num_rotations, self.num_rotations, 3, 3)

        # # gt_rotmat = in_dict['gt_rots']

        # # predicted_euler_angles = pytorch3d.transforms.matrix_to_euler_angles(pred_rotmat, 'ZYZ')

        # # gt_euler_angles = pytorch3d.transforms.matrix_to_euler_angles(gt_rotmat, 'ZYZ')

        # # predicted_euler_angles[0] = gt_euler_angles[0]
        # # predicted_euler_angles[1] = gt_euler_angles[1]

        # # pred_rotmat = pytorch3d.transforms.euler_angles_to_matrix(predicted_euler_angles, 'ZYZ')

        proj = in_dict['proj_input']
        proj = self.gaussian_filters(proj)
        latent_code = torch.flatten(self.cnn_encoder(proj), start_dim=1)
        
        # Get delta_theta and delta_psi predictions and scale them
        delta_angles = self.known_angle_regressor(latent_code) * self.angle_scale_factor
        
        # Get multiple phi predictions
        all_phi = []
        for phi_reg in self.phi_regressor:
            phi = phi_reg(latent_code)  # Shape: [B, 1]
            all_phi.append(phi)
        all_phi = torch.stack(all_phi, dim=1)  # Shape: [B, num_rotations, 1]
        
        # Combine with known angles from input
        gt_angles = pytorch3d.transforms.matrix_to_euler_angles(in_dict['gt_rots'], 'ZYZ')  # [B, 3]

        #Swap the order of the angles to be ZYZ
        # gt_angles = gt_angles[:, [2, 1, 0]]

        # Zero out deltas, and randomize phi, expect rotationally averaged

        # Construct final euler angles for each hypothesis
        B = latent_code.shape[0]
        # pred_angles = torch.zeros(B, self.num_rotations, 3, device=latent_code.device)
        # pred_angles[:, :, 0] = gt_angles[:, 0:1].expand(-1, self.num_rotations) + delta_angles[:, 0:1].expand(-1, self.num_rotations)  # psi (second Z rotation)
        # pred_angles[:, :, 1] = gt_angles[:, 1:2].expand(-1, self.num_rotations) + delta_angles[:, 1:2].expand(-1, self.num_rotations)  # theta (Y rotation)
        # pred_angles[:, :, 2] = all_phi.squeeze(-1)  # phi (first Z rotation)

        pred_angles = torch.zeros(B, self.num_rotations, 3, device=latent_code.device)
        pred_angles[:, :, 0] = gt_angles[:, 0:1].expand(-1, self.num_rotations)  # psi (second Z rotation)
        pred_angles[:, :, 1] = gt_angles[:, 1:2].expand(-1, self.num_rotations)  # theta (Y rotation)
        pred_angles[:, :, 2] = gt_angles[:, 2:3].expand(-1, self.num_rotations)  # phi (first Z rotation)
        
        
        # Convert to rotation matrices
        pred_rotmat = pytorch3d.transforms.euler_angles_to_matrix(
            pred_angles.view(-1, 3), 'ZYZ'
        ).view(B, self.num_rotations, 3, 3)


        # Remove pose inference altogether, try using explicit again
        # Expected: rotationally averaged
        
        # Reshape for decoder
        pred_rotmat = pred_rotmat.view(-1, 3, 3)

        # decoder
        out_dict = self.pred_map(pred_rotmat, r=r)
        pred_fproj_prectf = out_dict['pred_fproj_prectf']
        mask = out_dict['mask']
        _, _, H, W = pred_fproj_prectf.shape
        pred_fproj_prectf = pred_fproj_prectf.view(B, self.num_rotations, H, W)
        expanded_mask = mask.repeat(B, 1, 1, 1)

        # ctf
        if self.experimental:
            ctf = self.ctf.compute_ctf(in_dict['idx'])
            pred_fproj = pred_fproj_prectf * ctf
        else:
            pred_ctf_params = {k: in_dict[k] for k in ('defocusU', 'defocusV', 'angleAstigmatism')
                            if k in in_dict}
            pred_fproj = self.ctf(
                pred_fproj_prectf,
                in_dict['idx'],
                pred_ctf_params,
                mode='gt',
                frequency_marcher=None
            )

        output_dict = {'rotmat': pred_rotmat,
                       'pred_fproj': pred_fproj,
                       'pred_fproj_prectf': pred_fproj_prectf,
                       'mask': expanded_mask}

        return output_dict
    
    def forward_unamortized(self, in_dict, pred_rotmat, r=None):
        # decoder
        B = pred_rotmat.shape[0]
        out_dict = self.pred_map(pred_rotmat, r=r)
        pred_fproj_prectf = out_dict['pred_fproj_prectf']
        mask = out_dict['mask']
        expanded_mask = mask.repeat(B, 1, 1, 1)

        # ctf
        if self.experimental:
            ctf = self.ctf.compute_ctf(in_dict['idx'])
            pred_fproj = pred_fproj_prectf * ctf
        else:
            pred_ctf_params = {k: in_dict[k] for k in ('defocusU', 'defocusV', 'angleAstigmatism')
                            if k in in_dict}
            pred_fproj = self.ctf(
                pred_fproj_prectf,
                in_dict['idx'],
                pred_ctf_params,
                mode='gt',
                frequency_marcher=None
            )
    
        output_dict = {'rotmat': pred_rotmat,
                       'pred_fproj': pred_fproj,
                       'pred_fproj_prectf': pred_fproj_prectf,
                       'mask': expanded_mask}

        return output_dict
    
    def forward(self, in_dict, r=None, amortized=True, pred_rotmat=None):
        if amortized:
            return self.forward_amortized(in_dict, r=r)
        else:
            assert pred_rotmat is not None
            return self.forward_unamortized(in_dict, pred_rotmat, r=r)