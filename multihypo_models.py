import torch
import torch.nn as nn
import numpy as np
import pytorch3d.transforms
from encoders import CNNEncoderVGG16, GaussianPyramid
from utils.nets import FCBlock, UnitCirclePhiRegressor, ResidualAngleRegressor, GatedResidualAngleRegressor
from decoders import Explicit3D, ImplicitFourierVolume
from utils.ctf import CTFRelion
from utils.real_ctf import ExperimentalCTF

def angular_difference(ang1, ang2, angle_type):
    """
    Compute the minimum angular difference accounting for periodicity and angle range.
    
    Parameters:
    -----------
    ang1, ang2 : torch.Tensor
        Angles to compute difference between
    angle_type : str
        'psi', 'theta', or 'phi' to indicate which Euler angle is being compared
        
    Returns:
    --------
    diff : torch.Tensor
        The smallest angular difference respecting the periodic boundaries
    """
    if angle_type in ['in_mem', 'in_plane']:  # For angles with range [-π, π]
        diff = ang2 - ang1
        # Use torch.where instead of direct comparison
        diff = torch.where(diff > np.pi, diff - 2 * np.pi, diff)
        diff = torch.where(diff < -np.pi, diff + 2 * np.pi, diff)
    elif angle_type == 'tilt':  # For angles with range [0, π]
        # For theta, we need special handling since it's in [0, π]
        diff = ang2 - ang1
    else:
        raise ValueError(f"Unknown angle_type: {angle_type}")
        
    return diff


class CryoSAPIENCE(nn.Module):
    def __init__(self,  
                 num_rotations, 
                 ctf_params=None,
                 sidelen=128,
                 vol_rep='explicit',
                 num_octaves=4,
                 hartley=True,
                 data_type='real',
                 amortized_method='split_reg'):
        super(CryoSAPIENCE, self).__init__()
        self.num_rotations = num_rotations
        self.sidelen = sidelen
        self.vol_rep = vol_rep
        self.hartley = hartley
        self.data_type = data_type
        self.amortized_method = amortized_method

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

        if amortized_method == 's2s2' or amortized_method == 'penalty_s2s2':
            # Orientation regressor
            self.latent_to_rot3d_fn = pytorch3d.transforms.rotation_6d_to_matrix
            self.orientation_dims = 6
            self.orientation_regressor = nn.ModuleList()
            for _ in range(self.num_rotations):
                # We split the regressor in 2 to have access to the latent code
                self.orientation_regressor.append(FCBlock(
                    in_features=latent_code_size,
                    out_features=self.orientation_dims,
                    features=[512, 256],
                    nonlinearity='relu',
                    last_nonlinearity=None,
                    batch_norm=True,
                    group_norm=0)
                )
        elif amortized_method == 'euler' or amortized_method == 'penalty_euler':
            # Euler regressor
            self.orientation_dims = 3
            self.euler_regressor = nn.ModuleList([
                FCBlock(
                    in_features=latent_code_size,
                    out_features=self.orientation_dims,
                    features=[256, 128],
                    nonlinearity='relu',
                    last_nonlinearity=None,
                    batch_norm=True
                ) for _ in range(self.num_rotations)
            ])

        elif amortized_method == 'split_reg' or amortized_method == 'none':
            # self.known_angle_regressor = FCBlock(
            #     in_features=latent_code_size,
            #     out_features=2,  # delta_theta, delta_psi
            #     features=[256, 128],
            #     nonlinearity='relu',
            #     last_nonlinearity='tanh',  # outputs in [-1, 1]
            #     batch_norm=True
            # )
            # self.angle_scale_factor = 1/18 * np.pi

            # self.phi_regressor = nn.ModuleList([
            #     FCBlock(
            #         in_features=latent_code_size,
            #         out_features=1,  # phi prediction
            #         features=[256, 128],
            #         nonlinearity='relu',
            #         last_nonlinearity=None,
            #         batch_norm=True
            #     ) for _ in range(self.num_rotations)
            # ])

            self.phi_regressor = nn.ModuleList([
                UnitCirclePhiRegressor(latent_code_size)
                for _ in range(self.num_rotations)
            ])

            # self.residual_angle_regressor = ResidualAngleRegressor(
            #     latent_dim=latent_code_size,  # from your CNN encoder
            #     angle_dim=2,  # for theta and psi, for instance
            #     hidden_features=[256, 128],
            #     max_delta=np.deg2rad(10)  # 10° in radians
            # )

            self.gated_residual_angle_regressor_rot = GatedResidualAngleRegressor(
                latent_dim=latent_code_size,
                angle_dim=1,
                hidden_features=[256, 128],
                max_delta=np.deg2rad(10)
            )
            self.gated_residual_angle_regressor_tilt = GatedResidualAngleRegressor(
                latent_dim=latent_code_size,
                angle_dim=1,
                hidden_features=[256, 128],
                max_delta=np.deg2rad(10)
            )
        if data_type == 'real':
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
        self.experimental = (data_type == 'real')

    def forward_amortized(self, in_dict, r):
        batch_size = in_dict['proj_input'].shape[0]
        if self.amortized_method == 's2s2' or self.amortized_method == 'penalty_s2s2':
            # encoder
            proj = in_dict['proj_input']
            proj = self.gaussian_filters(proj)
            latent_code = torch.flatten(self.cnn_encoder(proj), start_dim=1)
            all_latent_code_prerot = []
            for orientation_regressor in self.orientation_regressor:
                latent_code_prerot = orientation_regressor(latent_code)
                all_latent_code_prerot.append(latent_code_prerot)
            all_latent_code_prerot = torch.stack(all_latent_code_prerot, dim=1)
            pred_rotmat = self.latent_to_rot3d_fn(all_latent_code_prerot)
            pred_rotmat = pred_rotmat.view(-1, 3, 3)

            pred_euler_angles = pytorch3d.transforms.matrix_to_euler_angles(pred_rotmat, 'ZYZ')
            pred_euler_angles = pred_euler_angles.view(batch_size, self.num_rotations, 3)

            pred_euler_angles[:, :, 0] = (pred_euler_angles[:, :, 0] + np.pi) % (2 * np.pi) - np.pi
            pred_euler_angles[:, :, 1] = (pred_euler_angles[:, :, 1]) % np.pi
            pred_euler_angles[:, :, 2] = (pred_euler_angles[:, :, 2] + np.pi) % (2 * np.pi) - np.pi

            # inital comparison
            init_angles = pytorch3d.transforms.matrix_to_euler_angles(in_dict['init_rots'], 'ZYZ')  # [B, 3]

            

            # Reshape predicted Euler angles to shape [B, num_rotations, 3]
            init_psi_diff = pred_euler_angles[:, :, 0] - init_angles[:, 0].unsqueeze(1)
            init_theta_diff = pred_euler_angles[:, :, 1] - init_angles[:, 1].unsqueeze(1)
            init_phi_diff = pred_euler_angles[:, :, 2] - init_angles[:, 2].unsqueeze(1)

            # gt comparison
            gt_angles = pytorch3d.transforms.matrix_to_euler_angles(in_dict['gt_rots'], 'ZYZ')  # [B, 3]
            # Reshape predicted Euler angles to shape [B, num_rotations, 3]
            gt_psi_diff = pred_euler_angles[:, :, 0] - gt_angles[:, 0].unsqueeze(1)
            gt_theta_diff = pred_euler_angles[:, :, 1] - gt_angles[:, 1].unsqueeze(1)
            gt_phi_diff = pred_euler_angles[:, :, 2] - gt_angles[:, 2].unsqueeze(1)

        elif self.amortized_method == 'euler' or self.amortized_method == 'penalty_euler':
            # Process input projection through Gaussian filters and encoder.
            proj = in_dict['proj_input']
            proj = self.gaussian_filters(proj)
            latent_code = torch.flatten(self.cnn_encoder(proj), start_dim=1)
            
            # Predict Euler angles directly using the Euler regressors.
            # Each regressor in the ModuleList outputs a tensor of shape [B, 3].
            pred_euler_list = [regressor(latent_code) for regressor in self.euler_regressor]
            # Stack predictions along a new dimension to get shape [B, num_rotations, 3]
            pred_euler_angles = torch.stack(pred_euler_list, dim=1)

            pred_euler_angles[:, :, 0] = (pred_euler_angles[:, :, 0] + np.pi) % (2 * np.pi) - np.pi
            pred_euler_angles[:, :, 1] = (pred_euler_angles[:, :, 1]) % np.pi
            pred_euler_angles[:, :, 2] = (pred_euler_angles[:, :, 2] + np.pi) % (2 * np.pi) - np.pi
            
            # Extract initial and ground truth Euler angles from input rotation matrices.
            init_angles = pytorch3d.transforms.matrix_to_euler_angles(in_dict['init_rots'], 'ZYZ')  # Shape: [B, 3]
            gt_angles   = pytorch3d.transforms.matrix_to_euler_angles(in_dict['gt_rots'], 'ZYZ')    # Shape: [B, 3]
            
            # Compute differences for the initial angles and for the ground truth.
            init_psi_diff   = pred_euler_angles[:, :, 0] - init_angles[:, 0].unsqueeze(1)
            init_theta_diff = pred_euler_angles[:, :, 1] - init_angles[:, 1].unsqueeze(1)
            init_phi_diff   = pred_euler_angles[:, :, 2] - init_angles[:, 2].unsqueeze(1)
            
            gt_psi_diff     = pred_euler_angles[:, :, 0] - gt_angles[:, 0].unsqueeze(1)
            gt_theta_diff   = pred_euler_angles[:, :, 1] - gt_angles[:, 1].unsqueeze(1)
            gt_phi_diff     = pred_euler_angles[:, :, 2] - gt_angles[:, 2].unsqueeze(1)
            
            # Convert the predicted Euler angles into rotation matrices.
            B = latent_code.shape[0]
            pred_rotmat = pytorch3d.transforms.euler_angles_to_matrix(
                pred_euler_angles.view(-1, 3), 'ZYZ'
            ).view(B, self.num_rotations, 3, 3)
            # Flatten to a 3D tensor [B*num_rotations, 3, 3] for the decoder.
            pred_rotmat = pred_rotmat.view(-1, 3, 3)

        elif self.amortized_method == 'split_reg' or self.amortized_method == 'none':
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

            # Combine with known angles from input
            init_angles = pytorch3d.transforms.matrix_to_euler_angles(in_dict['init_rots'], 'ZYZ')  # [B, 3]
            # outputs rot tilt psi
            
            # Get delta_theta and delta_psi predictions and scale them
            refined_angles_rot = self.gated_residual_angle_regressor_rot(latent_code, init_angles[:, 0:1])
            refined_angles_tilt = self.gated_residual_angle_regressor_tilt(latent_code, init_angles[:, 1:2])
            # Get multiple phi predictions
            # all_phi = []
            # for phi_reg in self.phi_regressor:
            #     phi = phi_reg(latent_code)  # Shape: [B, 1]
            #     all_phi.append(phi)
            # all_phi = torch.stack(all_phi, dim=1)  # Shape: [B, num_rotations, 1]


            all_phi = []
            for phi_reg in self.phi_regressor:
                # Each returns a tensor of shape [B, 2]
                phi_unit = phi_reg(latent_code)
                all_phi.append(phi_unit)
            # Stack to shape [B, num_rotations, 2]
            all_phi = torch.stack(all_phi, dim=1)
            # Compute the angle using atan2: gives angle in [-pi, pi]
            phi_angles = torch.atan2(all_phi[..., 1], all_phi[..., 0])

            # Zero out deltas, and randomize phi, expect rotationally averaged

            gt_angles = pytorch3d.transforms.matrix_to_euler_angles(in_dict['gt_rots'], 'ZYZ')  # [B, 3]

            # Construct final euler angles for each hypothesis
            B = latent_code.shape[0]
            pred_angles = torch.zeros(B, self.num_rotations, 3, device=latent_code.device)

            if self.amortized_method == 'split_reg':
                # pred_angles[:, :, 0] = init_angles[:, 0:1].expand(-1, self.num_rotations) + delta_angles[:, 0:1].expand(-1, self.num_rotations)
                # pred_angles[:, :, 1] = init_angles[:, 1:2].expand(-1, self.num_rotations) + delta_angles[:, 1:2].expand(-1, self.num_rotations)  # theta (Y rotation)
                pred_angles[:, :, 0] = refined_angles_rot.expand(-1, self.num_rotations)
                pred_angles[:, :, 1] = refined_angles_tilt.expand(-1, self.num_rotations)
                # pred_angles[:, :, 2] = all_phi.squeeze(-1)
                # pred_angles[:, :, 2] = gt_angles[:, 2:3].expand(-1, self.num_rotations)
                # pred_angles[:, :, 2] = init_angles[:, 2:3].expand(-1, self.num_rotations)
                pred_angles[:, :, 2] = phi_angles
            elif self.amortized_method == 'none':
                pred_angles[:, :, 0] = init_angles[:, 0:1].expand(-1, self.num_rotations)
                pred_angles[:, :, 1] = init_angles[:, 1:2].expand(-1, self.num_rotations)
                pred_angles[:, :, 2] = init_angles[:, 2:3].expand(-1, self.num_rotations)
            
            # Convert to rotation matrices
            pred_rotmat = pytorch3d.transforms.euler_angles_to_matrix(
                pred_angles.view(-1, 3), 'ZYZ'
            ).view(B, self.num_rotations, 3, 3)

            # pred_angles_psi = (pred_angles[:, :, 0] + np.pi) % (2 * np.pi) - np.pi
            # pred_angles_tilt = (pred_angles[:, :, 1]) % np.pi
            # pred_angles_in_membrane = (pred_angles[:, :, 2] + np.pi) % (2 * np.pi) - np.pi

            # Remove pose inference altogether, try using explicit again
            # Expected: rotationally averaged
            
            # Reshape for decoder
            pred_rotmat = pred_rotmat.view(-1, 3, 3)
            # pred_euler_angles = pytorch3d.transforms.matrix_to_euler_angles(pred_rotmat, 'ZYZ')
            # pred_euler_angles = pred_euler_angles.view(batch_size, self.num_rotations, 3)

            # init comparison
            init_psi_diff = angular_difference(init_angles[:, 0].unsqueeze(1), pred_angles[:, :, 0], 'in_plane')
            init_theta_diff = angular_difference(init_angles[:, 1].unsqueeze(1), pred_angles[:, :, 1], 'tilt')
            init_phi_diff = angular_difference(init_angles[:, 2].unsqueeze(1), pred_angles[:, :, 2], 'in_mem')


            # gt comparison
            gt_psi_diff = angular_difference(gt_angles[:, 0].unsqueeze(1), pred_angles[:, :, 0], 'in_plane')
            gt_theta_diff = angular_difference(gt_angles[:, 1].unsqueeze(1), pred_angles[:, :, 1], 'tilt')
            gt_phi_diff = angular_difference(gt_angles[:, 2].unsqueeze(1), pred_angles[:, :, 2], 'in_mem')

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
                    'mask': expanded_mask,
                    'init_psi_diff': init_psi_diff,
                    'init_theta_diff': init_theta_diff,
                    'init_phi_diff': init_phi_diff,
                    'gt_psi_diff': gt_psi_diff,
                    'gt_theta_diff': gt_theta_diff,
                    'gt_phi_diff': gt_phi_diff}

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