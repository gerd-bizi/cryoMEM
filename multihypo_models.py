import torch
import torch.nn as nn
import numpy as np
import pytorch3d.transforms
from encoders import CNNEncoderVGG16, GaussianPyramid
from regression_nets import *
from decoders import Explicit3D
from utils.ctf import CTFRelion
from utils.real_ctf import ExperimentalCTF



class CryoMEM(nn.Module):
    def __init__(self,  
                 heads, 
                 ctf_params=None,
                 sidelen=128,
                 num_octaves=4,
                 hartley=True,
                 data_type='real',
                 amortized_method='in_membrane',
                 real_file_1=None,
                 real_file_2=None):
        super(CryoMEM, self).__init__()
        self.heads = heads
        self.sidelen = sidelen
        self.hartley = hartley
        self.data_type = data_type
        self.amortized_method = amortized_method
        self.experimental = data_type == 'real'

        self.pred_map = Explicit3D(downsampled_sz=sidelen, img_sz=sidelen, hartley=hartley)

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
                                            batch_norm=True, group_norm=0,
                                            high_res=True)
        cnn_encoder_out_shape = self.cnn_encoder.get_out_shape(sidelen, sidelen)
        latent_code_size = torch.prod(torch.tensor(cnn_encoder_out_shape))

        if amortized_method == 's2s2':
            self.s2s2_regressor = S2S2Regressor(latent_code_size)

        elif amortized_method != 'none':
            self.inmembrane_regressor = nn.ModuleList([
                InMembraneCircleRegressor(latent_code_size)
                for _ in range(self.heads)
            ])

            if self.amortized_method == 'split_reg':
                self.inplane_residual_regressor = GatedResidualAngleRegressor(
                    latent_dim=latent_code_size,
                    angle='inplane',
                    angle_dim=1,
                    hidden_features=[256, 128],
                    max_delta=np.deg2rad(10)
                )
                self.tilt_residual_regressor = GatedResidualAngleRegressor(
                    latent_dim=latent_code_size,
                    angle='tilt',
                    angle_dim=1,
                    hidden_features=[256, 128],
                    max_delta=np.deg2rad(10)
                )

            elif self.amortized_method == 'in_membrane_and_tilt':
                self.tilt_regressor = nn.ModuleList([
                    TiltRegressor(latent_code_size)
                    for _ in range(self.heads)
                ])
                
        if data_type == 'real':
            self.ctf = ExperimentalCTF(real_file_1, real_file_2)
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
            

    def forward_amortized(self, in_dict, r, current_epoch):
        proj = in_dict['proj_input']
        batch_size = proj.shape[0]
        proj = self.gaussian_filters(proj)
        latent_code = torch.flatten(self.cnn_encoder(proj), start_dim=1)
        B = latent_code.shape[0]

        if self.amortized_method == 's2s2':
            all_latent_code_prerot = self.s2s2_regressor(latent_code)  # [B, heads, 6]
            pred_rotmat = self.latent_to_rot3d_fn(all_latent_code_prerot)
            
        elif self.amortized_method == 'in_membrane' or \
             self.amortized_method == 'in_membrane_and_tilt' or \
             self.amortized_method == 'split_reg':
            
            init_angles = pytorch3d.transforms.matrix_to_euler_angles(in_dict['init_rots'], 'ZYZ')  # [B, 3]

            if self.amortized_method == 'split_reg':
                inplane_w_res = self.inplane_residual_regressor(latent_code, init_angles[:, 0:1])
                tilt_w_res = self.tilt_residual_regressor(latent_code, init_angles[:, 1:2])
            elif self.amortized_method == 'in_membrane_and_tilt':
                tilt_est = self.tilt_regressor(latent_code)
            
            inmembrane_trig_rep = []
            for inmembrane_reg in self.inmembrane_regressor:
                inmembrane_unit = inmembrane_reg(latent_code) # [B, 2]
                inmembrane_trig_rep.append(inmembrane_unit)
            
            inmembrane_trig_rep = torch.stack(inmembrane_trig_rep, dim=1) # [B, heads, 2]
            inmembrane_angles = torch.atan2(inmembrane_trig_rep[..., 1], inmembrane_trig_rep[..., 0])

            # Construct final euler angles for each hypothesis
            pred_angles = torch.zeros(B, self.heads, 3, device=latent_code.device)
            pred_angles[:, :, 2] = inmembrane_angles

            if self.amortized_method == 'in_membrane':
                pred_angles[:, :, 0] = init_angles[:, 0:1].expand(-1, self.heads)
                pred_angles[:, :, 1] = init_angles[:, 1:2].expand(-1, self.heads)
            elif self.amortized_method == 'in_membrane_and_tilt':
                pred_angles[:, :, 0] = init_angles[:, 0:1].expand(-1, self.heads)
                pred_angles[:, :, 1] = tilt_est
            elif self.amortized_method == 'split_reg':
                pred_angles[:, :, 0] = inplane_w_res.expand(-1, self.heads)
                pred_angles[:, :, 1] = tilt_w_res.expand(-1, self.heads)

        elif self.amortized_method == 'none':
            pred_angles[:, :, 0] = init_angles[:, 0:1].expand(-1, self.heads)
            pred_angles[:, :, 1] = init_angles[:, 1:2].expand(-1, self.heads)
            pred_angles[:, :, 2] = init_angles[:, 2:3].expand(-1, self.heads)

        elif self.amortized_method == 'gt':
            gt_angles = pytorch3d.transforms.matrix_to_euler_angles(in_dict['gt_rots'], 'ZYZ')
            pred_angles[:, :, 0] = gt_angles[:, 0:1].expand(-1, self.heads)
            pred_angles[:, :, 1] = gt_angles[:, 1:2].expand(-1, self.heads)
            pred_angles[:, :, 2] = gt_angles[:, 2:3].expand(-1, self.heads)

        # Convert to rotation matrices
        pred_rotmat = pytorch3d.transforms.euler_angles_to_matrix(
            pred_angles.view(-1, 3), 'ZYZ'
        ).view(B, self.heads, 3, 3)

        # Reshape for decoder
        pred_rotmat = pred_rotmat.view(-1, 3, 3)
            
        # decoder
        out_dict = self.pred_map(pred_rotmat, r=r)
        pred_fproj_prectf = out_dict['pred_fproj_prectf']
        mask = out_dict['mask']
        _, _, H, W = pred_fproj_prectf.shape
        pred_fproj_prectf = pred_fproj_prectf.view(batch_size, self.heads, H, W)
        expanded_mask = mask.repeat(batch_size, 1, 1, 1)

        output_dict = {'rotmat': pred_rotmat,
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
    
        output_dict = {'rotmat': pred_rotmat,
                       'pred_fproj_prectf': pred_fproj_prectf,
                       'mask': expanded_mask}

        return output_dict
    
    def forward(self, in_dict, r=None, amortized=True, pred_rotmat=None, current_epoch=None):
        if amortized:
            output_dict = self.forward_amortized(in_dict, r=r, current_epoch=current_epoch)
        else:
            output_dict = self.forward_unamortized(in_dict, pred_rotmat, r=r)

        pred_fproj_prectf = output_dict['pred_fproj_prectf']
        
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

        output_dict['pred_fproj'] = pred_fproj
        return output_dict