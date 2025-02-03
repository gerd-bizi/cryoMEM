import numpy as np
import torch
import torch.nn as nn

def print_ctf_params(params: np.ndarray) -> None:
    assert len(params) == 9
    print("Image size (pix)  : {}".format(int(params[0])))
    print("A/pix             : {}".format(params[1]))
    print("DefocusU (A)      : {}".format(params[2]))
    print("DefocusV (A)      : {}".format(params[3]))
    print("Dfang (deg)       : {}".format(params[4]))
    print("voltage (kV)      : {}".format(params[5]))
    print("cs (mm)           : {}".format(params[6]))
    print("w                 : {}".format(params[7]))
    print("Phase shift (deg) : {}".format(params[8]))

    
class ExperimentalCTF(nn.Module):
    def __init__(self):
        super(ExperimentalCTF, self).__init__()
        cs_file_path_1 = '/h/bizigerd/rotation_debugging/angle_estimated_equatorial_particles_rotation_matrix.cs'
        cs_file_path_2 = '/h/bizigerd/rotation_debugging/angle_estimated_equatorial_particles_passthrough_rotation_matrix.cs'
        metadata1 = np.load(cs_file_path_1)
        metadata2 = np.load(cs_file_path_2)

        paths = metadata1['blob/path'].copy()

        # Decode paths
        paths_decoded = np.array([p.decode('utf-8') if isinstance(p, bytes) else p for p in paths])

        # Remove entries with unwanted substring in paths
        unwanted_substring = 'FoilHole_11266022_Data_10173115_10173117_20231012_030400_EER_patch_aligned_doseweighted_particles.mrc'
        indices_to_remove = np.where([unwanted_substring in path for path in paths_decoded])[0]
        mask = np.ones(len(paths_decoded), dtype=bool)
        mask[indices_to_remove] = False

        self.original_D = metadata1["blob/shape"][0][0]
        self.D = 128  # Downsampled image size
        self.original_apix = metadata1['blob/psize_A'][0]
        self.apix = self.original_apix * (self.original_D / self.D)  # Adjusted pixel size
        self.n_data = metadata1['blob/psize_A'].shape[0]
        kv = metadata2['ctf/accel_kv'].copy()
        w = metadata2['ctf/amp_contrast'].copy()
        dfu = metadata2['ctf/df1_A'].copy()
        dfv = metadata2['ctf/df2_A'].copy()
        dfang = metadata2['ctf/df_angle_rad'].copy()
        cs = metadata2['ctf/cs_mm'].copy()
        phase_shift = metadata2['ctf/phase_shift_rad'].copy()

        # Remove the indexing with mask for scalar values
        kv = kv[mask]
        w = w[mask]
        dfu = dfu[mask]
        dfv = dfv[mask]
        dfang = dfang[mask]
        cs = cs[mask]
        phase_shift = phase_shift[mask]

        # # Calculate downsampling factor
        # downsampling_factor = self.original_D / self.D

        # # Adjust defocus values and pixel size
        # dfu = dfu * downsampling_factor
        # dfv = dfv * downsampling_factor
        # cs = cs * downsampling_factor

        l = np.linspace(-0.5, 0.5, self.D, endpoint=False)
        x0, x1 = np.meshgrid(l, l)
        freqs = np.stack([x0.ravel(), x1.ravel()], axis=-1) / self.apix

        self.register_buffer('freqs', torch.tensor(freqs).float())
        self.register_buffer('defocusU', torch.tensor(dfu).float())
        self.register_buffer('defocusV', torch.tensor(dfv).float())
        self.register_buffer('dfang', torch.tensor(dfang).float())
        self.register_buffer('kv', torch.tensor(kv).float())
        self.register_buffer('cs', torch.tensor(cs).float())
        self.register_buffer('w', torch.tensor(w).float())
        self.register_buffer('phase_shift', torch.tensor(phase_shift).float())


    def compute_ctf(
        self,
        idx,
        scalefactor=None,
        bfactor=None,
    ):
        """
        Compute the 2D CTF

        Input:
            freqs: Nx2 array of 2D spatial frequencies
            dfu: DefocusU (Angstrom)
            dfv: DefocusV (Angstrom)
            dfang: DefocusAngle (degrees)
            volt: accelerating voltage (kV)
            cs: spherical aberration (mm)
            w: amplitude contrast ratio
            phase_shift: degrees
            scalefactor : scale factor
            bfactor: envelope fcn B-factor (Angstrom^2)
        """
        # convert units
        volt = self.kv[idx] * 1000
        cs = (self.cs[idx] * 10**7)[:, None]
        dfang = self.dfang[idx][:, None]
        phase_shift = self.phase_shift[idx][:, None]
        dfu = self.defocusU[idx][:, None]
        dfv = self.defocusV[idx][:, None]
        w = self.w[idx][:, None]

        # lam = sqrt(h^2/(2*m*e*Vr)); Vr = V + (e/(2*m*c^2))*V^2
        lam = (12.2639 / torch.sqrt(volt + 0.97845e-6 * volt**2))[:, None]
        x = self.freqs[..., 0]
        y = self.freqs[..., 1]
        ang = torch.arctan2(y, x)[None] # D^2
        s2 = (x**2 + y**2)[None] # D^2
        df = 0.5 * (dfu + dfv + (dfu - dfv) * torch.cos(2 * (ang - dfang)))
        gamma = (
            2 * torch.pi * (-0.5 * df * lam * s2 + 0.25 * cs * lam**3 * s2**2)
            - phase_shift
        )
        ctf = torch.sqrt(1 - w**2) * torch.sin(gamma) - w * torch.cos(gamma)
        if scalefactor is not None:
            ctf *= scalefactor
        if bfactor is not None:
            ctf *= torch.exp(-bfactor / 4 * s2)
        ctf = ctf.reshape(-1, 1, self.D, self.D)
        return ctf
