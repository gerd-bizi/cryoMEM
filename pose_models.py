import torch
import torch.nn as nn
import numpy as np
from utils.so3_exp import SO3Exp
from pytorch3d.transforms import matrix_to_euler_angles, euler_angles_to_matrix
import math

def generate_random_axis(n_points):
    u = np.random.normal(loc=0, scale=1, size=(n_points, 3))
    u /= np.sqrt(np.sum(u ** 2, axis=-1, keepdims=True))
    return u

class RotModel(nn.Module):
    def __init__(self, n_data, rotations=None):
        super(RotModel, self).__init__()
        self.rotations = torch.nn.Parameter(rotations, requires_grad=False)
        # Initialize perturbations with zeros - these are axis-angle representations
        self.perturbations_w = torch.nn.Parameter(torch.zeros(n_data, 3), requires_grad=True)
        self.so3exp_func = SO3Exp.apply
    
    def update(self, idx):
        with torch.no_grad():
            # Apply current perturbations to update the rotations
            delta_rot = self.so3exp_func(self.perturbations_w[idx])
            self.rotations.data[idx] = torch.matmul(delta_rot, self.rotations.data[idx])
            
            # Reset perturbations to allow fresh optimization from new position
            self.perturbations_w.data[idx].zero_()
    
    def forward(self, idx):
        # Apply current perturbations (which may be partially optimized)
        perturb_rotations = self.so3exp_func(self.perturbations_w[idx])
        # Combine with the stored rotations
        return torch.matmul(perturb_rotations, self.rotations[idx])


class CircularEulerRotModel(nn.Module):
    def __init__(self, n_data, rotations=None):
        super(CircularEulerRotModel, self).__init__()
        # Convert initial rotation matrices to Euler angles (ZYZ convention)
        euler_angles = matrix_to_euler_angles(rotations, convention="ZYZ")
        
        # Create parameters for circular representation of angles
        self.n_data = n_data
        
        # For psi (in-plane rotation, first angle)
        psi = euler_angles[:, 0]
        self.psi_cos = nn.Parameter(torch.cos(psi), requires_grad=True)
        self.psi_sin = nn.Parameter(torch.sin(psi), requires_grad=True)
        
        # For theta (tilt angle, second angle) - direct parameterization
        self.theta = nn.Parameter(euler_angles[:, 1], requires_grad=True)
        
        # For phi (in-membrane rotation, third angle)
        phi = euler_angles[:, 2]
        self.phi_cos = nn.Parameter(torch.cos(phi), requires_grad=True)
        self.phi_sin = nn.Parameter(torch.sin(phi), requires_grad=True)
    
    def normalize_circular_params(self, idx=None):
        """Renormalize vectors to maintain unit circle constraint"""
        with torch.no_grad():
            if idx is None:
                # Normalize psi parameters
                psi_norm = torch.sqrt(self.psi_cos**2 + self.psi_sin**2)
                self.psi_cos.data = self.psi_cos.data / psi_norm
                self.psi_sin.data = self.psi_sin.data / psi_norm
                
                # Normalize phi parameters
                phi_norm = torch.sqrt(self.phi_cos**2 + self.phi_sin**2)
                self.phi_cos.data = self.phi_cos.data / phi_norm
                self.phi_sin.data = self.phi_sin.data / phi_norm
                
                # Clamp theta to valid range [0, π]
                self.theta.data = torch.clamp(self.theta.data, 0.0, math.pi)
            else:
                # Only normalize for the specified indices
                psi_norm = torch.sqrt(self.psi_cos[idx]**2 + self.psi_sin[idx]**2)
                self.psi_cos.data[idx] = self.psi_cos.data[idx] / psi_norm
                self.psi_sin.data[idx] = self.psi_sin.data[idx] / psi_norm
                
                phi_norm = torch.sqrt(self.phi_cos[idx]**2 + self.phi_sin[idx]**2)
                self.phi_cos.data[idx] = self.phi_cos.data[idx] / phi_norm
                self.phi_sin.data[idx] = self.phi_sin.data[idx] / phi_norm
                
                self.theta.data[idx] = torch.clamp(self.theta.data[idx], 0.0, math.pi)
    
    def get_euler_angles(self, idx=None):
        """Recover Euler angles from the circular parameterization"""
        if idx is None:
            psi = torch.atan2(self.psi_sin, self.psi_cos)
            theta = self.theta
            phi = torch.atan2(self.phi_sin, self.phi_cos)
        else:
            psi = torch.atan2(self.psi_sin[idx], self.psi_cos[idx])
            theta = self.theta[idx]
            phi = torch.atan2(self.phi_sin[idx], self.phi_cos[idx])
        
        return torch.stack([psi, theta, phi], dim=-1)
    
    def forward(self, idx):
        """Convert current parameters to rotation matrices"""
        # Get Euler angles from our parameterization
        euler_angles = self.get_euler_angles(idx)
        
        # Convert to rotation matrices using PyTorch3D
        return euler_angles_to_matrix(euler_angles, convention="ZYZ")

class ShiftModel(nn.Module):
    def __init__(self, shifts=None, shift_grad=True):
        super(ShiftModel, self).__init__()
        self.shifts = torch.nn.Parameter(shifts, requires_grad=shift_grad)
    
    def forward(self, idx):
        return self.shifts[idx]
        
class PerturbationEulerRotModel(nn.Module):
    def __init__(self, n_data, rotations=None):
        super(PerturbationEulerRotModel, self).__init__()
        # Convert initial rotation matrices to Euler angles (ZYZ convention)
        euler_angles = matrix_to_euler_angles(rotations, convention="ZYZ")
        
        # Base parameters (non-perturbed)
        self.n_data = n_data
        
        # For psi (in-plane rotation)
        psi = euler_angles[:, 0]
        self.psi_cos = nn.Parameter(torch.cos(psi), requires_grad=False)
        self.psi_sin = nn.Parameter(torch.sin(psi), requires_grad=False)
        
        # For theta (tilt angle)
        self.theta = nn.Parameter(euler_angles[:, 1], requires_grad=False)
        
        # For phi (in-membrane rotation)
        phi = euler_angles[:, 2]
        self.phi_cos = nn.Parameter(torch.cos(phi), requires_grad=False)
        self.phi_sin = nn.Parameter(torch.sin(phi), requires_grad=False)
        
        # Perturbation parameters that will be optimized
        self.psi_delta = nn.Parameter(torch.zeros(n_data), requires_grad=True)
        self.theta_delta = nn.Parameter(torch.zeros(n_data), requires_grad=True)
        self.phi_delta = nn.Parameter(torch.zeros(n_data), requires_grad=True)
        
    def update(self, idx):
        with torch.no_grad():
            # Apply perturbations to the base angles
            psi = torch.atan2(self.psi_sin[idx], self.psi_cos[idx]) + self.psi_delta[idx]
            self.psi_cos.data[idx] = torch.cos(psi)
            self.psi_sin.data[idx] = torch.sin(psi)
            
            # Update theta with clamping
            self.theta.data[idx] = torch.clamp(self.theta[idx] + self.theta_delta[idx], 0.0, math.pi)
            
            # Update phi
            phi = torch.atan2(self.phi_sin[idx], self.phi_cos[idx]) + self.phi_delta[idx]
            self.phi_cos.data[idx] = torch.cos(phi)
            self.phi_sin.data[idx] = torch.sin(phi)
            
            # Reset perturbations to zero for next iteration
            self.psi_delta.data[idx].zero_()
            self.theta_delta.data[idx].zero_()
            self.phi_delta.data[idx].zero_()
    
    def get_euler_angles(self, idx=None):
        if idx is None:
            # Get base angles
            psi_base = torch.atan2(self.psi_sin, self.psi_cos)
            theta_base = self.theta
            phi_base = torch.atan2(self.phi_sin, self.phi_cos)
            
            # Apply perturbations
            psi = psi_base + self.psi_delta
            theta = torch.clamp(theta_base + self.theta_delta, 0.0, math.pi)
            phi = phi_base + self.phi_delta
        else:
            # Same but only for specified indices
            psi_base = torch.atan2(self.psi_sin[idx], self.psi_cos[idx])
            theta_base = self.theta[idx]
            phi_base = torch.atan2(self.phi_sin[idx], self.phi_cos[idx])
            
            psi = psi_base + self.psi_delta[idx]
            theta = torch.clamp(theta_base + self.theta_delta[idx], 0.0, math.pi)
            phi = phi_base + self.phi_delta[idx]
        
        return torch.stack([psi, theta, phi], dim=-1)
    
    def forward(self, idx):
        """Convert current parameters to rotation matrices"""
        euler_angles = self.get_euler_angles(idx)
        return euler_angles_to_matrix(euler_angles, convention="ZYZ")

class PoseModel(nn.Module):
    def __init__(self, n_data, rotations=None, shifts=None, shift_grad=True, euler=False):
        super(PoseModel, self).__init__()
        self.euler = euler
        if shifts is not None:
            self.shifts = ShiftModel(shifts, shift_grad=shift_grad)
        else:
            self.shifts = None
        if euler is False:
            self.rots = RotModel(n_data, rotations)
        else:
            self.rots = CircularEulerRotModel(n_data, rotations)
    
    def update(self, idx):
        if self.euler is False:
            with torch.no_grad():
                self.rots.update(idx)
    
    def forward(self, idx):
        rots = self.rots(idx)
        if self.shifts is not None:
            shifts = self.shifts(idx)
            return rots, shifts
        else:
            return rots