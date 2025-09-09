# utils.py
# Levi Brunelle
# July 2025

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import math
from typing import Callable

def lens_carve(
    RoC1: float,
    RoC2: float,
    side_1_type: str,
    side_2_type: str,
    thickness: float,
    X: torch.Tensor,
    Z: torch.Tensor,
    diameter: float,
    lens_z0: float = 0.0,
    ap_inner_diameter: float = None,
    ap_thickness: float = None,
    ap_z0: float = None,
)-> tuple[torch.Tensor, torch.Tensor]:
    """
    Carves a lens with optional aperture stop and creates boolean masks to overlay in simulation.
    Uses center thickness along the optical axis, with lens_z0 defined at the on-axis apex of the first surface.

    Args
    ----
        RoC1: float
            Radius of curvature of the first surface
        RoC2: float
            Radius of curvature of the second surface
        side_1_type: str
            Type of curvature of the first surface (concave, convex, flat)
        side_2_type: str
            Type of curvature of the second surface (concave, convex, flat)
        thickness: float
            Center thickness of the lens
        X: torch.Tensor
            2D meshgrid of x coordinates
        Z: torch.Tensor
            2D meshgrid of z coordinates
        diameter: float
            Lens diameter
        lens_z0: float
            z-position of the first-surface apex at x=0
        ap_inner_diameter: float
            ID of the aperture or None if no aperture
        ap_thickness: float
            Thickness of the aperture or None
        ap_z0: float
            z-position of the aperture start or None

    Returns
    -------
        lens_mask: torch.Tensor
            Boolean mask of the carved lens volume
        ap_mask: torch.Tensor
            Boolean mask of the aperture stops
    """
    radius = diameter / 2.0

    def compute_sag(RoC, X):
        return RoC - torch.sqrt(torch.clamp(RoC**2 - X**2, min=0.0))

    def compute_delta(RoC, X, surface_type):
        if surface_type == "flat" or np.isinf(RoC):
            return torch.zeros_like(X)

        sag = compute_sag(RoC, X)
        if surface_type == "convex":
            # edges carved deepest
            return sag
        # concave: center deepest → invert sag
        sag_edge = float(RoC - math.sqrt(RoC**2 - radius**2))
        return sag_edge - sag

    # raw carve depths for front and back
    Δ1 = compute_delta(RoC1, X, side_1_type)
    Δ2 = compute_delta(RoC2, X, side_2_type)

    # center carve depths
    cx = X.shape[0] // 2
    Δ1_c = Δ1[cx, 0]
    Δ2_c = Δ2[cx, 0]

    # front surface: apex at lens_z0
    z_front = lens_z0 + (Δ1 - Δ1_c)

    # back surface: apex at lens_z0 + thickness, carved by Δ2
    z_back = lens_z0 + thickness - (Δ2 - Δ2_c)

    # carve mask between surfaces
    lens_mask = (Z >= z_front) & (Z <= z_back) & (X.abs() <= radius)

    # --- aperture stop ---
    if ap_inner_diameter is None or ap_thickness is None or ap_z0 is None:
        ap_stop = torch.zeros_like(X, dtype=torch.bool)
    else:
        top = (
            (X >= (ap_inner_diameter / 2))
            & (Z >= ap_z0)
            & (Z <= ap_z0 + ap_thickness)
        )
        bot = (
            (X <= -(ap_inner_diameter / 2))
            & (Z >= ap_z0)
            & (Z <= ap_z0 + ap_thickness)
        )
        ap_stop = top | bot

    return lens_mask, ap_stop

def smooth_refractive_index_conv(
        n: torch.Tensor, 
        pixels_filtering: int = 20,
    )-> torch.Tensor:
    """
    Smooth the refractive index map with a 1D convolution along z.

    Args
    ----
        n: torch.Tensor
            2D tensor of refractive index values [nx, nz]

        pixels_filtering: int
            Number of pixels to filter over in z direction

    Returns
    -------
        n_smoothed: torch.Tensor
            Smoothed refractive index map [nx, nz]
    """

    # Setup 1D Gaussian kernel
    kernel_size = pixels_filtering * 2 + 1
    sigma = pixels_filtering / 2.0

    t = (torch.arange(kernel_size, dtype=torch.float32, device=n.device) - pixels_filtering)

    kernel_1d = torch.exp(-(t**2) / (2 * sigma**2)) # makes the gaussian curve
    kernel_1d /= kernel_1d.sum()
    kernel = kernel_1d.view(1, 1, -1)  # conv_1D expects [out_channels, in_channels, kernel_size]

    def smooth_along_z(n_tensor):
        # n_tensor: [nx, nz]
        x = n_tensor.unsqueeze(1)           
        n_smoothed = F.conv1d(x, kernel, padding="same")
        return n_smoothed.squeeze(1)           

    if torch.is_complex(n):
        return smooth_along_z(n.real) + 1j * smooth_along_z(n.imag)
    else:
        return smooth_along_z(n)

def make_plane_wave(
        x: torch.Tensor, 
        wavelength: float, 
        theta_deg: float = 0.0, 
        z0: float = 0.0, 
        A: float = 1.0, 
        device: str = "cpu"
    ) -> torch.Tensor:
    """
    Generates a tilted plane wave along x.

    Args
    -----
        x: torch.Tensor
            1D x-axis [nx]
        wavelength: float
            Wavelength in meters
        theta_deg: float
            Illumination angle in degrees (positive = from left)
        z0: float
            Initial z position (default 0)
        A: float
            Amplitude (default 1.0)
        device: str
            Device string (e.g., 'cuda' or 'cpu')

    Returns
    -------
        U: torch.Tensor, cfloat 
            Complex field U0(x)

        U0: torch.Tensor, cfloat
            Complex field at z=0
    """
    theta = torch.tensor(np.deg2rad(theta_deg), dtype=torch.float32, device=device)

    # find wavenumber
    k = torch.tensor((2 * torch.pi / wavelength), dtype=torch.float32, device=device)

    # calculate phase
    x = x.to(device)
    phase = k * (x * torch.sin(theta) + z0 * torch.cos(theta))

    # calculate initial field
    U0 = A * torch.exp(1j * phase)

    return U0

def BPM(
        U0: torch.Tensor, 
        n_grid: torch.Tensor, 
        x: torch.Tensor, 
        z: torch.Tensor, 
        wavelength: float,
        pad_x: int = 0,       # how many rows to pad at each edge (default 0 for no padding)
        alpha: float = 5000,  # absorption strength for zero-padding
        show_progress=True, 
        running: Callable[[], bool] = lambda: True
    ) -> torch.Tensor:
    """
    Propagates the field U0 using the Beam Propagation Method (BPM)

    Args
    -----
        U0: torch.Tensor, cfloat
            Initial field at z=0 [nx]
        n_grid: torch.Tensor
            Refractive index grid [nx, nz]
        x: torch.Tensor 
            1D x-axis [nx]
        z: torch.Tensor
            1D z-axis [nz]
        wavelength: float
            Wavelength in meters
        pad_x: int
            How many rows to pad at each edge (default 0 for no padding)
        alpha: float
            Absorption strength for zero-padding (default 5000)
        show_progress: bool
            Whether to show a progress bar
        running: Callable[[], bool]
            Function to check if the simulation has been stopped


    Returns
    -------
        U: torch.Tensor, cfloat
            Propagated field [nx, nz]
    """

    nx, nz = n_grid.shape
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    k0 = 2 * torch.pi / wavelength
    absorption_factor = torch.exp(-alpha * dz)


    # Frequency domain axis
    fx = torch.fft.fftfreq(nx, d=dx.item()).to(U0.device)
    kx = 2 * torch.pi * fx

    # Paraxial propagation transfer function
    H = torch.exp(-1j * (kx**2) / (2 * k0) * dz)

    # Initialize field
    U = torch.zeros((nx, nz), dtype=torch.cfloat, device=U0.device)
    U[:, 0] = U0

    # tqdm progress bar
    iterator = range(1, nz)
    if show_progress:
        iterator = tqdm(iterator, desc="BPM Progress", unit="slice")

    
    for zi in iterator:
        # check to continue
        if not running():
            break

        U_prev = U[:, zi - 1]

        # Step 1: Propagate in k-space
        U_fft = torch.fft.fft(U_prev)
        U_fft *= H
        U_propagated = torch.fft.ifft(U_fft)

        # Step 2: Phase shift due to refractive index
        phase_shift = torch.exp(1j * k0 * (n_grid[:, zi] - 1.0) * dz)
        U[:, zi] = U_propagated * phase_shift

        if pad_x:
            # absorb top & bottom pad_x rows
            U[:pad_x, zi]    *= absorption_factor
            U[-pad_x:, zi]   *= absorption_factor

    return U

def WPM(
        U0: torch.Tensor, 
        n_grid: torch.Tensor, 
        x: torch.Tensor, 
        z: torch.Tensor, 
        wavelength: float,
        N_background: float = 1.0,
        pad_x: int = 0,       
        alpha: float = 5000,  
        show_progress=True,
        running: Callable[[], bool] = lambda: True
    ) -> torch.Tensor:
    """
    Wave Propagation Method using Angular Spectrum + refractive index phase shifts.

    Args
    -----
        U0: torch.Tensor, cfloat
            Initial field at z=0 [nx]
        n_grid: torch.Tensor
            Refractive index grid [nx, nz]
        x: torch.Tensor 
            1D x-axis [nx]
        z: torch.Tensor
            1D z-axis [nz]
        wavelength: float
            Wavelength in meters
        N_background: float
            Background refractive index (default 1.0)
        pad_x: int
            How many rows to pad at each edge (default 0 for no padding)
        alpha: float
            Absorption strength for zero-padding (default 5000)
        show_progress: bool
            Whether to show a progress bar
        running: Callable[[], bool]
            Function to check if the simulation has been stopped


    Returns
    -------
        U: torch.Tensor, cfloat
            Propagated field [nx, nz]
    """

    nx, nz = n_grid.shape
    dx = x[1] - x[0]
    dz = z[1] - z[0]
    k0 = 2 * torch.pi / wavelength
    absorption_factor = torch.exp(-alpha * dz)

    device = U0.device

    # Precompute free‐space transfer function H for one dz step
    fx = torch.fft.fftfreq(nx, d=dx.item()).to(device)          # x‐axis spatial frequencies
    kx = 2 * torch.pi * fx                                      # angular frequencies
    kz = torch.sqrt((k0 * N_background) ** 2 - kx**2 + 0j)      # axial wavevector component
    H = torch.exp(1j * kz * dz)                                 # propagation factor

    # Initialize output field array and plug in the input slice
    U = torch.zeros((nx, nz), dtype=torch.cfloat, device=device)
    U[:, 0] = U0

    # tqdm progress bar
    iterator = range(1, nz)
    if show_progress:
        iterator = tqdm(iterator, desc="WPM Progress", unit="slice")

    for zi in iterator:
        #check to continue
        if not running():
            break

        # 1) Free‐space propagation via FFT → multiply H → IFFT
        U_fft = torch.fft.fft(U[:, zi - 1])
        U_fft *= H
        U_prop = torch.fft.ifft(U_fft)

        # 2) Phase shift from local refractive index
        delta_n = n_grid[:, zi] - N_background
        phase_shift = torch.exp(1j * k0 * delta_n * dz)
        U[:, zi] = U_prop * phase_shift

        if pad_x:
            # absorb top & bottom pad_x rows
            U[:pad_x, zi]    *= absorption_factor
            U[-pad_x:, zi]   *= absorption_factor

    return U

