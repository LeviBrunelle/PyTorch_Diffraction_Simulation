import torch
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt




def get_signed_radius(
        radius: float, 
        side: str, 
        is_exit: bool=False
    ) -> float:
    """
    Helper function to ensure correct signage of user input based on specified type of curvature.

    Args
    ----
        radius: float
            value entered by user for the radius of curvature of lens surface
        
        side: str
            specifies if side is concave, convex, or flat
        
        is_exit: bool
            whether or not it's a front or back surface
    """

    if side == "flat":
        return float('inf')
    
    if side == "convex":
        sign = 1
    else:
        sign = -1

    if is_exit:
        sign *= -1  # reverse sign on exit surface

    return sign * radius

def format_distance(val_m: float) -> str:
    if val_m > 2e-3:
        return f"{val_m * 1e3:.3f} mm"
    else:
        return f"{val_m * 1e6:.3f} µm"

def search_focus(
        U: torch.Tensor, 
        z: torch.Tensor,
    ) -> tuple[float, torch.Tensor]:
    """
    Returns the z-position of the max intensity in the field U.

    Args
    ----
        U: torch.Tensor, cfloat
            complex intesnity field [nx, nz]

        z: torch.tensor
            1D tensor [nz]

    Returns
    -------
        z_peak_val: float
            z-position where max intensity occurs

        z_nearest_index: torch.Tensor
            index of the nearest z-coordinate to the peak
    """

    # find max intensity in U
    intensity = torch.abs(U) ** 2 
    max_vals, _ = intensity.max(dim=0)  # max along x for each z

    # get coordinate and index of peak 
    z_peak_index = torch.argmax(max_vals)
    z_peak_val = z[z_peak_index]
    z_nearest_index = (torch.abs(z - z_peak_val)).argmin()

    return z_peak_val.item(), z_nearest_index

def compute_na(
        diameter: float, 
        distance: float
    ) -> float:
    """
    Numerical aperture (NA) calculator

    Args
    ----
        diameter: float
            clear aperture of system
        
        distance: float
            either focal length or working distance of system
    
    Returns
    -------
        NA: float
            numerical aperture value
    """
    return math.sin(math.atan((diameter / 2) / abs(distance)))

def calc_focal_lengths(
        zf_real: float = 0.0, 
        lens: dict = None, 
        n_outside: float = 1.0
    ) -> tuple[dict[str, object], dict[str, object]]:
    
    """
    Calculates all the ideal focal lengths (EFL, BFL, FFL) based on geometry.
    Calculates all the real focal lengths (EFL, BFL, FFL) based on geometry and sim results.

    *** WIP, currently some values are nonsense since you might call this before a sim is run, with no zf value, 
    but it will still try to calculate real FL's. Also FFL will not make sense most of the time, but it isn't being used yet. ***


    Args
    ----
        zf: float
            z value of focal point in meters
        
        lens: dict
            dictionary of design paramaters for a lens
        
        n_outside: float
            background refractive index


    Returns
    -------
        ideal_focals: dict[str, object]
            dictionary of ideal focal lengths (EFL, BFL, FFL)

        real_focals: dict[str, object]
            dictionary of real focal lengths (EFL_front, EFL_back, FFL, BFL)

    """

    R1 = get_signed_radius(lens["r1"], lens["side1"], is_exit=False)
    R2 = get_signed_radius(lens["r2"], lens["side2"], is_exit=True)
    CT = lens["thickness"]
    n_L = lens["n_lens"]
    n_OS = n_outside
    n_IS = n_outside
    z_v1 = lens["z0"]
    z_v2 = z_v1 + CT

    # Surface powers
    phi_OS = (n_L - n_OS)/R1
    phi_IS = (n_IS - n_L)/R2
    phi    = phi_OS + phi_IS - (phi_OS * phi_IS) * (CT / n_L)

    # Plane offsets
    P_front =  (phi_IS / phi) * (n_OS / n_L) * CT       # offset of plane1 from the front (surface 1)
    P_back  = - (phi_OS / phi) * (n_IS / n_L) * CT       # offset of plane2 from the back (surface 2)

    # Z location of planes
    z_plane_1  = z_v1 + P_front
    z_plane_2 = z_v2 + P_back

    # Calculating ideal focal lengths
    EFL_ideal = 1/phi
    BFL_ideal = (n_IS * EFL_ideal) + P_back
    FFL_ideal = -(n_OS * EFL_ideal) + P_front
    zf_ideal = z_v2 + BFL_ideal


    ideal_focals = {
        "EFL": EFL_ideal, 
        "BFL": BFL_ideal, 
        "FFL": FFL_ideal,
        "zf": zf_ideal,
    }
    # Calculate real focal lengths
    EFL_real = zf_real - z_plane_2
    BFL_real = zf_real - z_v2
    FFL_real = z_v1 - zf_real
    real_focals = {
        "EFL": EFL_real, 
        "FFL": FFL_real, 
        "BFL": BFL_real,
        "zf": zf_real,

    }

    return ideal_focals, real_focals 

def calc_real_working_distance(
        zf: float, 
        lens: dict
    ) -> float:
    """
    Finds the distance between a lens exit surface and the focal point.

    Args
    ----    
        zf: float
            z coordinate of the focal point in meters
        
        lens: dict
            dictionary of paramaters for a lens

    Returns
    -------
        WD: float
            distance between a lens exit surface and the focal point
    """

    z0 = lens["z0"]
    gap = lens.get("gap_before", 0.0)
    thickness = lens["thickness"]
    z_end = z0 + gap + thickness
    return abs(zf - z_end)

def calculate_fwhm(
        x: np.ndarray, 
        I: np.ndarray
    ) -> float:
    """
    Find FWHM by bracketed search around the main peak:

    Args
    ----
        x: ndarray
            1D array of x-coordinates

        I: ndarray
            1D array of I-coordinates

    Returns
    -------
        fwhm: float 
            the width of the peak at half intensity
    """
    # 1) normalize
    I_norm = I / np.max(I)
    half = 0.5

    # 2) peak index
    idx_peak = np.argmax(I_norm)

    # 3) search left for crossing
    left_idx = idx_peak
    while left_idx > 0 and I_norm[left_idx] >= half:
        left_idx -= 1
    # linear interp between left_idx and left_idx+1
    x0, x1 = x[left_idx], x[left_idx + 1]
    I0, I1 = I_norm[left_idx], I_norm[left_idx + 1]
    if I1 != I0:
        left_cross = x0 + (half - I0) * (x1 - x0) / (I1 - I0)
    else:
        left_cross = x0

    # 4) search right for crossing
    right_idx = idx_peak
    N = len(I_norm)
    while right_idx < N - 1 and I_norm[right_idx] >= half:
        right_idx += 1
    x0, x1 = x[right_idx - 1], x[right_idx]
    I0, I1 = I_norm[right_idx - 1], I_norm[right_idx]
    if I1 != I0:
        right_cross = x0 + (half - I0) * (x1 - x0) / (I1 - I0)
    else:
        right_cross = x1

    return right_cross - left_cross

def compute_diffraction_limit_psf(
        clear_ap: float,
        na_dist: float,
        wavelength: float,
        x_um: np.ndarray,
    ) -> tuple[np.ndarray, float]:
    """
    Calculates the y datapoints for the diffraction limited psf

    Args
    ----
        clear_ap: float
            clear aperture size of the lens system

        na_dist: float
            either focal length or WD, to be used in NA calculation

        wavelength: float
            wavelength of light being propagated in the sim

        x_um: ndarray
            1D array of x coordinates for sim field
            
    Returns
    -------
      y : ndarray
        1D Gaussian PSF (normalized) on x_um grid

      fwhm_diff: float
        FWHM of that Gaussian in µm
    """
    # get actual NA from your aperture & distance
    na = compute_na(clear_ap, na_dist)

    # analytic FWHM (m) -> convert to sigma
    fwhm_m = 0.51 * wavelength / na
    sigma  = fwhm_m / 2.355

    # center x so that x=0 at the PSF peak
    idx0 = np.argmin(np.abs(x_um))
    x_m  = (x_um - x_um[idx0]) * 1e-6    

    y = np.exp(-0.5 * (x_m / sigma)**2)
    y /= y.max()

    return y, fwhm_m * 1e6 

def compute_diffraction_limit_mtf(
        clear_ap: float, 
        na_dist: float,
        wavelength: float, 
        x_um: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:

    """
    Computes an analytic MTF of a circular pupil: ideal cutoff f_c = NA/λ.

    Args
    ----
        clear_ap: float
            clear aperture size of the lens system

        na_dist: float
            either focal length or WD, to be used in NA calculation

        wavelength: float
            wavelength of light being propagated in the sim

        x_um: ndarray
            1D array of x coordinates for sim field
            
    Returns
    -------
        freqs : ndarray
            1D array of spatial frequency samples (e.g. cycles/mm)

        mtf : ndarray
            1D array of corresponding MTF values (unitless modulation depth)
    """
    NA = compute_na(clear_ap, na_dist)
    f_c = NA / wavelength             # cycles per meter
    f_um = f_c * 1e-6                      # convert to 1/um
    freqs = np.linspace(0, f_um, len(x_um)//2)
    mtf = (2/np.pi)*(np.arccos(freqs/f_um) - (freqs/f_um)*np.sqrt(1-(freqs/f_um)**2))
    return freqs, mtf

def compute_mtf_from_psf(
        psf: torch.Tensor,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

    """
    computes the mtf of our simulation, using the 1D psf

    Args
    ----
        psf: torch.Tensor
            1D slice of psf intensity values from the focal z of our simulation
        
        x: torch.Tensor
            corresponding x values for the psf

    Returns:
    --------
        freqs : torch.Tensor
            1D tensor of spatial frequency samples (e.g. cycles/mm)

        mtf : torch.Tensor
            1D tensor of corresponding MTF values (unitless modulation depth)
    """
    # normalize
    psf = psf.abs()
    psf = psf / psf.max()

    # FFT to get mtf
    mtf_raw = torch.abs(torch.fft.fft(psf))
    mtf = mtf_raw / mtf_raw.max()

    # build frequency axis in µm
    x_um = x.cpu() * 1e6                
    dx = (x_um[1] - x_um[0]).item()
    freqs = torch.from_numpy(
        np.fft.fftshift(np.fft.fftfreq(len(x_um), dx))
    )

    # shift mtf
    mtf = torch.fft.fftshift(mtf)

    # keep only positive freqs
    mask = freqs >= 0

    return freqs[mask], mtf[mask]

def plot_psf(
        x_um: np.ndarray | None = None,
        y_sim: np.ndarray | None = None,
        fwhm_sim: float | None = None,
        y_diff: np.ndarray | None = None,
        fwhm_diff: float | None = None,
        zemax_df: pd.DataFrame | None = None,
        fwhm_zemax: float | None = None,
        xlim: tuple[float,float] | None = None,
    ) -> plt.Figure:
    
    """
    Plots the PSF(s) of the simulation, the diffraction limit, and/or Zemax data.

    Args
    ----
        x_um : np.ndarray or None
            1D array of sample positions (in µm) for the simulated PSF

        y_sim : np.ndarray or None
            Simulated PSF intensity values corresponding to x_um

        fwhm_sim : float or None
            Full-width at half-maximum of the simulated PSF (same units as x_um

        y_diff : np.ndarray or None
            Diffraction-limited PSF intensity values over the same x_um axis

        fwhm_diff : float or None
            Full-width at half-maximum of the diffraction-limited PSF

        zemax_df : pd.DataFrame or None
            DataFrame loaded from a Zemax-exported CSV containing PSF or FWHM columns

        fwhm_zemax : float or None
            Full-width at half-maximum value extracted from the Zemax data

        xlim : tuple of float or None
            Optional x-axis limits for the plot as xmin, xmax

    Returns
    -------
        fig : plt.Figure
            The Matplotlib Figure containing overlaid PSF curves
    """

    fig, ax = plt.subplots(figsize=(8,6))

    if y_sim is not None:
        ax.plot(x_um, y_sim,
                lw=2, label=f"Sim PSF (FWHM={fwhm_sim:.2f} µm)")

    if y_diff is not None:
        ax.plot(x_um, y_diff, "--",
                lw=2, label=f"Diffraction Limit (FWHM={fwhm_diff:.2f} µm)")

    if isinstance(zemax_df, pd.DataFrame):
        x_z = zemax_df["Position"].values
        y_z = zemax_df["Value"].values
        y_z = y_z / np.max(y_z)
        label = "Zemax PSF"
        if fwhm_zemax is not None:
            label += f" (FWHM={fwhm_zemax:.2f} µm)"
        ax.plot(x_z, y_z, lw=2, label=label)

    ax.axhline(0.5, color="red", ls=":", label="0.5 level")
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("Normalized Intensity")
    ax.set_title("PSF Comparison")
    ax.legend(); ax.grid(True)

    if xlim:
        ax.set_xlim(xlim)

    fig.tight_layout()
    return fig

def plot_mtf(
        mtf_sim: tuple[np.ndarray, np.ndarray] | None = None, 
        mtf_diff: tuple[np.ndarray, np.ndarray] | None = None, 
        mtf_zemax: tuple[np.ndarray, np.ndarray] | None = None,  
        freq_max: float = None,
    ) -> plt.Figure:
    
    """
    Plots the mtf('s) of the simulation psf, diffraction limit psf, and/or zemax psf

    Args
    ----
        mtf_sim: tuple[np.ndarray, np.ndarray]
            datapoints for the simulated mtf
        
        mtf_diff: tuple[np.ndarray, np.ndarray]
            datapoints for the diffraction limit mtf
        
        mtf_zemax: tuple[np.ndarray, np.ndarray]
            datapoints for a zemax mtf
        freq_max: float
            maximum frequency to display on the x-axis (in cycles/µm)

    Returns
    -------
        fig: plt.Figure
            Matplotlib plot of the overlaid mtf's
    """
    
    fig, ax = plt.subplots(figsize=(8,6))

    if mtf_sim is not None:
        ax.plot(mtf_sim[0], mtf_sim[1], lw=2, label="Simulation")

    if mtf_zemax is not None:
        ax.plot(mtf_zemax[0], mtf_zemax[1], ':', lw=2, label="Zemax")

    if mtf_diff is not None:
        ax.plot(mtf_diff[0], mtf_diff[1], '--', lw=2, label="Diffraction Limit")

    ax.set_xlabel("Spatial Frequency (cycles/µm)")
    ax.set_ylabel("MTF")
    ax.set_title("Modulation Transfer Function")
    ax.legend(); ax.grid(True)

    if freq_max: 
        ax.set_xlim(0, freq_max)

    fig.tight_layout()

    return fig

def plot_simulation(
        n: torch.Tensor,
        U: torch.Tensor,
        x: torch.Tensor = None,
        z: torch.Tensor = None,
        backend: str = "cv2",          
        fast_plotting: bool = False,
        downsample_dim: int = 1500,
        dark_mode: bool = True,
        threshold: float = 1e-3,
        use_log: bool = True,
        panel: str = "all",     
        title_prefix: str = "",
    ) -> any:
    """
    Plots the simulation results using either cv2 or pyplot and returns a Figure.

    Args:
        lens_mask (torch.Tensor): Lens mask [nx, nz]
        n (torch.Tensor): Refractive index field (smoothed or not) [nx, nz]
        U (torch.Tensor): Complex field [nx, nz]
        x (torch.Tensor): 1D tensor of x axis values 
        z (torch.Tensor): 1D tensor of z axis values
        backend (str): selects between "cv2" and "pyplot"
        fast_plotting(bool): downsample if True
        downsample_dim (int): max dimension for downsampled image
        dark_mode (bool): clamp intensity floor
        threshold (float): for darkmode, intensity level below which everything is set to black
        use_log: toggle for logarithmic intensity scaling
        panel (str): "all" for 3-panel, "intensity" for single plot
        title_prefix(str): title prefix

    Returns:
        if backend=="pyplot": matplotlib.Figure

        if backend=="cv2": (combined_mask_index_img, intensity_img)
    """

    # --- Shared helpers ---
    def downsample(t: torch.Tensor) -> torch.Tensor:
        H, W = t.shape[:2]
        stride = max(1, math.ceil(max(H, W) / downsample_dim))
        return t[::stride, ::stride]

    def to_uint8(img: torch.Tensor) -> np.ndarray:
        a = img.real.cpu().detach().numpy().astype(np.float32)
        a -= a.min()
        a /= a.max() + 1e-8
        return (a * 255).astype(np.uint8)

    # --- Preprocess arrays ---
    #mask_torch  = downsample(lens_mask)   
    index_torch = downsample(n)  

    I = torch.abs(U)**2
    I /= I.max()
    if dark_mode:
        I = torch.clamp(I, min=threshold)

    if use_log:
        I = torch.log10(I + 1e-10)
        I -= I.min()
        I /= I.max() + 1e-6
        intensity_label = f"{title_prefix} log10(Intensity)"
    else:
        I = I / (I.max() + 1e-6)
        intensity_label = f"{title_prefix} Intensity"

    intensity_torch = downsample(I) if fast_plotting else I

    # --- Branch on backend ---
    if backend == "cv2":
        # mask + index side by side
        def color_cv2(tensor, cmap, label, pos):
            u8 = to_uint8(tensor)
            cimg = cv2.applyColorMap(u8, cmap)
            cv2.putText(
                cimg, f"{title_prefix} {label}", pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2
            )
            return cimg

        #mask_img  = color_cv2(mask_torch,  cv2.COLORMAP_BONE,   "Lens Mask",  (30,40))
        index_img = color_cv2(index_torch, cv2.COLORMAP_VIRIDIS,"n (smoothed)",(30,40))

        # intensity
        int_img = to_uint8(intensity_torch)
        int_img = cv2.applyColorMap(int_img, cv2.COLORMAP_INFERNO)
        cv2.putText(
            int_img, intensity_label, (30,40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2
        )

        # stack mask & index
        # H = min(mask_img.shape[0], index_img.shape[0])
        # W = min(mask_img.shape[1], index_img.shape[1])
        # combined = np.hstack([mask_img[:H,:W], index_img[:H,:W]])
        return index_img, int_img

    elif backend == "pyplot":
        # turn into numpy arrays
        #mask_numpy = mask_torch.real.cpu().numpy()
        index_numpy = index_torch.real.cpu().numpy()
        intensity_numpy = intensity_torch.real.cpu().numpy()

        # build extent from x,z if provided
        if x is not None and z is not None:
            extent = [z[0].item()*1e3, z[-1].item()*1e3,
                      x[0].item()*1e3, x[-1].item()*1e3]
        else:
            extent = None

        # plot panels
        if panel == "Index + Intensity":
            fig, axes = plt.subplots(1, 2, figsize=(15,5))
            # axes[0].imshow(mask_numpy, extent=extent, origin="lower", aspect="auto", cmap="bone")
            # axes[0].set_title(f"{title_prefix}Lens Mask")

            im1 = axes[0].imshow(index_numpy, extent=extent, origin="lower", aspect="auto", cmap="viridis")
            axes[0].set_title(f"{title_prefix}n (smoothed)")
            fig.colorbar(im1, ax=axes[0], label="Refractive index")

            im2 = axes[1].imshow(intensity_numpy, extent=extent, origin="lower", aspect="auto", cmap="inferno")
            axes[1].set_title(intensity_label)
            fig.colorbar(im2, ax=axes[1], label=intensity_label)

            plt.tight_layout()
            return fig

        elif panel == "Intensity":
            fig, ax = plt.subplots(1,1,figsize=(12,8),dpi=80)
            im = ax.imshow(intensity_numpy, extent=extent, origin="lower", aspect="auto", cmap="inferno")
            ax.set_title(intensity_label)
            fig.colorbar(im, ax=ax, label=intensity_label)
            plt.tight_layout()
            return fig

        else:
            raise ValueError(f"Unknown panel: {panel!r}")

    else:
        raise ValueError(f"Unknown backend: {backend!r}")

