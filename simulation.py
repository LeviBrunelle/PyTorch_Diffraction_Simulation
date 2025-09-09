# simulation.py
# Levi Brunelle
# July 2025

import torch
from lens import Lens
from utils.simtools import (
    smooth_refractive_index_conv,
    make_plane_wave,
    BPM,
    WPM,
)
from utils.analysis import (
    search_focus,
    compute_na,
    calculate_fwhm,
    plot_simulation,
    compute_mtf_from_psf,
)
from typing import Callable


class OpticalSimulation:

    def __init__(self,
                Grid_size_x: float,
                Grid_size_z: float,
                margin_left: float,
                resolution: float = 1e-6,
                n_background: float = 1.0,
                ap_index: complex = 5 + 1j,
                wavelength: float = 1e-6,
                device: str = "cpu",
                padding_scale: float = 1.0,           
        ):
        """
        Simulation object to help manage steps and intermediate data for a 2D optical lens simulation.

        Builds a refractive-index field, carves in lenses, smooths interfaces,
        generates an input plane wave, propagates it slice-by-slice, and
        computes/plots the resulting PSF or field map.

        Args
        ----
            Grid_size_x : float
                Physical width of the simulation region (meters).

            Grid_size_z : float
                Physical depth of the simulation region (meters).

            margin_left : float
                Extra z-padding beyond the furthest optical element (meters).

            resolution : float, optional
                Spatial sampling interval for both x and z (meters), by default 1e-6.

            n_background : float, optional
                Background refractive index outside any lens, by default 1.0.

            ap_index : complex, optional
                Refractive index for any aperture stop (absorptive), by default 5+1j.

            wavelength : float, optional
                Illumination wavelength (meters), by default 1e-6.

            device : str, optional
                PyTorch device specifier ("cpu" or "cuda"), by default "cpu".

            padding_scale : float, optional
                Factor to enlarge the grid for edge handling, by default 1.0.

                
        Attributes
        ----------
            x : torch.Tensor
                1D x-coordinates of the simulation grid.

            z : torch.Tensor
                1D z-coordinates of the simulation grid.

            X, Z : torch.Tensor
                2D meshgrid of x and z (for carving lenses).

            n_field : torch.Tensor
                Current refractive-index map (real).

            n_smoothed : torch.Tensor
                Smoothed refractive-index map after calling `smooth_index_field()`.

            U0 : torch.Tensor or None
                Initial complex field (set by `generate_wave()`).

            U : torch.Tensor or None
                Propagated complex field (set by `propagate()`).

                
        Methods
        -------
            add_lens(lens):
                Carve a (compound) lens into `n_field` and record its mask.

            smooth_index_field(pixels_filtering):
                Apply convolutional smoothing to the refractive-index map.

            generate_wave(theta_deg):
                Create a tilted plane wave as the input field.

            propagate(method, show_progress):
                Propagate the wave through `n_smoothed` via BPM or WPM.

            compute_focus():
                Find the focal point using maximum intensity in the simulation field.

            analyze_psf():
                Compute the PSF at focus and return its FWHM.

            get_na(diameter, distance):
                Returns the Numerical Aperture of the system.
            
            compute_mtf():
                Returns the data required to plot an mtf of the system.

            plot(...):
                Render the field and index map using CV2 or matplotlib.
        """
        
        self.device        = device
        self.wavelength    = wavelength
        self.n_background  = n_background
        self.ap_index      = ap_index
        self.alpha         = 5000

        self.core_size_x   = Grid_size_x
        self.core_size_z   = Grid_size_z
        self.margin_left      = margin_left
        self.resolution    = resolution

        self.padding_scale = padding_scale

        (self.x, 
         self.z, 
         self.X, 
         self.Z, 
         self.n_field,
         self.nx, 
         self.nz) = self.grid_setup()

        self.lenses = [] 
        self.U0     = None
        self.U      = None

    def grid_setup(self
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
        """
        Creates the simulation grids with optional zero-padding. 

        Returns
        -------
            x: torch.Tensor
                1D tensor of simulation x-coordinates [nx]

            z: torch.Tensor
                1D tensor of simualtion z-coordinates [nz]

            X: torch.Tensor
                2D meshgrid of simulation x coordinates

            Z: torch.Tensor
                2D meshgrid of simulation z coordinates

            nx: int
                number of points in x 

            nz: int
                number of points in z
        """

        dx = self.resolution
        dz = self.resolution

        # apply padding_scale immediately
        # core sizes are the sim size without padding (the result you see at the end)
        grid_x = self.core_size_x * self.padding_scale
        grid_z = self.core_size_z

        core_nx = int(self.core_size_x / dx)
        nx = int(grid_x / dx)
        nz = int(grid_z / dz)

        # how many “pad” rows on each side
        self._pad_x = (nx - core_nx) // 2

        # make 1D tensors for each axis
        x = torch.linspace(-grid_x/2, grid_x/2, steps=nx, device=self.device)
        z = -self.margin_left + torch.arange(nz, device=self.device) * dz

        # use the 1D tensors to make a meshgrid
        X, Z = torch.meshgrid(x, z, indexing="ij")

        # Create an index field for the simulation, set to background index
        n = torch.full((nx, nz), 
                       fill_value=self.n_background,
                        dtype=torch.cfloat, 
                        device=self.device)
        
        return x, z, X, Z, n, nx, nz
    
    def add_lens(self, 
                 lens: Lens,
        ) -> None:

        """
        Adds a lens object to the simulation. 

        Args
        ----
            lens: Lens
                instance of a Lens class 

        """
        lens_mask, ap_mask = lens.carve(self.X, self.Z)
        self.n_field[lens_mask] = lens.n_lens
        self.n_field[ap_mask] = self.ap_index
        self.lenses.append((lens, lens_mask, ap_mask))

    def smooth_index_field(self, 
            pixels_filtering: int=20
        ) -> None:
        """
        Smooths index transitions in z for the simulation field.

        Args
        ----
            pixels_filtering: int
                width of the gaussian blur in pixels

        """

        self.n_smoothed = smooth_refractive_index_conv(self.n_field, pixels_filtering)

    def generate_wave(self, 
            theta_deg: float=0.0
        ) -> None:
        
        """
        Creates a plane wave at the start of the simulation field.

        Args
        ----
            theta_deg: float
                angle of the wave direction from the central axis, defaults to 0
        """
        self.U0 = make_plane_wave(
            self.x,
            self.wavelength,
            theta_deg=theta_deg,
            device=self.device,
        )

    def propagate(self, 
            method: str="BPM", 
            smoothed: bool=False,
            show_progress: bool=True,
            running: Callable[[], bool] = lambda: True,
        ) -> None:
        
        """         
        Propagates the input wave through the simulation domain using BPM or WPM

        Args
        ----
            method: str
                The propagation method to use, either BPM or WPM

            smoothed: bool
                Toggle for index smoothing, used to determine whichindex field is used for propagation

            show_progress: bool
                If True, displays a progress bar during propagation.
        """

        if self.U0 is None:
            raise RuntimeError("Call generate_wave() first.")

        if smoothed:
            n_to_use=self.n_smoothed
        else:
            n_to_use=self.n_field

        if method == "BPM":
            self.U = BPM(
                self.U0,
                n_to_use,
                self.x,
                self.z,
                self.wavelength,
                pad_x=self._pad_x,
                alpha=self.alpha,
                show_progress=show_progress,
                running = running,
            )
        elif method == "WPM":
            self.U = WPM(
                self.U0,
                n_to_use,
                self.x,
                self.z,
                self.wavelength,
                self.n_background,
                pad_x=self._pad_x,
                alpha=self.alpha,
                show_progress=show_progress,
                running = running,
            )
        else:
            raise ValueError(f"Unknown propagation method: {method}")

    def compute_focus(self) -> tuple[float, float]:
        """
        Finds the focal z.

        Returns
        -------
            z_peak: float
                focal point z-coordinate in microns

            z_index: float
                nearest z index to the focal point
        """
        z_peak, z_index = search_focus(self.U, self.z)
        return z_peak, z_index

    def get_na(self,
            diameter: float=0,
            distance: float=0, 
        ) -> float:
        """
        Calls the compute_na() function and returns the numerical aperture of the simulation

        Args
        ----
            diameter: float 
                diameter of the clear aperture 
            
            distance: float
                either focal length or working distance
        """
        return compute_na(diameter, distance)

    def analyze_psf(self) -> tuple[torch.Tensor, float, float]:
        """
        Obtains the psf data and fwhm from the z-slice at the focal point.
        
        Returns
        -------
            psf: torch.Tensor 
                slice of pixels from the focal point

            fwhm: float
                calcualted fwhm of the central psf peak

            z_exit: float 
                focal point z-coordinate
        """

        z_peak, z_index = search_focus(self.U, self.z)
        z_exit = self.z[z_index].item()

        psf = torch.abs(self.U[:, z_index]) ** 2
        fwhm = calculate_fwhm(self.x.cpu().numpy(), psf.cpu().numpy())


        return psf, fwhm, z_exit

    def compute_mtf(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Gets the frequency and MTF data points used for plotting the system MTF.

        Returns
        -------
            freqs : torch.Tensor
                1D tensor of spatial frequency samples (e.g. cycles/mm).

            mtf : torch.Tensor
                1D tensor of corresponding MTF values (unitless modulation depth).
        """ 
        psf_slice, _, _ = self.analyze_psf()
        return compute_mtf_from_psf(psf_slice, self.x)
    
    def plot(self,
            backend: str = "cv2",
            fast_plotting: bool = False,
            downsample_dim: int=1500,
            dark_mode: bool = True,
            threshold: float = 1e-3,
            panel: str = "all",
            use_log: bool = True,
            smoothed: bool = True,
            title_prefix: str = "",
        ):
        """
        Makes simulation plots in either openCV or Matplotlib. 
        Plots include the simulation lens mask, index field, and intensity field.

        Args:
            backend (str): either "cv2" or "pyplot"
            fast_plotting (bool): toggle for downsampling images
            downsample_dim (int): maximum dimension for downsampling images
            dark_mode (bool): toggle for using dark mode (sets any pixels below threshold intensity to black)
            threshold (float): threhsold value for dark mode
            panel (str): selects between viewing all pyplot images or only the intensity image (good for zooming in)
            use_log (bool): activates log10 intenstiy scale instead of linear. Much easier to view intensity field.
            smoothed (bool): toggle fro whether or not to use the smoothed index
            title_prefix (str): Adds a string of choice to the plot titles

        Returns:
            img (openCV imshow figure): combined panel of lens mask and index field

            intensity_img (openCV imshow figure): image of simulation intensity field

            fig (pyplot figure): pyplot figure of simulation results
        """

        # —— build and crop everything for display ——
        pad = self._pad_x

        # Make sure the correct n is used for plotting
        n_base = self.n_smoothed if smoothed else self.n_field


        # now crop U, n_smoothed, lens_mask, and x
        U_disp = self.U
        n_disp = n_base
        x_disp = self.x
        if pad:
            U_disp = U_disp[pad:-pad, :]
            n_disp = n_disp[pad:-pad, :]
            #lens_mask = lens_mask[pad:-pad, :]
            x_disp = x_disp[pad:-pad]
        # everything downstream uses *_disp versions

        # --- CV2 branch ---
        if backend=="cv2":
            return plot_simulation(
                n=n_disp,
                U=U_disp,
                x=x_disp,
                z=self.z,
                backend=backend,
                fast_plotting=fast_plotting,
                downsample_dim=downsample_dim,
                dark_mode=dark_mode,
                threshold=threshold,
                use_log=use_log,
                panel=panel,
                title_prefix=title_prefix,
            )

        # --- Pyplot branch ---
        elif backend=="pyplot":
            return plot_simulation(
                n=n_disp,
                U=U_disp,
                x=x_disp,
                z=self.z,
                backend=backend,
                fast_plotting=fast_plotting,
                downsample_dim=downsample_dim,
                dark_mode=dark_mode,
                threshold=threshold,
                use_log=use_log,
                panel=panel,
                title_prefix=title_prefix,
            )

