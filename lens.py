# lens.py
# Levi Brunelle
# July 2025

from utils.simtools import lens_carve 
import warnings
import torch



class Lens:

    def __init__(
            self, 
            Roc1, 
            Roc2, 
            side_1_type, 
            side_2_type, 
            thickness, 
            diameter, 
            n_lens, 
            lens_position_z=0.0, 
            ap_position_z=None, 
            ap_inner_diameter=None, 
            ap_thickness=None,
    ):      
        """
        Represents a single optical lens element defined by two curved surfaces, center thickness and dimater/aperture.

        Parameters
        ----------
            Roc1: float
                Radius of curvature for the front surface.

            Roc2: float
                Radius of curvature for the back surface.

            side_1_type: str
                Surface type for the front surface ('convex', 'concave', 'flat', etc.).

            side_2_type: str
                Surface type for the back surface.

            thickness: float
                Central thickness of the lens.

            diameter: float
                Aperture diameter of the lens.

            n_lens: float
                Refractive index of the lens material.

            lens_position_z: float
                Z-coordinate of the lens front surface origin.

            ap_position_z: float
                Z-coordinate of the aperture stop.

            ap_inner_diameter: float
                Inner diameter of the aperture.

            ap_thickness: float
                Thickness of the aperture stop.


        Methods
        -------
            carve() :
                computes boolean masks for the lens material and aperture using the lens_carve utility
        """

        self.Roc1 = Roc1
        self.Roc2 = Roc2
        self.side1 = side_1_type
        self.side2 = side_2_type
        self.thickness = thickness
        self.diameter = diameter
        self.n_lens = n_lens
        self.lens_z0 = lens_position_z
        self.ap_z0 = ap_position_z
        self.ap_inner_diameter = ap_inner_diameter
        self.ap_thickness = ap_thickness

    def carve(self,
               X: torch.Tensor, 
               Z: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate boolean masks for the lens and aperture geometry on a grid.
        Uses the lens_carve function to compute the lens shape and aperture region.

        Args
        ----
            X: torch.Tensor
                2D tensor of x-coordinate values

            Z: torch.Tensor
                2D tensor of z-coordinate values

        Returns
        -------
            lens_mask: torch.BoolTensor
                Boolean mask where True indicates lens material

            ap_mask: torch.BoolTensor
                Boolean mask where True indicates aperture stop region.
        """

        x_min, x_max = X.min().item(), X.max().item()
        z_min, z_max = Z.min().item(), Z.max().item()

        # Check X bounds
        if self.diameter / 2 > max(abs(x_min), abs(x_max)):
            warnings.warn("Lens diameter exceeds simulation X bounds.")

        # Check Z bounds
        z_front = self.ap_z0 if self.ap_z0 is not None else 0
        z_back = z_front + self.thickness
        if z_front < z_min or z_back > z_max:
            warnings.warn("Lens depth exceeds simulation Z bounds.")

        return lens_carve(
            self.Roc1, 
            self.Roc2, 
            self.side1,
            self.side2,
            self.thickness,
            X, Z, 
            self.diameter,
            lens_z0=self.lens_z0,
            ap_inner_diameter=self.ap_inner_diameter,
            ap_thickness=self.ap_thickness,
            ap_z0=self.ap_z0,
        )


