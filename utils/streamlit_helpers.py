import streamlit as st
from collections import Counter, defaultdict


def labeled_number_with_unit(
        container,
        label: str, 
        default_val: float, 
        unit_options: list, 
        unit_default: str, 
        num_key: str, 
        unit_key: str,
        ratio: list=[2,1], 
        help: str=None, 
        convert_to_meters: bool=True
    ) -> float:
    """
    Template for a labeled number input with a unit selector drowpdown.

    Parameters
    ----------
        container: streamlit container
            The UI container to place the input elements in. 
        label: str
            The label displayed above the number input.
        default_val: float
            The default value for the number input.
        unit_options: list
            A list of unit options to choose from.
        unit_default: str
            The default unit to select from the dropdown.
        num_key: str
            The key for the number input widget.
        unit_key: str
            The key for the unit selector widget.
        ratio: list[int]
            The ratio of the two columns in which the number input and unit selector will be placed.
        help: str
            Help text to display for the number input.
        convert_to_meters: bool
            If True, converts the input value to meters based on the selected unit. 
        
    Returns
    -------
        val: float
            The value entered in the number input, converted to meters if specified.

    """


    val_col, unit_col = container.columns(ratio, gap="small")
    val = val_col.number_input(label, value=default_val, format="%.3f", key=num_key, help=help)
    unit = unit_col.selectbox("Unit", unit_options, index=unit_options.index(unit_default), key=unit_key)
    if unit=="mm":
        conv=1e-3
    elif unit=="µm":
        conv=1e-6
    elif unit=="nm":
        conv=1e-9

    if convert_to_meters:
        val = val * conv
    return val

def toggle_with_number_input(
        toggle_label: str,
        toggle_key: str,
        input_label: str,
        input_key: str,
        default: int | float = 0.0,
        val_default: int | float = 0.0,
        active: bool = False,
        min_value: int | float | None = None,
        max_value: int | float | None = None,
        format: str = "%.3f",
        ratio: list[int] = [2, 1],
        container=None,
        help: str = None,
    ) -> tuple[bool, int | float]:

    """
    Template for a toggle switch with an optional number input field.

    Parameters
    ----------
        toggle_label: str
            The label for the toggle switch.
        toggle_key: str
            The key for the toggle switch widget.
        input_label: str
            The label for the number input field.
        input_key: str
            The key for the number input widget.
        default: int | float
            The default value for the number input field.
        val_default: int | float
            The default value to return when the toggle is off.
        active: bool
            Whether the toggle switch is initially active.
        min_value: int | float | None
            The minimum value for the number input field. If None, no minimum is enforced.
        max_value: int | float | None
            The maximum value for the number input field. If None, no maximum is enforced.
        format: str
            The format string for the number input field.
        ratio: list[int]
            The ratio of the two columns in which the toggle and input will be placed.
        container: streamlit container
            The UI container to place the input elements in. If None, uses the default Streamlit container.
        help: str
            Help text to display for the number input field.

    Returns
    -------
        enabled: bool
            Whether the toggle switch is active.
        val: int | float
            The value entered in the number input field if the toggle is active, otherwise returns val_default.
    
    """



    col_toggle, col_input = container.columns(ratio, vertical_alignment="center")

    enabled = col_toggle.toggle(toggle_label, key=toggle_key, value=active, help=help)
    val = None
    if enabled:
        val = col_input.number_input(
            input_label,
            value=default,
            min_value=min_value,
            max_value=max_value,
            format=format,
            key=input_key,
        )
    else:
        # leave that column blank so nothing shifts
        col_input.write("")
        val = val_default
    return enabled, val

def check_flat(
        RoC: float=None, 
        type: str=None
    ) -> float:
    """
    Check if the radius of curvature (RoC) is flat or not.
    If the type is 'flat', returns infinity to indicate no curvature.

    Parameters
    ----------
        RoC: float
            The radius of curvature of the lens surface.
        type: str
            The type of the lens surface (e.g., 'flat', 'convex', 'concave').
    
    Returns
    -------
        float
            Returns infinity if the type is 'flat', otherwise returns the original RoC value.
    """

    if type == "flat":
        return float('inf')
    else:
        return RoC

@st.dialog("High-Res Simulation Warning")
def pixels_dialog(sim_pixels: int):
    """
    Display a warning dialog when the simulation grid is large enough to slow down rendering in the GUI.


    Parameters
    ----------
    sim_pixels: int
        The number of pixels in the simulation grid.

    Returns
    -------
        None


    """
    st.warning(
        f"Your grid is {sim_pixels:,} pixels (Alot).\n"
        "Downsampling is recommended to speed up rendering times\n"
    )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Return to Config", key="pix_return"):
            st.session_state.running = False
            st.session_state.pix_handled = True
            st.rerun()
    with col2:
        if st.button("Continue", key="pix_continue"):
            st.session_state.pix_handled = True
            st.rerun()

def warn_duplicate_z0(lenses: list[dict]) -> None:
    """
    Scan a list of lens-dicts and, if any z0 value is used more than once, emit a red st.error banner.

    Parameters
    ----------
        lenses: list[dict]
            A list of lens dictionaries, each containing a 'z0' key representing the z-coordinate
            of the lens front surface origin.   
        
    Returns
    -------
        None
    """
    # Pull out all the z0 values
    z0_list = [l.get("z0", None) for l in lenses]

    # Count how often each appears
    counts = Counter(z0_list)

    # Find the ones that repeat and aren’t None
    duplicates = {z for z,c in counts.items() if c > 1 and z is not None}
    if not duplicates:
        return

    # Group by z0 [lens indices]
    groups: dict[float, list[int]] = defaultdict(list)
    for idx, z in enumerate(z0_list, start=1):
        if z in duplicates:
            groups[z].append(idx)

    # Emit an error banner per group
    for z, idxs in groups.items():
        st.error(
            f"⚠️ Lenses {', '.join(f'#{i}' for i in idxs)} all have z0 = {z}; "
            "They will overlap in the simulation!"
        )

def validate_distance(dist: float) -> bool:
    """
    Ensure *one* of WD or focal_length is > 0.0.
    If neither is, emits a red error banner and returns False.

    Parameters
    ----------
        dist: float
            The distance to validate, typically the working distance or focal length.   

    Returns
    -------
        bool
            Returns True if the distance is greater than 0.0, otherwise returns False and displays an error message.
    """
    if dist <= 0.0:
        st.error(
            "⚠️ Focal Length/Working Distance is set to 0.  \n"
            "Numerical aperture, focal table, PSF and MTF calculations all require a non-zero distance.  \n"
            "Please enter a value > 0 or proceed without using any of these metrics."
        )
        return False
    return True



