# Levi Brunelle
# July 2025
#
# To run this UI, type `streamlit run gui.py` in the terminal.
# To terminate the gui, go to the terminal and press ctrl C, simply closing the window will not terminate the process


import streamlit as st
import streamlit.components.v1 as components
import torch
import mpld3
import uuid

import pandas as pd
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from lens import Lens
from simulation import OpticalSimulation
from utils.analysis import (
    calc_focal_lengths, 
    calc_real_working_distance,
    format_distance,
    compute_diffraction_limit_psf,
    compute_diffraction_limit_mtf,
    plot_psf,
    plot_mtf,
    calculate_fwhm,
)
from utils.streamlit_helpers import (
    labeled_number_with_unit,
    toggle_with_number_input,
    check_flat,
    pixels_dialog,
    warn_duplicate_z0,
    validate_distance,
)


# ===== hide +/- boxes on inputs via CSS =====
st.markdown(
    """
        <style>
        input[type=number] {
            -moz-appearance: textfield;
        }
        input[type=number]::-webkit-outer-spin-button,
        input[type=number]::-webkit-inner-spin-button {
            -webkit-appearance: none;
            margin: 0;
        }
        /* Hide the Streamlit +/- buttons on number_input */
        .stNumberInput button {
            display: none !important;
        }
        </style>
    """,
    unsafe_allow_html=True,
)


# ===== Streamlit app setup =====
st.set_page_config(layout="wide", page_title="F.L.A.S.H.")


# ===== Sidebar: Simulation Parameter Input =====
sidebar = st.sidebar
with sidebar:
    sidebar.header(r"$\large\textbf{\textsf{Simulation\ Parameters}}$")

    # --- system parameters ---
    system_exp = sidebar.expander(label=r"$\large\textbf{\textsf{System}}$", expanded=True)
    with system_exp:
        # Wavelength + unit
        wavelength = labeled_number_with_unit(
            label="Wavelength", default_val=10.6, unit_options=["nm", "µm"], unit_default="µm", num_key="wl_val", unit_key="wl_unit", ratio=[2, 1], container=system_exp
        )

        # off-axis angle
        field_angle = system_exp.number_input("Field_angle (deg)", value=0.000, format="%.3f", key="field_angle")

        # Background index
        n_background = system_exp.number_input("Background index", value=1.0, format="%.3f", key="n_bg")


    # --- grid paramaters ---
    sim_grid_exp = sidebar.expander(label=r"$\large\textbf{\textsf{Simulation\ Grid}}$", expanded=True)
    with sim_grid_exp:

        # Headroom + unit
        margin_left = labeled_number_with_unit(
            label="Left Margin", default_val=2.0, unit_options=["mm", "µm"], unit_default="mm", num_key="margin_left_val", unit_key="margin_left_unit", ratio=[2, 1], container=sim_grid_exp
        )

        # Tailroom + unit
        margin_right = labeled_number_with_unit(
            label="Right Margin", default_val=2.0, unit_options=["mm", "µm"], unit_default="mm", num_key="margin_right_val", unit_key="margin_right_unit", ratio=[2, 1], container=sim_grid_exp
        )

        # Resolution + unit
        resolution = labeled_number_with_unit(
            label="Resolution", default_val=1.0, unit_options=["nm", "µm"], unit_default="µm", num_key="res_val", unit_key="res_unit", ratio=[2, 1], container=sim_grid_exp, help="recommended λ/10"
        )

        # Zero-padding option
        padding_enabled, padding_scale = toggle_with_number_input(
            toggle_label="Enable zero padding",
            toggle_key="pad_toggle",
            input_label="Padding scale",
            input_key="pad_scale_input",
            default=1.25,
            val_default=1.0,
            active=False,
            min_value=1.0,
            max_value=3.0,
            ratio=[2,1],
            container=sim_grid_exp,
            help=("Zero-padding enlarges the FFT grid to push index discontinuities out of the region you display, reducing wrap-around artifacts. It does _not_ "
            "reapply or alter the Gaussian smoothing of the index field, although it will result in a smoothed psf, regardless of whether you have applied the smoothing filter or not."),
        )


    # --- add component buttons ---
    component_exp = sidebar.expander(label=r"$\large\textbf{\textsf{Components}}$", expanded=True)
    with component_exp:

        left_col, right_col = component_exp.columns([1,1], gap="small")

        # add lens button
        if "lenses" not in st.session_state:
            st.session_state.lenses = []
        if left_col.button("➕ Add Lens", key="add_lens"):
            st.session_state.lenses.append({"id": uuid.uuid4().hex})


        # add prism button (dummy)
        right_col.button("➕ Add Prism")

        # add phase mask button (dummy)
        left_col.button("➕ Add Phase Mask")

        # add meta-optic button (dummy)
        right_col.button("➕ Add Meta-Optic")


    # --- propagation parameters ---
    propagation_exp = sidebar.expander(label=r"$\large\textbf{\textsf{Propagation}}$", expanded=True)
    with propagation_exp:

        # Propagation method
        method = propagation_exp.selectbox("Propagation Method", ["BPM", "WPM"], key="method", help="BPM relies on paraxial approximation, can be inaccurate for large angles.")

        # Smooth refractive index toggle + smoothing px
        smooth_n, smooth_px = toggle_with_number_input(
            toggle_label="Enable Index Smoothing",
            toggle_key="smooth_n",
            input_label="# pixels",
            input_key="smooth_px",
            default=20,
            val_default=20,
            active=True,
            min_value=0,
            ratio=[2,1],
            format=None,
            container=propagation_exp,
            help=("Applies a gaussian smoothing filter across index transitions")
        )

        # Device setting
        hardware = propagation_exp.selectbox("Hardware", ["GPU", "CPU"], key="hardware")
        if hardware == "GPU":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
                propagation_exp.markdown("GPU device not detected. Using CPU instead.")
        else:
            device = "cpu"

    # --- result display paramaters ---
    display_exp = sidebar.expander(label=r"$\large\textbf{\textsf{Display}}$", expanded=True)
    with display_exp:

        # Image result display options
        display_engine = display_exp.selectbox(
            "Engine",
            ["OpenCV", "Matplotlib", "Both"],
            index=0,
            key="display_engine",
        )

        if display_engine in ("Matplotlib", "Both"):
            panel = display_exp.selectbox(
                "Which Matplotlib panel?",
                ["Index + Intensity", "Intensity"],
                index=0,
                key="panel",
            )
        else:
            panel = None
        
        # intensity scaling
        intensity_scale = display_exp.selectbox(
                "Intensity Scale", 
                ["Logarithmic", "Linear"], 
                help="Plot intensity on a log10 or linear scale. Linear will make it difficult to see anything besides the focal point."
            )
        if intensity_scale == "Logarithmic":
            use_log=True
        else:
            use_log=False


        # Image downsampling option
        downsample_toggle, max_dim = toggle_with_number_input(
            toggle_label="Downsample Images",
            toggle_key="downsample",
            input_label="Max dimension",
            input_key="max_dim",
            default=1500,
            val_default=1500,
            active=False,
            min_value=1,
            ratio=[2,1],
            format=None,
            container=display_exp,
            help=("Downsamples images with slicing to speed up render time in GUI.  \nYou can specify the maximum dimension of the final image in pixels.  \n "
                "Note: Many matplotlib image files are too large to display without downsampling "
            )
        )

        # Dark mode with variable intensity threshold
        dark_sim, intensity_threshold = toggle_with_number_input(
            toggle_label="Enable Dark Mode",
            toggle_key="dark_mode",
            input_label="Intensity Threshold",
            input_key="threshold",
            default=0.001,
            val_default=0.001,
            active=False,
            min_value=0.0,
            ratio=[2,1],
            format="%.5f",
            container=display_exp,
            help=("Cuts off intensity values below threshold to hide background noise")
        )


        lens_metrics_exp = display_exp.expander(label="Lens Metrics", expanded=True)
        with lens_metrics_exp:

            # Show ideal vs simulated values in table 
            show_focal_table = lens_metrics_exp.toggle("Print Focal Table", help="Prints a table to compare ideal vs simulated values of NA and working distance / focal length.")

            # PSF config inputs
            show_psf = lens_metrics_exp.toggle("PSF", key="show_psf")
            if show_psf:
                psf_opts = lens_metrics_exp.expander("Options", expanded=False)
                with psf_opts:
                    sim_psf      = psf_opts.checkbox("Simulation", value=True, key="show_sim_psf")
                    diff_limit   = psf_opts.checkbox("Diffraction limit", value=True, key="show_dl_psf")
                    use_zemax_psf= psf_opts.checkbox("Zemax", value=False, key="show_zemax_psf")
                    if use_zemax_psf:
                        zemax_psf_file = psf_opts.file_uploader(
                            "Upload Zemax PSF CSV", type="csv", key="zemax_psf_csv"
                        )
                    psf_xmin = psf_opts.number_input(
                        "PSF x-min (µm)", value=-100.0, format="%.1f", key="psf_xmin"
                    )
                    psf_xmax = psf_opts.number_input(
                        "PSF x-max (µm)", value=+100.0, format="%.1f", key="psf_xmax"
                    )

            # MTF config inputs
            show_mtf = lens_metrics_exp.toggle("MTF", key="show_mtf")
            if show_mtf:
                mtf_opts = lens_metrics_exp.expander("Options", expanded=False)
                with mtf_opts:
                    sim_mtf      = mtf_opts.checkbox("Simulation", value=True, key="show_sim_mtf")
                    diff_mtf     = mtf_opts.checkbox("Diffraction-limit", value=True, key="show_dl_mtf")
                    use_zemax_mtf= mtf_opts.checkbox("Zemax", value=False, key="show_zemax_mtf")
                    if use_zemax_mtf:
                        zemax_mtf_file = mtf_opts.file_uploader(
                            "Upload Zemax MTF CSV", type="csv", key="zemax_mtf_csv"
                        )
                    mtf_fmin = mtf_opts.number_input(
                        "Freq. min (cyc/µm)", value=0.0, format="%.2f", key="mtf_fmin"
                    )
                    mtf_fmax = mtf_opts.number_input(
                        "Freq. max (cyc/µm)", value=2.0, format="%.2f", key="mtf_fmax"
                    )


# === Adding title for component expanders ===
if st.session_state.lenses:
    st.title("Lenses")
# add more options here if adding other component types (prism, meta-optic, etc...)


# ===== Lens Config Expanders =====
for i, params in enumerate(st.session_state.lenses):
    lid = params["id"]
    lens_exp = st.expander(f"Lens #{i + 1}", expanded=True) # this is the main expander for each lens object

    with lens_exp:

        # Diameter + unit
        diameter = labeled_number_with_unit(
            label="Diameter", 
            default_val=12.701, 
            unit_options=["mm", "µm"], 
            unit_default="mm", 
            num_key=f"dia_val_{lid}",
            unit_key=f"dia_unit_{lid}", 
            ratio=[4, 1], 
            container=lens_exp
        )

        # Thickness + unit
        thickness = labeled_number_with_unit(
            label="Center Thickness", 
            default_val=2.50, 
            unit_options=["mm", "µm"], 
            unit_default="mm", 
            num_key=f"thick_val_{lid}",
            unit_key=f"thick_unit_{lid}", 
            ratio=[4, 1], 
            container=lens_exp
        )

        # RoC1 + unit + side type
        RoC1 = labeled_number_with_unit(
            label="RoC1", 
            default_val=45.058, 
            unit_options=["mm", "µm"], 
            unit_default="mm", 
            num_key=f"r1_val_{lid}",
            unit_key=f"r1_unit_{lid}", 
            ratio=[4, 1], 
            container=lens_exp
        )
        side1 = lens_exp.selectbox(
            "Side 1 type",
            ["convex", "concave", "flat"],
            index=0,
            key=f"s1_{lid}",
        )
        RoC1 = check_flat(RoC1, side1)

        # RoC2 + unit + side type
        RoC2 = labeled_number_with_unit(
            label="RoC2", 
            default_val=0.00, 
            unit_options=["mm", "µm"], 
            unit_default="mm", 
            num_key=f"r2_val_{lid}",
            unit_key=f"r2_unit_{lid}", 
            ratio=[4, 1], 
            container=lens_exp
        )
        side2 = lens_exp.selectbox(
            "Side 2 type",
            ["convex", "concave", "flat"],
            index=2,
            key=f"s2_{lid}",
        )
        RoC2 = check_flat(RoC2, side2)

        # lens origin z0 + unit
        z0_val = labeled_number_with_unit(
            label="Lens z0", default_val=0.0, unit_options=["mm", "µm"], unit_default="mm", 
            num_key=f"z0_val_{lid}", unit_key=f"z0_unit_{i}", ratio=[4, 1], container=lens_exp
        )

        # Refractive index
        n_lens = lens_exp.number_input(
            "Lens index (n)",
            value=st.session_state.get("n_lens", 4.002),
            format="%.3f",
            key=f"n_lens_{lid}",
        )

       # --- aperture stop parameters ---

        #toggle for putting an aperture mask 
        use_ap = lens_exp.toggle(
            "Use aperture stop",
            key=f"use_ap_{lid}"
        )

        if use_ap:
            # Stop diameter + unit
            apd_val = labeled_number_with_unit(
                label="Aperture inner diameter", default_val=0.0, unit_options=["mm", "µm"], unit_default="mm", 
                num_key=f"ap_dia_{lid}", unit_key=f"ap_dia_unit_{lid}", ratio=[3, 1], container=lens_exp
            )

            # Stop thickness + unit
            apt_val = labeled_number_with_unit(
                label="Aperture thickness", default_val=0.0, unit_options=["mm", "µm"], unit_default="mm", 
                num_key=f"ap_thick_{lid}", unit_key=f"ap_thick_unit_{lid}", ratio=[3, 1], container=lens_exp
            )

            # Stop position + unit
            apz_val = labeled_number_with_unit(
                label="Aperture origin z0", default_val=0.0, unit_options=["mm", "µm"], unit_default="mm", 
                num_key=f"ap_z0_{lid}", unit_key=f"ap_z0_unit_{lid}", ratio=[3, 1], container=lens_exp
            )

            # Stop index
            ap_idx = lens_exp.text_input(
                "Aperture index (complex)",
                value=str("5+1j"),
                key=f"ap_index_{lid}"
            )
        else:
            pass


        st.session_state.lenses[i] = {
            "id": lid,   
            "label": f"Lens {i+1}",
            "thickness": thickness,
            "r1": RoC1,
            "r2": RoC2,
            "side1": side1,
            "side2": side2,
            "n_lens": n_lens,
            "diameter": diameter,
            "z0": z0_val,
            "gap_before": 0.0,
            "aperture": {
                "enabled": use_ap,
                "diameter": apd_val if use_ap else None,
                "thickness": apt_val if use_ap else None,
                "z0": apz_val if use_ap else None,
                "index": complex(ap_idx) if use_ap else 1.0
            }
        }

        # Remove button
        lens_exp.button("❌ Remove this lens", key=f"rm_{i}", on_click=lambda i=i: st.session_state.lenses.pop(i))

warn_duplicate_z0(st.session_state.lenses)



# === Find the clear aperture ===
lenses = st.session_state.lenses
if lenses:
    lens = lenses[0]
    ap_data = lens.get("aperture", {})
    ap_dia = ap_data.get("diameter", 0.0)
    clear_ap = min(lens["diameter"], ap_dia) if ap_data.get("enabled", False) else lens["diameter"]
else:
    pass

# === Display either ideal focal length input or WD === 
# Only continue if all lenses and their elements are properly initialized
if lenses:
    single_lens_system = (
        len(lenses) == 1 and lenses[0].get("gap_before", 0.0) == 0.0
    )

    if single_lens_system:
        try:
            lens = lenses[0]
            ideal_focals, _ = calc_focal_lengths(lens=lens, n_outside=n_background)
            prefill_ideal_fl_m = ideal_focals["EFL"]

        except Exception:
            prefill_ideal_fl_m = 0.0

        # Units first for this so the autofill scales properly
        val_focal_col, unit_focal_col  = st.columns([2, 1], gap="small")
        f_unit = unit_focal_col.selectbox("Unit", ["mm", "µm"], key="f_unit")
        display_val = prefill_ideal_fl_m * (1e3 if f_unit == "mm" else 1e6)

        ideal_focal = val_focal_col.number_input(
            "Effective Focal Length (ideal)",
            min_value=0.0,
            value=round(display_val, 6),
            format="%.3f",
        )

        ideal_fl_m = ideal_focal * (1e-3 if f_unit == "mm" else 1e-6)
        ideal_na_dist = ideal_fl_m
        lenses[0]["ideal_fl"] = ideal_fl_m


    else:
        ideal_wd = labeled_number_with_unit(
            label="Working Distance (ideal)", 
            default_val=0.0, 
            unit_options=["mm", "µm"], 
            unit_default="mm", 
            num_key="ideal_wd", 
            unit_key="ideal_wd_unit", 
            ratio=[2, 1], 
            container=st
        )
        lenses[-1]["ideal_wd"] = ideal_wd
        ideal_na_dist = ideal_wd
else:
    single_lens_system = False

# === Validate the distance inputs ===
if lenses:    
    dist_check = validate_distance(ideal_na_dist)
    if not dist_check:
        # clear out any metrics the user asked for
        show_focal_table = False
        show_psf = False
        show_mtf = False


# === Compute grid sizes ===
if st.session_state.lenses:
    grid_size_x = max(l["diameter"] for l in st.session_state.lenses)

    if single_lens_system:
        lens = st.session_state.lenses[0]
        z0 = lens["z0"]
        thickness_last = lens["thickness"]
        ideal_fl = lens.get("ideal_fl", 0.0)

        grid_size_z = (
            z0 + thickness_last + ideal_fl + margin_left + margin_right
        )

    else:
        lens = st.session_state.lenses[-1]
        z0 = lens["z0"]
        gap = lens.get("gap_before", 0.0)
        thickness_last = lens["thickness"]
        ideal_wd = lens.get("ideal_wd", 0.0)

        grid_size_z = (
            z0 + gap + thickness_last + ideal_wd + margin_left + margin_right
        )
else:
    grid_size_x = 0.0
    grid_size_z = margin_left + margin_right

if lenses:
    st.markdown(f"**Grid Size (X):** {format_distance(grid_size_x)}")
    st.markdown(f"**Grid Size (Z):** {format_distance(grid_size_z)}")


# === Make Start and Stop Buttons ===
if "running" not in st.session_state:
    st.session_state.running = False
stop = lambda: not st.session_state.running

run_col, stop_col = st.sidebar.columns([1, 1], gap="small")
if run_col.button("▶️ Run Simulation", key="run_sim", disabled=not lenses):
    st.session_state.running = True
    st.session_state.pix_handled = False

if stop_col.button("🛑 Stop Simulation", key="stop_sim"):
    st.session_state.running = False



# === Run simulation ===
try:
    if st.session_state.running:

        # Check resolution and conditonally recommend downsampling
        sim_px = int(grid_size_x / resolution) * int(grid_size_z / resolution)
        if sim_px > 300e6 and not st.session_state.pix_handled and not st.session_state.get("downsample", False):
            pixels_dialog(sim_px)
            st.stop()      # scrub the rest until they make a choice

        progress_bar = st.sidebar.progress(value=0, text='Building Simulation Field...')
        with st.spinner("Running simulation..."):
            ap_idx = (
                st.session_state.lenses[0]["aperture"]["index"]
                if st.session_state.lenses
                else (5 + 1j)
            )
            if stop():
                st.warning("Aborted before building sim.")
                st.session_state.running = False
                st.stop()


            # Instantiate the OpticalSimulation object
            sim = OpticalSimulation(
                grid_size_x,
                grid_size_z,
                margin_left,
                resolution,
                n_background,
                ap_idx,
                wavelength,
                device,
                padding_scale,
            )

            if stop():
                st.warning("Aborted before adding lenses")
                st.session_state.running = False
                st.stop()
            progress_bar.progress(value=10, text='Carving Lenses...')

            # Add all the lenses to the simulation 
            for percent, l in enumerate(st.session_state.lenses, start=1):

                ap_data = l.get("aperture", {})
                ap_idx = ap_data.get("index", 1.0)
                ap_z0  = ap_data.get("z0", 0.0)
                ap_dia = ap_data.get("diameter", 0.0)
                ap_thk = ap_data.get("thickness", 0.0)

                lens = Lens(
                    Roc1=l["r1"],
                    Roc2=l["r2"],
                    side_1_type=l["side1"],
                    side_2_type=l["side2"],
                    thickness=l["thickness"],
                    diameter=l["diameter"],
                    n_lens=l["n_lens"],
                    lens_position_z=l["z0"],
                    ap_position_z=ap_z0,
                    ap_inner_diameter=ap_dia,
                    ap_thickness=ap_thk,
                )

                sim.add_lens(lens)
                progress_bar.progress(int(20 * percent / len(st.session_state.lenses)))


            # Smoothing
            if stop():
                st.warning("Aborted before smoothing")
                st.session_state.running = False
                st.stop()
            if smooth_n:
                progress_bar.progress(value=20, text='Smoothing Index Field...')
                sim.smooth_index_field(pixels_filtering=smooth_px)

            # Adding plane wave
            if stop():
                st.warning("Aborted before wave generation")
                st.session_state.running = False
                st.stop()
            progress_bar.progress(value=30, text='Generating Wave...')
            sim.generate_wave(theta_deg=field_angle)

            # Propagate wave
            if stop():
                st.warning("Aborted before propagation")
                st.session_state.running = False
                st.stop()
            progress_bar.progress(value=40, text='Propagating...')
            sim.propagate(
                smoothed=smooth_n, 
                method=method, 
                running = lambda: st.session_state.running
            )
            progress_bar.progress(value=100, text='Done!')
            st.session_state.running = False


        st.success("Simulation complete")
        st.write("# Simulation Results")
                


        # === Show Results ===


        #OpenCV plots
        if display_engine in ("OpenCV", "Both"):
            with st.spinner("Creating image…"):
                index_img, intensity_img = sim.plot(
                    backend="cv2", 
                    fast_plotting=downsample_toggle,
                    downsample_dim=max_dim,             
                    dark_mode=dark_sim, 
                    threshold=intensity_threshold, 
                    use_log=use_log,
                    smoothed=smooth_n,
                )

    

            # Convert BGR to RGB for Streamlit display
            index_img = index_img[..., ::-1]  
            intensity_img = intensity_img[..., ::-1] 

            # Show the grouped lens + index image at full column width
            if hasattr(index_img, "cpu"):
                index_img = index_img.cpu().numpy()
            with st.spinner("Rendering image…"):
                st.image(
                    index_img,
                    caption=" Index Map (smoothed)" if smooth_n else "Index Map",
                    use_container_width=True,
                    output_format="PNG",
                )

            # Show the intensity separately so it gets full width
            if hasattr(intensity_img, "cpu"):
                intensity_img = intensity_img.cpu().numpy()
            with st.spinner("Rendering image…"):
                st.image(
                    intensity_img,
                    caption="log10(Intensity)" if use_log else "Intensity",
                    use_container_width=True, 
                    output_format="PNG",
                )

        # Matplotlib + mpld3 interactive output
        if display_engine in ("Matplotlib", "Both"):
            
            if panel == "Intensity":
                if not downsample_toggle:
                    st.info(
                        "⚠️ Matplotlib interactive intensity plots are HUGE and can take a long time to render.  \n"
                        "Downsampling is being forced to avoid hitting Streamlit's size limit."
                    )
                    downsample_toggle = True

            with st.spinner("Creating image…"):
                fig = sim.plot(
                    backend="pyplot", 
                    fast_plotting=downsample_toggle,
                    downsample_dim=max_dim,             
                    dark_mode=dark_sim, 
                    threshold=intensity_threshold, 
                    panel=panel,
                    use_log=use_log,
                    smoothed=smooth_n,
                )

            with st.spinner("Rendering image…"):
                if panel == "Index + Intensity":
                    st.pyplot(fig)
                else:
                    html = mpld3.fig_to_html(fig)
                    st.components.v1.html(
                        html, width=1200, height=800, scrolling=True
                    )

        # Display PSF if requested
        if show_psf:
            # get the 1D PSF, its FWHM and focus z
            psf_slice, fwhm_sim, _ = sim.analyze_psf()
            fwhm_sim *= 1e6
            x_um    = sim.x.cpu().numpy() * 1e6
            psf_np  = psf_slice.cpu().numpy()
            y_sim   = psf_np / psf_np.max()

            # analytic Gaussian
            y_diff, fwhm_diff = compute_diffraction_limit_psf(
                clear_ap, ideal_na_dist, wavelength, x_um
            )

            # load Zemax DF if requested
            zemax_df = pd.read_csv(zemax_psf_file) if use_zemax_psf and zemax_psf_file else None
            fwhm_zemax   = None
            if isinstance(zemax_df, pd.DataFrame):
                # compute its FWHM on its own grid
                xz, yz = zemax_df["Position"].values, zemax_df["Value"].values
                fwhm_zemax = calculate_fwhm(xz, yz)

            # now plot
            fig = plot_psf(
                x_um,
                y_sim=y_sim,    
                fwhm_sim=fwhm_sim,
                y_diff=y_diff,  
                fwhm_diff=fwhm_diff,
                zemax_df=zemax_df,
                fwhm_zemax=fwhm_zemax,
                xlim=(psf_xmin, psf_xmax),
            )
            st.pyplot(fig)

            # FWHM table
            rows = []
            if sim_psf:
                rows.append(("Simulation",        f"{fwhm_sim:.2f}"))
            if diff_limit:
                rows.append(("Diffraction limit", f"{fwhm_diff:.2f}"))
            if use_zemax_psf and fwhm_zemax is not None:
                rows.append(("Zemax",             f"{fwhm_zemax:.2f}"))
            if rows:
                df_f = pd.DataFrame(rows, columns=["Source","FWHM (µm)"])
                st.subheader("PSF FWHM")
                st.dataframe(df_f, use_container_width=True, hide_index=True)

        # MTF plot
        if show_mtf:
            sim_data   = None
            diff_data  = None
            zemax_data = None

            # simulation MTF
            if sim_mtf:
                sim_freq, sim_mtf = sim.compute_mtf()
                sim_data = (sim_freq.cpu().numpy(), sim_mtf.cpu().numpy())

            # analytic diffraction MTF
            if diff_mtf:
                # x_um we already have from PSF
                freqs_dl, mtf_dl = compute_diffraction_limit_mtf(
                                    clear_ap, ideal_na_dist, wavelength, x_um
                                )
                diff_data = (freqs_dl, mtf_dl)

            # Zemax MTF
            if use_zemax_mtf and zemax_mtf_file:
                zemax_df = pd.read_csv(zemax_mtf_file)
                zemax_data = (zemax_df["Freq"].values, zemax_df["MTF"].values)

            # plot them all together
            fig_mtf = plot_mtf(
                        sim_data,
                        diff_data,
                        zemax_data,
                        freq_max=mtf_fmax,
                    )
            st.pyplot(fig_mtf)

        # Prints info table if requested
        if show_focal_table:

            zf, _ = sim.compute_focus()
            
            # Construct the correct data table
            if single_lens_system:

                try:
                    ideal_focals, real_focals = calc_focal_lengths(zf, lenses[0], n_background)
                except ZeroDivisionError:
                    st.error(
                        "⚠️ Cannot compute focal table (Both surfaces are flat).  \n"
                        "Please give at least one curved surface or disable the focal table."
                    )
                else:
                    df = pd.DataFrame({
                        "Metric": [
                            "Effective Focal Length (EFL)",
                            "Back Focal Length (BFL)",
                            "Focus Plane (zf)",
                            "Numerical Aperture (NA)"
                        ],
                        "Simulated": [
                            format_distance(real_focals["EFL"]),
                            format_distance(real_focals["BFL"]),
                            format_distance(real_focals["zf"]),
                            f"{sim.get_na(clear_ap, real_focals["EFL"]):.3f}"
                        ],
                        "Ideal": [
                            format_distance(ideal_focals["EFL"]),
                            format_distance(ideal_focals["BFL"]),
                            format_distance(ideal_focals["zf"]),
                            f"{sim.get_na(clear_ap, ideal_focals["EFL"]):.3f}"
                        ],
                    })
                    st.subheader("Simulation Summary")
                    st.dataframe(df, use_container_width=True, hide_index=True)     

            else:
                last_lens = lenses[-1]
                sim_wd = calc_real_working_distance(zf, last_lens)
                ideal_wd = last_lens["ideal_wd"]              
                zf_ideal = ideal_wd + last_lens["z0"] + last_lens["thickness"]

                df = pd.DataFrame({
                    "Metric": [
                        "Working Distance",
                        "Focal Plane (zf)",
                        "Numerical Aperture (NA)"
                    ],
                    "Simulated": [
                        format_distance(sim_wd),
                        format_distance(zf),
                        round(sim.get_na(clear_ap, sim_wd), 3)
                    ],
                    "Ideal": [
                        format_distance(ideal_wd),
                        format_distance(zf_ideal),
                        round(sim.get_na(clear_ap, ideal_wd), 3)
                    ]
                })
                st.subheader("Simulation Summary")
                st.dataframe(df, use_container_width=True, hide_index=True)

except MemoryError:
    st.error(
        "⚠️ Device out of memory!  \n"
        "Your grid is too large. Please use a coarser resolution or smaller grid."
    )
    st.stop()

except torch.cuda.OutOfMemoryError:
    st.error(
        "⚠️ Device out of memory!  \n"
        "Your grid is too large. Please use a coarser resolution or smaller grid."
    )
    torch.cuda.empty_cache()
    st.stop()

except RuntimeError as e:
    if "out of memory" in str(e).lower():
        st.error(
            "⚠️ Device out of memory!\n"
            "Your grid is too large. Please use a coarser resolution or smaller grid.."
        )
        torch.cuda.empty_cache()
        st.stop()
    else:
        raise
