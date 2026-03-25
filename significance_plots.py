"""
Create significance_plots.pdf with cell-by-cell significance analysis using only the chosen reference frame (head or body).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Light blue colormap for threshold ratemaps (white -> light blue)
CMAP_LIGHT_BLUE = LinearSegmentedColormap.from_list("light_blue", ["white", "#87CEEB"], N=256)
from scipy.optimize import curve_fit
from tqdm import tqdm


def _weibull_with_baseline(x, c, a, k, lam):
    """Baseline + scaled Weibull so fit can be high at x=0 (Weibull PDF is zero at origin)."""
    x_safe = np.maximum(x.astype(float), 1e-6)
    w = (k / lam) * ((x_safe / lam) ** (k - 1)) * np.exp(-((x_safe / lam) ** k))
    return c + a * w


def _fit_weibull(x, y):
    """Fit baseline + scaled Weibull to firing rate vs distance. Returns (fit_success, y_fit) or (False, None)."""
    try:
        max_y = np.nanmax(y)
        min_y = np.nanmin(y)
        if max_y <= 0 or len(x) < 4:
            return False, None
        # Initial baseline ~ value at first bin; peak amplitude; shape/scale
        p0 = (float(y.flat[0]) if y.size else 0, max_y - min_y, 2.0, np.median(x[x > 0]) if np.any(x > 0) else 50.0)
        bounds = ([0, 0, 0.5, 1e-6], [max_y * 2, max_y * 500, 10, np.max(x) * 2])
        popt, _ = curve_fit(_weibull_with_baseline, x, y, p0=p0, bounds=bounds, maxfev=3000) # this is implemented differently in the rest of the code, might want to unify PRLP 03/20/2026
        y_fit = _weibull_with_baseline(x, *popt)
        return True, y_fit
    except Exception:
        return False, None


def create_significance_plots(
    phy_df,
    ref_frames,
    cell_types,
    MRLs_h, Mrlthresh_h, MALs_h,
    MRLs_b, Mrlthresh_b, MALs_b,
    head_ebc_plot_data, head_distance_bins,
    body_ebc_plot_data, body_distance_bins,
    ebc_plot_data_binary_head, ebc_plot_data_binary,
    max_bins_head, max_bins,
    pref_dist_head, pref_dist_body,
    MRLS_1, MRLS_2, MALS_1, MALS_2, pref_dist_1, pref_dist_2,
    half_ebc_head_1, half_ebc_head_2,
    half_ebc_body_1, half_ebc_body_2,
    shuffled_mrls_head, shuffled_mrls_body,
    full_session_MRL, Mrlthresh,
    ebc_angle_bin_size, ebc_dist_bin_size,
    output_base,
    pseudo_half_label_1="1st half",
    pseudo_half_label_2="2nd half",
):
    """
    Create significance_plots.pdf with bootstrap distribution, first/second half comparisons,
    and full-session summary for each cell, using only the chosen reference frame.
    """
    filename = output_base + f"angle_{ebc_angle_bin_size}dis_{ebc_dist_bin_size}_significance_plots.pdf"
    n_cells = len(phy_df)
    abins = np.linspace(0, 2 * np.pi, 360 // ebc_angle_bin_size)

    print("creating significance plots PDF...")
    with PdfPages(filename) as pdf:
        for i in tqdm(range(n_cells)):
            is_head = ref_frames[i] == "head"
            if is_head:
                ebc_data = head_ebc_plot_data[i]
                rbins = head_distance_bins.copy()
                ebc_binary = ebc_plot_data_binary_head[i]
                max_bins_i = max_bins_head[i]
                pref_dist = pref_dist_head[i]
                MRL, Mrlthresh_val, MAL = MRLs_h[i], Mrlthresh_h[i], MALs_h[i]
                half_1, half_2 = half_ebc_head_1[i], half_ebc_head_2[i]
                shuffled_mrls = shuffled_mrls_head[i]
                theta_offset = np.pi
            else:
                ebc_data = body_ebc_plot_data[i]
                rbins = body_distance_bins.copy()
                ebc_binary = ebc_plot_data_binary[i]
                max_bins_i = max_bins[i]
                pref_dist = pref_dist_body[i]
                MRL, Mrlthresh_val, MAL = MRLs_b[i], Mrlthresh_b[i], MALs_b[i]
                half_1, half_2 = half_ebc_body_1[i], half_ebc_body_2[i]
                shuffled_mrls = shuffled_mrls_body[i]
                theta_offset = np.pi / 2.0

            real_mrl = full_session_MRL[i]
            sig = real_mrl > Mrlthresh_val

            fig = plt.figure(figsize=(10.8, 14))
            fig.suptitle(f"Cell: {phy_df.index[i]} | {cell_types[i]} | ref: {ref_frames[i]} | significant: {sig}")
            gs = GridSpec(5, 2, figure=fig, hspace=0.58, wspace=0.35, height_ratios=[0.62, 1, 1, 1, 1])

            # --- Section 1: Bootstrap distribution ---
            ax1 = fig.add_subplot(gs[0, :])
            ax1.hist(shuffled_mrls, bins=12, color="steelblue", alpha=0.7, edgecolor="k")
            ax1.axvline(Mrlthresh_val, color="orange", linestyle="--", linewidth=2, label=f"99th: {Mrlthresh_val:.3f}")
            ax1.axvline(real_mrl, color="red", linestyle="-", linewidth=2, label=f"real MRL: {real_mrl:.3f}")
            ax1.set_xlabel("MRL")
            ax1.set_ylabel("Count")
            ax1.set_title(f"Bootstrap MRL | real: {real_mrl:.3f} | 99th: {Mrlthresh_val:.3f} | significant: {sig}")
            ax1.legend()
            ax1.spines[["top", "right"]].set_visible(False)

            # --- Section 2: Pseudo-half comparison (contiguous or interleaved blocks) ---
            def make_threshold(data):
                thresh = np.percentile(data, 75)
                return np.where(data >= thresh, 1, 0)

            def get_max_bins(data):
                idx = np.unravel_index(np.argmax(data), data.shape)
                return abins[idx[1]], rbins[idx[0]]

            A, R = np.meshgrid(abins, rbins)

            # Half 1 ratemap (style matched to summary_plots.pdf)
            ax2 = fig.add_subplot(gs[1, 0], projection="polar")
            ax2.grid(False)
            pc = ax2.pcolormesh(A, R, half_1, cmap="jet", edgecolors="none", rasterized=True)
            ax2.set_theta_direction(1)
            ax2.set_theta_offset(theta_offset)
            ax2.set_title(f"{pseudo_half_label_1} | MRL: {MRLS_1[i]:.3f} | MRA: {MALS_1[i]:.3f} | pref_dist: {pref_dist_1[i]:.1f}", fontsize=9)
            ax2.axis("off")
            ax2.set_frame_on(False)
            fig.colorbar(pc, ax=ax2)

            # Half 2 ratemap (style matched to summary_plots.pdf)
            ax3 = fig.add_subplot(gs[1, 1], projection="polar")
            ax3.grid(False)
            pc = ax3.pcolormesh(A, R, half_2, cmap="jet", edgecolors="none", rasterized=True)
            ax3.set_theta_direction(1)
            ax3.set_theta_offset(theta_offset)
            ax3.set_title(f"{pseudo_half_label_2} | MRL: {MRLS_2[i]:.3f} | MRA: {MALS_2[i]:.3f} | pref_dist: {pref_dist_2[i]:.1f}", fontsize=9)
            ax3.axis("off")
            ax3.set_frame_on(False)
            fig.colorbar(pc, ax=ax3)

            # Half 1 threshold (light blue so black line pops)
            bin1 = make_threshold(half_1)
            ang1, rad1 = get_max_bins(half_1)
            ax4 = fig.add_subplot(gs[2, 0], projection="polar")
            ax4.grid(False)
            ax4.pcolormesh(A, R, bin1, cmap=CMAP_LIGHT_BLUE, vmin=0, vmax=1, edgecolors="none", rasterized=True)
            ax4.plot([0, ang1], [0, rad1], "k-", linewidth=2)
            ax4.set_theta_direction(1)
            ax4.set_theta_offset(theta_offset)
            ax4.set_title(f"{pseudo_half_label_1} threshold", fontsize=9)
            ax4.axis("off")
            ax4.set_frame_on(False)

            # Half 2 threshold (light blue so black line pops)
            bin2 = make_threshold(half_2)
            ang2, rad2 = get_max_bins(half_2)
            ax5 = fig.add_subplot(gs[2, 1], projection="polar")
            ax5.grid(False)
            ax5.pcolormesh(A, R, bin2, cmap=CMAP_LIGHT_BLUE, vmin=0, vmax=1, edgecolors="none", rasterized=True)
            ax5.plot([0, ang2], [0, rad2], "k-", linewidth=2)
            ax5.set_theta_direction(1)
            ax5.set_theta_offset(theta_offset)
            ax5.set_title(f"{pseudo_half_label_2} threshold", fontsize=9)
            ax5.axis("off")
            ax5.set_frame_on(False)

            # --- Section 3: Summary (full session) ---
            ax6 = fig.add_subplot(gs[3, 0], projection="polar")
            ax6.grid(False)
            pc = ax6.pcolormesh(A, R, ebc_data, cmap="jet", edgecolors="none", rasterized=True)
            ax6.set_theta_direction(1)
            ax6.set_theta_offset(theta_offset)
            ax6.set_title(f"Full session | MRL: {MRL:.3f} | MRA: {MAL:.3f} | pref_dist: {pref_dist:.1f}", fontsize=9)
            ax6.axis("off")
            ax6.set_frame_on(False)
            fig.colorbar(pc, ax=ax6)

            ax7 = fig.add_subplot(gs[3, 1], projection="polar")
            ax7.grid(False)
            ax7.pcolormesh(A, R, ebc_binary, cmap=CMAP_LIGHT_BLUE, vmin=0, vmax=1, edgecolors="none", rasterized=True)
            ax7.plot([0, max_bins_i[0]], [0, max_bins_i[1]], "k-", linewidth=2)
            ax7.set_theta_direction(1)
            ax7.set_theta_offset(theta_offset)
            ax7.set_title("Full session threshold", fontsize=9)
            ax7.axis("off")
            ax7.set_frame_on(False)

            # Firing rate vs distance at preferred angle (MRA)
            pref_orient_idx = np.argmin(np.abs(abins - MAL))
            firing_rate_vector = ebc_data[:, pref_orient_idx]
            n_dist = len(firing_rate_vector)
            x_dist = rbins[:n_dist] if len(rbins) >= n_dist else np.arange(n_dist, dtype=float)

            ax8 = fig.add_subplot(gs[4, :])
            ax8.plot(x_dist, firing_rate_vector, "k-o", markersize=4, label="firing rate")
            fit_ok, y_fit = _fit_weibull(x_dist, firing_rate_vector)
            if fit_ok and y_fit is not None:
                ax8.plot(x_dist, y_fit, "r-", linewidth=2, label="Weibull fit")
                peak_idx = np.argmax(y_fit)
                pref_dist_val = x_dist[peak_idx] if peak_idx < len(x_dist) else x_dist[pref_dist]
                ax8.axvline(pref_dist_val, color="green", linestyle="--", alpha=0.8, label=f"pref_dist: {pref_dist_val:.1f}")
            else:
                idx = min(pref_dist, len(x_dist) - 1)
                ax8.axvline(x_dist[idx], color="green", linestyle="--", alpha=0.8, label=f"pref_dist: {pref_dist:.1f}")
            ax8.set_xlabel("Distance bin")
            ax8.set_ylabel("Firing rate (Hz)")
            ax8.set_title("Firing rate at preferred angle (MRA)")
            ax8.set_ylim(bottom=0)
            ax8.legend(fontsize=8)
            ax8.spines[["top", "right"]].set_visible(False)

            fig.tight_layout(rect=[0, 0, 1, 0.955])

            pdf.savefig(fig)
            plt.close(fig)

    print("done creating significance plots PDF.")