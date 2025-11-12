import phaseco as pc
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from helper_funcs import *
import scipy.signal as signal
import pandas as pd

# Output params
show_plots = 1
output_plots = 1
output_spreadsheet = 1

# Plotting params
stick_lw = 1
stick_hw = 0.05
fpad = 0.1
ypad = 3
s_pick_mag = 10
s_pick_C = 10
thresh_db = 2


# Choose parameter set
method_type = "Unfiltered"
# Global parameters
fs = 44100
tau = 3072
# Method-specific parameters
match method_type:
    case "Chris_OG":
        # Filtering parameters
        hpf = "butter"
        hpf_cf = 150

        # Coherence parameters
        xi = 665
        hop_C = 665
        win_meth = {"method": "rho", "rho": 1.0, "win_type": "boxcar"}
        # Pure magnitude parameters
        hop_mags = 3072
        win_mags = "hann"

    case "Filtered":
        # Filtering parameters
        hpf = "kaiser"
        hpf_cf = 150

        # Coherence parameters
        xi = 665
        hop_C = 441
        win_meth = {"method": "rho", "rho": 1.0, "win_type": "boxcar"}
        # Pure magnitude parameters
        hop_mags = 441
        win_mags = "hann"

    case "Unfiltered":
        # Filtering parameters
        hpf = None

        # Coherence parameters
        xi = 665
        hop_C = 441
        win_meth = {"method": "rho", "rho": 1.0, "win_type": "boxcar"}
        # Pure magnitude parameters
        hop_mags = 441
        win_mags = "hann"

# Filenames
wf_fns = [
    "human_TH14RearwaveformSOAEshort",
    "human_RRrearSOAEwf1short",
    "human_TH13RearwaveformSOAEshort",
    "human_KClearSOAEwf2",
    "human_AP7RearwaveformSOAEshort",
    "human_coNW_fgF090728R",
    "human_TH21RearwaveformSOAE",
    "human_AVGrearSOAEwf2",
    "human_FMlearSOAEwfA01",
    "human_JBrearSOAEwf2short",
    "human_LSrearSOAEwf1short",
    "human_JIrearSOAEwf2short",
]

# Filepaths
results_folder = os.path.join("results", "soae", "Human Peak Picks (Fig.4)")
data_fp = os.path.join("data", "additional_humans")
results_folder_C_xi_phi = os.path.join(results_folder, "C_xi_phi Plots")
os.makedirs(results_folder, exist_ok=True)
os.makedirs(results_folder_C_xi_phi, exist_ok=True)

# Define plotting helper to plot stick and base on each pick
def plot_sticknbase(f, y, idxs, thresh_db, stick_hw, stick_lw, color):
    stick_f = f[idxs]
    stick_min = y[idxs] - thresh_db
    stick_max = y[idxs]
    plt.vlines(
        stick_f,
        stick_min,
        stick_max,
        color=color,
        lw=stick_lw,
    )
    plt.hlines(
        stick_min,
        stick_f - stick_hw,
        stick_f + stick_hw,
        color=color,
        lw=stick_lw,
    )

# Initialize Spreadsheet
rows_mags = []
rows_C = []


"Start Analysis Loop"
for wf_fn in wf_fns:
    print(f"Processing {wf_fn}")
    # Get waveform and title
    if wf_fn == "human_coNW_fgF090728R":
        wf = sio.loadmat(os.path.join(data_fp, wf_fn))["wf"][0, :]
    else:
        wf = sio.loadmat(os.path.join(data_fp, wf_fn))["wf"][:, 0]
    if hpf == "butter":
        sos = signal.butter(6, hpf_cf, "hp", fs=fs, output="sos")
        wf = signal.sosfilt(sos, wf)
    elif hpf == "kaiser":
        wf = filter_wf(
            wf,
            fs,
            {
                "type": "kaiser",
                "cf": hpf_cf,
                "df": 50,
                "rip": 100,
            },
        )
        # cf=cutoff freq, df=transition band width, rip=max allowed ripple (in dB)
    title = (
        wf_fn
        + ": "
        + r"$ \tau=$"
        + f"{1000*tau/fs:.2f}"
        + " & "
        + r"$ \xi=$"
        + f"{1000*xi/fs:.2f}"
        + " ms"
    )
    f, C_xi_M = pc.get_autocoherence(
        wf,
        fs,
        xi,
        tau,
        hop=hop_C,
        nfft=tau,
        win_meth=win_meth,
        mode="M",
        ref_type="time",
    )
    mags = pc.get_welch(wf, fs, tau, nfft=tau, hop=hop_mags, win=win_mags, avg_exp=1)[1]

    # Convert to dB and kHz
    C_xi_M, mags = 20 * np.log10(np.array([C_xi_M, mags]))
    f = f / 1000

    # Get frequencies
    mag_freqs, C_freqs = get_human_peak_freqs(wf_fn)
    mag_freq_idxs = np.argmin((np.abs(f[None, :] - mag_freqs[:, None])), axis=1)
    C_freq_idxs = np.argmin((np.abs(f[None, :] - C_freqs[:, None])), axis=1)

    # Get max and min freqs
    both_freqs_idxs = np.concat((mag_freq_idxs, C_freq_idxs))
    fmin_idx, fmax_idx = np.min(both_freqs_idxs), np.max(both_freqs_idxs)
    fmin, fmax = f[fmin_idx], f[fmax_idx]
    xmin, xmax = fmin - fpad, fmax + fpad

    # Get max and min in db
    xmin_idx = np.argmin(np.abs(f - xmin))
    xmax_idx = np.argmin(np.abs(f - xmax))

    ymin_mag, ymax_mag = (
        np.min(mags[fmin_idx:fmax_idx]) - ypad,
        np.max(mags[fmin_idx:fmax_idx]) + ypad,
    )
    ymin_C, ymax_C = (
        np.min(C_xi_M[fmin_idx:fmax_idx]) - ypad,
        np.max(C_xi_M[fmin_idx:fmax_idx]) + ypad,
    )

    "Start Plot"
    plt.close("all")
    plt.figure(figsize=(10, 6))
    # Magnitudes
    plt.subplot(2, 1, 1)
    # Mark picks
    plt.scatter(
        f[mag_freq_idxs],
        mags[mag_freq_idxs],
        color="r",
        marker="x",
        s=s_pick_mag,
        label=r" > 2dB in Amags",
    )
    # Plot stick and base
    plot_sticknbase(f, mags, mag_freq_idxs, thresh_db, stick_hw, stick_lw, "r")

    # Mark C_xi_M picks that didn't show up in mags
    C_not_mags_idxs = np.setdiff1d(C_freq_idxs, mag_freq_idxs)
    plt.scatter(
        f[C_not_mags_idxs],
        mags[C_not_mags_idxs],
        color="b",
        marker="*",
        s=s_pick_mag,
        label=r"> 2dB in $C_\xi^M$, not Amags",
    )
    # Plot stick and base
    plot_sticknbase(f, mags, C_not_mags_idxs, thresh_db, stick_hw, stick_lw, "b")
    
    # Plot spectra
    plt.plot(f, mags, label="Avg. Magnitude", color="k", alpha=0.5)
    # Set lims and labels
    plt.ylim(ymin_mag, ymax_mag)
    plt.xlim(xmin, xmax)
    plt.ylabel("Magnitude [dB]", fontsize=12)
    plt.xlabel("Frequency [kHz]", fontsize=12)
    plt.legend()
    plt.title("Averaged Magnitude", fontsize=12)

    # C_xi_M
    plt.subplot(2, 1, 2)
    # Mark Picks
    plt.scatter(
        f[C_freq_idxs],
        C_xi_M[C_freq_idxs],
        color="b",
        marker="*",
        s=s_pick_C,
        label=r"> 2dB Peaks",
    )
    # Plot stick and base
    plot_sticknbase(f, C_xi_M, C_freq_idxs, thresh_db, stick_hw, stick_lw, "b")
    # Mark mag picks that didn't show up in C_xi_M
    mags_not_C_idxs = np.setdiff1d(mag_freq_idxs, C_freq_idxs)
    plt.scatter(
        f[mags_not_C_idxs],
        C_xi_M[mags_not_C_idxs],
        color="r",
        marker="*",
        s=s_pick_mag,
        label=r"> 2dB in Amags, not $C_\xi^M$",
    )
    # Plot stick and base
    plot_sticknbase(f, C_xi_M, mags_not_C_idxs, thresh_db, stick_hw, stick_lw, "r")

    # Plot spectra
    plt.plot(f, C_xi_M, label=r"$C_\xi^M$", color="k", alpha=0.5)
    # Set lims and labels
    plt.ylim(ymin_C, ymax_C)
    plt.xlim(xmin, xmax)
    plt.ylabel("Magnitude [dB]", fontsize=12)
    plt.xlabel("Frequency [kHz]", fontsize=12)
    plt.title(r"$C_\xi^M$", fontsize=12)
    plt.legend()
    # Wrap it up
    plt.suptitle(title, fontsize=8, color=[0.5, 0.5, 0.5])
    plt.tight_layout()
    if output_plots:
        fig_fp = os.path.join(results_folder, f"{wf_fn} [{method_type}].jpg")
        plt.savefig(fig_fp, dpi=500)
    if show_plots:
        plt.show()

    "Make C_xi plot"
    C_xi_phi = pc.get_autocoherence(
        wf,
        fs,
        xi,
        tau,
        hop=hop_C,
        nfft=tau,
        win_meth=win_meth,
        mode="phi",
        ref_type="time",
    )[1]
    plt.close("all")
    plt.figure()
    # Plot spectra
    plt.plot(f, C_xi_phi, label=r"$C_\xi^M$", color="r", alpha=0.4, lw=2.2)
    plt.plot(f, 0.2 * np.ones(len(f)), color="k")
    # Mark Picks
    plt.scatter(
        f[mag_freq_idxs], C_xi_phi[mag_freq_idxs], color="r", marker="x", s=s_pick_mag
    )
    plt.scatter(
        f[C_freq_idxs], C_xi_phi[C_freq_idxs], color="red", marker="*", s=s_pick_C
    )
    # Set lims and labels
    plt.ylim(0, 1)
    plt.xlim(xmin, xmax)
    plt.ylabel(r"$C_\xi^\phi$", fontsize=12)
    plt.xlabel("Frequency [kHz]", fontsize=12)
    plt.title(title, fontsize=8, loc="right", color=[0.5, 0.5, 0.5])
    plt.legend()
    plt.tight_layout()
    if output_plots:
        fig_fp = os.path.join(results_folder, "C_xi_phi Plots", f"{wf_fn} [{method_type}].jpg")
        plt.savefig(fig_fp, dpi=500)

    # Check we are always above the threshold
    for freq in C_freqs:
        freq_idx = np.argmin(np.abs(freq - f))
        if C_xi_phi[freq_idx] < 0.2:
            raise ValueError(
                f"Found one! {freq}Hz has C_xi^phi={C_xi_phi[freq_idx]} < 0.2"
            )
    
    # Add to spreadsheet
    for k in range(len(mag_freq_idxs)):
        rows_mags.append({wf_fn:f[mag_freq_idxs[k]]})
    for k in range(len(C_freq_idxs)):
        rows_C.append({wf_fn:f[C_freq_idxs[k]]})

# Wrap up spreadsheet
if output_spreadsheet:
    df_mags = pd.DataFrame(rows_mags)
    df_C = pd.DataFrame(rows_C)

    # Write to Excel with multiple sheets
    spreadsheet_fn = "Mags vs C_xi_M Picked Peaks.xlsx"
    ss_path = os.path.join(results_folder, spreadsheet_fn)
    with pd.ExcelWriter(ss_path, engine='openpyxl') as writer:
        df_mags.to_excel(writer, index=False, sheet_name="Mags")
        df_C.to_excel(writer, index=False, sheet_name="C_xi_M")

    print(f"Saved Excel file as: {ss_path}")


