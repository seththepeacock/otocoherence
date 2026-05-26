import phaseco as pc
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from helper_funcs import *
import scipy.signal as signal
import pandas as pd

# Output params
show_plots = 0
output_plots = 1
output_spreadsheet = 1



# Plotting params
plot_C_xi = False
plot_diffs = True
plot_bases = True
stick_lw = 1
stick_hw = 25 # Hz
fpad = 100 
ypad = 3
s_pick_mag = 10
s_pick_C = 10
s_bases = 5
s_C_xi_mag = 7
s_C_xi_C = 20

# Choose parameter set
# param_set = "v3"
# pp_type = "scipy"
# subfolder = ""
check_C_xi = False
C_xi_thresh = 0.21
php = get_params_human_picking()

# SciPy Peak Picking paramters
wlen_hz = php["wlen_hz"]
prominence_C = php["prominence_C"]
prominence_mag = php["prominence_mag"]


        
# Filenames
wf_fns = get_human_fns()
dirs = get_dirs()

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
rows_mag = []
rows_C = []

pp_params_subfolders = [("scipy", "v3", ""), ("scipy", "v3_mag", "")]
# pp_params_subfolders = [("scipy", "v3_mag", "")]

"Start Analysis Loop"
wf_fns.sort(key=str.lower)
for pp_type, param_set, subfolder in pp_params_subfolders:
    for wf_fn in wf_fns:
        # Method-specific parameters
        match param_set:
            case "v3":
                avg_meth = php["avg_meth"]
                hpf = "na"
                hpf_meth = php["hpf_meth"]
                fs = php["fs"]
                tau = int(round(php["tau_s"] * fs))
                xi = int(round(php["xi_s"] * fs))
                hop_C = int(round(php["hop_C_s"] * fs))
                win_meth_C = php["win_meth_C"]
                hop_mag = int(round(php["hop_mag_s"] * fs))
                win_mag = php["win_mag"]
                flim = php["flim"]
            case "v3_mag":
                avg_meth = "mag"
                hpf = "na"
                hpf_meth = php["hpf_meth"]
                fs = php["fs"]
                tau = int(round(php["tau_s"] * fs))
                xi = int(round(php["xi_s"] * fs))
                hop_C = int(round(php["hop_C_s"] * fs))
                win_meth_C = php["win_meth_C"]
                hop_mag = int(round(php["hop_mag_s"] * fs))
                win_mag = php["win_mag"]
                flim = php["flim"]
            case "biorxiv": # Used for bioRxiv preprint
                # Filtering parameters
                hpf = None

                # Coherence parameters
                hop_C = 441
                win_meth_C = {"method": "rho", "rho": 1.0, "win_type": "boxcar"}
                # Pure magnitude parameters
                hop_mag = hop_C
                win_mag = "hann"
                fs = 44100
                tau = 3072
                xi = 665
                flim = [0, 22499]
                avg_meth="mag"

            case _:
                raise ValueError(f"param_set={param_set} hasn't been defined! (Check commented out section at end?)")

        C_xi_thresh_str = f", C_xi_thresh={C_xi_thresh}" if check_C_xi else ""
        meth_id = f"{param_set}, {pp_type}{C_xi_thresh_str}, {php["meth_id"]}"
        print(f"Processing {wf_fn}")
        # Get waveform and title
        if wf_fn == "human_coNW_fgF090728R":
            wf = sio.loadmat(os.path.join(dirs["additional_humans"], wf_fn))["wf"][0, :]
        else:
            wf = sio.loadmat(os.path.join(dirs["additional_humans"], wf_fn))["wf"][:, 0]
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
        elif param_set[0:2] == "v3" :
            # wf = crop_wf(wf, fs, wf_len_s) # Chris already cropped these; not all of them are 60s
            wf -= np.mean(wf)
            wf = filter_wf_cgram(wf, fs, hpf_meth)

        suptitle = rf"{wf_fn}: $\tau={1000*tau/fs:.2f}$ms & $\xi=${1000*xi/fs:.2f}ms [{pp_type} peak-picking, {param_set}]"
        if avg_meth == "mag":
            f, C_xi_M = pc.get_autocoherence(
                wf,
                fs,
                xi,
                tau,
                hop=hop_C,
                nfft=tau,
                win_meth=win_meth_C,
                mode="M",
                ref_type="time",
            )
            mag = pc.get_welch(wf, fs, tau, nfft=tau, hop=hop_mag, win=win_mag, avg_exp=1)[1]
        elif avg_meth == "power":
            f, C_xi_M = pc.get_autocoherence(
                wf,
                fs,
                xi,
                tau,
                hop=hop_C,
                nfft=tau,
                win_meth=win_meth_C,
                mode="P2",
                ref_type="time",
            ) # P2 (and P) take the sqrt at the end
            mag = pc.get_welch(wf, fs, tau, nfft=tau, hop=hop_mag, win=win_mag, avg_exp=2, scaling="amplitude")[1] 
            # With scaling="amplitude", this takes the sqrt at the end

        # Convert to dB and kHz
        C_xi_M, mag = 20 * np.log10(np.array([C_xi_M, mag])) # Both are 20 because you use scaling="amplitude"
        bin_width = f[1] - f[0]
        # f = f / 1000

        # Calculate C_xi_phi
        C_xi_phi = pc.get_autocoherence(
                wf,
                fs,
                xi,
                tau,
                hop=hop_C,
                nfft=tau,
                win_meth=win_meth_C,
                mode="phi",
                ref_type="time",
            )[1]
        
        # Get frequencies
        if pp_type == "manual":
            if param_set not in ["biorxiv", "biorxivflim"]:
                raise ValueError("We didn't do manual peak-picking for this param set!")
            mag_freqs, C_freqs = get_human_peak_freqs_manual(wf_fn, khz=False)
            peak_idxs_mag = np.argmin((np.abs(f[None, :] - mag_freqs[:, None])), axis=1)
            peak_idxs_C = np.argmin((np.abs(f[None, :] - C_freqs[:, None])), axis=1)
        elif pp_type == "scipy":
            
            # Conversion
            wlen = int(round(wlen_hz)/bin_width)
            peak_idxs_mag, peak_properties_mag = find_peaks(mag, prominence=prominence_mag, wlen=wlen)
            peak_idxs_C, peak_properties_C = find_peaks(C_xi_M, prominence=prominence_C, wlen=wlen)
        else:
            raise ValueError(f"picking_type={pp_type} is not supported!")

        # Remove everything out of range
        fmin, fmax = flim
        keep_mask_C = (f[peak_idxs_C] >= fmin) & (f[peak_idxs_C] <= fmax)
        keep_mask_mag = ((f[peak_idxs_mag] >= fmin) & (f[peak_idxs_mag] <= fmax))
        peak_idxs_C = peak_idxs_C[keep_mask_C]
        peak_idxs_mag = peak_idxs_mag[keep_mask_mag]

        # Shift spectra so minimum within the main range is 0 dB
        fmin_0db_ref = 500
        fmax_0db_ref = 10000
        fmin_idx_0db_ref, fmax_idx_0db_ref = np.argmin(np.abs(f-fmin_0db_ref)), np.argmin(np.abs(f-fmax_0db_ref))
        C_xi_M = C_xi_M - np.min(C_xi_M[fmin_idx_0db_ref:fmax_idx_0db_ref])
        mag = mag - np.min(mag[fmin_idx_0db_ref:fmax_idx_0db_ref])
        ymin_mag = -2
        ymax_mag = 75
        ymin_C, ymax_C = ymin_mag, ymax_mag


        # Deal with C_xi thresh requirement
        peak_idxs_C_unthreshed  = peak_idxs_C
        keep_mask_C_xi_thresh = (C_xi_phi[peak_idxs_C] > C_xi_thresh)
        peak_idxs_C_above_thresh = peak_idxs_C[keep_mask_C_xi_thresh]
        peak_idxs_C_removed = np.setdiff1d(peak_idxs_C_unthreshed, peak_idxs_C)
        if check_C_xi:
            peak_idxs_C = peak_idxs_C_above_thresh


        # Get diffs 
        mag_not_C_idxs = np.setdiff1d(peak_idxs_mag, peak_idxs_C)
        C_not_mag_idxs = np.setdiff1d(peak_idxs_C, peak_idxs_mag)

        xmin = fmin - fpad
        xmax = fmax + fpad
        # # Switch xmax to be the maximum one that actually exists
        # xmax = f[np.max(np.concat([peak_idxs_C_unthreshed, peak_idxs_mag]))] + fpad

        "Start Plot"
        os.makedirs(os.path.join(dirs["pp_human"], subfolder), exist_ok=True)
        plt.close("all")
        plt.figure(figsize=(10, 9))
        if avg_meth == "mag":
            C_xi_M_str = rf"$C_\xi^M$" 
            mag_str = r"AMag"
        elif avg_meth == "power":
            C_xi_M_str = rf"$C_\xi^P$"
            mag_str = r"PSD"
        else:
            raise ValueError(f"Invalid avg_meth={avg_meth}!")

        # Magnitudes
        p = 1 if plot_C_xi else 0 
        plt.subplot(2+p, 1, 1)
        plt.plot(f, mag, label=mag_str, color="k", alpha=0.5)
        # Mark picks
        plt.scatter(
            f[peak_idxs_mag],
            mag[peak_idxs_mag],
            color="orange",
            marker="x",
            s=s_pick_mag,
            label=rf" > {prominence_mag}dB in {mag_str}",
        )
        plot_sticknbase(f, mag, peak_idxs_mag, prominence_mag, stick_hw, stick_lw, "orange")
        if pp_type == "scipy" and plot_bases:
            bases_left, bases_right = peak_properties_mag["left_bases"], peak_properties_mag["right_bases"]
            plt.scatter(f[bases_left], mag[bases_left], color="green", s=s_bases)
            plt.scatter(f[bases_right], mag[bases_right], color="green", s=s_bases)

        if plot_diffs:
            # Mark ones that were in C but not mag
            plt.scatter(
                f[C_not_mag_idxs],
                mag[C_not_mag_idxs],
                color="b",
                marker="*",
                s=s_pick_mag,
                label=rf"> {prominence_C}dB {C_xi_M_str}, not > {prominence_mag}dB {mag_str}",
            )
            plot_sticknbase(f, mag, C_not_mag_idxs, prominence_mag, stick_hw, stick_lw, "b")
            
        # Set lims and labels
        plt.ylim(ymin_mag, ymax_mag)
        plt.xlim(xmin, xmax)
        plt.ylabel(f"{mag_str} [dB]", fontsize=12)
        plt.xlabel("Frequency [Hz]", fontsize=12)
        plt.legend()

        # C_xi_M
        plt.subplot(2+p, 1, 2)
        plt.plot(f, C_xi_M, label=rf"{C_xi_M_str}", color="k", alpha=0.5)
        # Mark Picks
        plt.scatter(
            f[peak_idxs_C],
            C_xi_M[peak_idxs_C],
            color="b",
            marker="*",
            s=s_pick_C,
            label=rf"> {prominence_C}dB in {C_xi_M_str}",
        )
        plot_sticknbase(f, C_xi_M, peak_idxs_C, prominence_C, stick_hw, stick_lw, "b")
        if pp_type == "scipy" and plot_bases:
            bases_left, bases_right = peak_properties_C["left_bases"], peak_properties_C["right_bases"]
            plt.scatter(f[bases_left], C_xi_M[bases_left], color="green", s=s_bases)
            plt.scatter(f[bases_right], C_xi_M[bases_right], color="green", s=s_bases)
        if plot_diffs:
            # Plot the diffs
            plt.scatter(
                f[mag_not_C_idxs],
                C_xi_M[mag_not_C_idxs],
                color="orange",
                marker="*",
                s=s_pick_mag,
                label=rf"> {prominence_mag}dB in {mag_str}, not > {prominence_C} in {C_xi_M_str}",
            )
            plot_sticknbase(f, C_xi_M, mag_not_C_idxs, prominence_C, stick_hw, stick_lw, "orange") # Use the C one cuz we wanna see why excluded

        # Set lims and labels
        plt.ylim(ymin_C, ymax_C)
        plt.xlim(xmin, xmax)
        plt.ylabel(f"{mag_str} [dB]", fontsize=12)
        plt.xlabel("Frequency [Hz]", fontsize=12)
        plt.title(rf"{C_xi_M_str}", fontsize=12)
        plt.legend()
        

        if plot_C_xi:
            "Make C_xi plot"
            # plt.close("all")
            # plt.figure()

            plt.subplot(3, 1, 3)
            # Plot spectra
            plt.plot(f, C_xi_phi, label=rf"$C_\xi^\phi$", color="black", alpha=0.4, lw=2.2)
            plt.plot(f, C_xi_thresh * np.ones(len(f)), color="green")
            # Mark Picks
            plt.scatter(
                f[peak_idxs_mag], C_xi_phi[peak_idxs_mag], color="orange", marker="x", s=s_C_xi_mag, zorder=2
            )
            plt.scatter(
                f[peak_idxs_C_unthreshed], C_xi_phi[peak_idxs_C_unthreshed], color="blue", marker="*", s=s_C_xi_C, zorder=1
            )
            plt.scatter(
                f[peak_idxs_C_removed], C_xi_phi[peak_idxs_C_removed], color="red", marker="*", s=s_C_xi_C, zorder=1
            )
            # Set lims and labels
            plt.ylim(0, 1)
            plt.xlim(xmin, xmax)
            plt.ylabel(rf"$C_\xi^\phi$", fontsize=12)
            plt.xlabel("Frequency [kHz]", fontsize=12)
            # plt.title(suptitle, fontsize=8, loc="right", color=[0.5, 0.5, 0.5])
            plt.legend()
            plt.tight_layout()
            # if output_plots:
            #     fig_fp = os.path.join(dirs["pp_human"], subfolder, "pp_C_xi_phi", f"{wf_fn} [{meth_id}].jpg")
            #     plt.savefig(fig_fp, dpi=500)

        # Wrap it up
        plt.suptitle(suptitle, fontsize=8, color=[0.5, 0.5, 0.5])
        plt.tight_layout()
        if output_plots:
            fig_fp = os.path.join(dirs["pp_human"], subfolder, f"{wf_fn} [{meth_id}].jpg")
            plt.savefig(fig_fp, dpi=500)
        if show_plots:
            plt.show()

            
        # Add to spreadsheet
        for mag_freq_idx in peak_idxs_mag:
            rows_mag.append({wf_fn:f[mag_freq_idx]})
        for C_freq_idx in peak_idxs_C:
            rows_C.append({wf_fn:f[C_freq_idx]})

    # Wrap up spreadsheet
    if output_spreadsheet:
        df_mag = pd.DataFrame(rows_mag)
        df_C = pd.DataFrame(rows_C)

        # Write to Excel with multiple sheets
        spreadsheet_fn = f"Additional Human Picked Peaks [{meth_id}].xlsx"
        ss_path = os.path.join(dirs["pp_human"], subfolder, spreadsheet_fn)
        with pd.ExcelWriter(ss_path, engine='openpyxl') as writer:
            df_mag.to_excel(writer, index=False, sheet_name="mag")
            df_C.to_excel(writer, index=False, sheet_name="C_xi_M")

        print(f"Saved Excel file as: {ss_path}")




# CRF
# case "biorxivflim": # Used for bioRxiv preprint except more strict flims
#     # Filtering parameters
#     hpf = None

#     # Coherence parameters
#     hop_C = 441
#     win_meth_C = {"method": "rho", "rho": 1.0, "win_type": "boxcar"}
#     # Pure magnitude parameters
#     hop_mag = hop_C
#     win_mag = "hann"
#     fs = 44100
#     tau = 3072
#     xi = 665
#     flim = php["flim"]
#     avg_meth="mag"
# case "chris_og":
#     # Filtering parameters
#     hpf = "butter"
#     hpf_cf = 150

#     # Coherence parameters
#     hop_C = 665
#     win_meth_C = {"method": "rho", "rho": 1.0, "win_type": "boxcar"}
#     # Pure magnitude parameters
#     hop_mag = 3072
#     win_mag = "hann"
#     fs = 44100
#     tau = 3072
#     xi = 665
#     flim = [0, 22499]
#     avg_meth="mag"

# case "filtered":
#     # Filtering parameters
#     hpf = "kaiser"
#     hpf_cf = 150

#     # Coherence parameters
#     hop_C = 441
#     win_meth_C = {"method": "rho", "rho": 1.0, "win_type": "boxcar"}
#     # Pure magnitude parameters
#     hop_mag = 441
#     win_mag = "hann"
#     fs = 44100
#     tau = 3072
#     xi = 665
#     flim = [0, 22499]
#     avg_meth="mag"