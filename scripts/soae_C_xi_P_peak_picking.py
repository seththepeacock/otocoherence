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
s_pick_PSD = 10
s_pick_C = 10
s_bases = 5
s_C_xi_PSD = 7
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
prominence_PSD = php["prominence_psd"]


        
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
rows_PSD = []
rows_C = []

pp_params_subfolders = [("scipy", "v3", "")]
# pp_params_subfolders = [("scipy", "v3_PSD", "")]
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
                hop_PSD = int(round(php["hop_psd_s"] * fs))
                win_PSD = php["win_psd"]
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
                hop_PSD = int(round(php["hop_psd_s"] * fs))
                win_PSD = php["win_psd"]
                flim = php["flim"]
            case "biorxiv": # Used for bioRxiv preprint
                # Filtering parameters
                hpf = None

                # Coherence parameters
                hop_C = 441
                win_meth_C = {"method": "rho", "rho": 1.0, "win_type": "boxcar"}
                # Pure magnitude parameters
                hop_PSD = hop_C
                win_PSD = "hann"
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
            f, C_xi_P = pc.get_autocoherence(
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
            mag = pc.get_welch(wf, fs, tau, nfft=tau, hop=hop_PSD, win=win_PSD, avg_exp=1)[1]
            # Convert to db (both 20 because never using squares of any kind)
            C_xi_P = 20 * np.log10(C_xi_P) 
            mag = 20 * np.log10(mag) 
        elif avg_meth == "power":
            f, C_xi_P = pc.get_autocoherence(
                wf,
                fs,
                xi,
                tau,
                hop=hop_C,
                nfft=tau,
                win_meth=win_meth_C,
                mode="P",
                ref_type="time",
            ) 
            psd = pc.get_welch(wf, fs, tau, nfft=tau, hop=hop_PSD, win=win_PSD, avg_exp=2, scaling="density")[1] 
            # Convert to db
            C_xi_P = 10 * np.log10(C_xi_P) #
            psd = 10 * np.log10(psd) # 10 because scaling = density (no sqrt taken)

        bin_width = f[1] - f[0]

        # Calculate C_xi_phi (no longer used)
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
            psd_freqs, C_freqs = get_human_peak_freqs_manual(wf_fn, khz=False)
            peak_idxs_PSD = np.argmin((np.abs(f[None, :] - psd_freqs[:, None])), axis=1)
            peak_idxs_C = np.argmin((np.abs(f[None, :] - C_freqs[:, None])), axis=1)
        elif pp_type == "scipy":
            
            # Conversion
            wlen = int(round(wlen_hz)/bin_width)
            peak_idxs_PSD, peak_properties_PSD = find_peaks(psd, prominence=prominence_PSD, wlen=wlen)
            peak_idxs_C, peak_properties_C = find_peaks(C_xi_P, prominence=prominence_C, wlen=wlen)
        else:
            raise ValueError(f"picking_type={pp_type} is not supported!")

        # Remove everything out of range
        fmin, fmax = flim
        keep_mask_C = (f[peak_idxs_C] >= fmin) & (f[peak_idxs_C] <= fmax)
        keep_mask_PSD = ((f[peak_idxs_PSD] >= fmin) & (f[peak_idxs_PSD] <= fmax))
        peak_idxs_C = peak_idxs_C[keep_mask_C]
        peak_idxs_PSD = peak_idxs_PSD[keep_mask_PSD]


        # Deal with C_xi thresh requirement
        peak_idxs_C_unthreshed  = peak_idxs_C
        keep_mask_C_xi_thresh = (C_xi_phi[peak_idxs_C] > C_xi_thresh)
        peak_idxs_C_above_thresh = peak_idxs_C[keep_mask_C_xi_thresh]
        peak_idxs_C_removed_C_xi_thresh = np.setdiff1d(peak_idxs_C_unthreshed, peak_idxs_C)
        if check_C_xi:
            peak_idxs_C = peak_idxs_C_above_thresh

        # Deal with manually excluded peaks
        peak_idxs_C_manually_excluded = []
        for freq in php["C_ignore"].get(wf_fn, []):
            # 1. compute which f-values are within manual_thresh of freq
            mask = np.abs(f[peak_idxs_C] - freq) <= php["manual_thresh"]

            # 2. get the actual indices into f
            matching_idxs = peak_idxs_C[mask]

            # 3. if any match, append them to the exclusion list
            if matching_idxs.size > 0:
                peak_idxs_C_manually_excluded.extend(matching_idxs.tolist())
        peak_idxs_C = np.setdiff1d(peak_idxs_C, peak_idxs_C_manually_excluded)


        # Get diffs 
        psd_not_C_idxs = np.setdiff1d(peak_idxs_PSD, peak_idxs_C)
        C_not_PSD_idxs = np.setdiff1d(peak_idxs_C, peak_idxs_PSD)

        # # Switch xmax to be the maximum one that actually exists
        # xmax = f[np.max(np.concat([peak_idxs_C_unthreshed, peak_idxs_PSD]))] + fpad

        "Start Plot"
        os.makedirs(os.path.join(dirs["pp_human"], subfolder), exist_ok=True)
        plt.close("all")
        for fig in ["lf", "hf"]:
            match fig:
                case "lf":
                    xmin, xmax = 500, 8100
                case "hf":
                    xmin, xmax = 7900, f[-1]
            
            # Shift spectra so minimum within the main range is 0 dB
            xmin_idx, xmax_idx = np.argmin(np.abs(f-xmin)), np.argmin(np.abs(f-xmax))
            C_xi_P = C_xi_P - np.min(C_xi_P[xmin_idx:xmax_idx+1])
            psd = psd - np.min(psd[xmin_idx:xmax_idx+1])
            ymin_PSD = np.min([-2, np.min(C_xi_P[xmin_idx:xmax_idx+1]-1), np.min(psd[xmin_idx:xmax_idx+1]-1)])
            ymax_PSD = np.max([75, np.max(C_xi_P[xmin_idx:xmax_idx+1]+5), np.max(psd[xmin_idx:xmax_idx+1]+5)])
            ymin_C, ymax_C = ymin_PSD, ymax_PSD

            plt.figure(fig, figsize=(11, 6))
            if avg_meth == "mag":
                C_xi_P_str = rf"$C_\xi^M$" 
                mag_str = r"AMag"
            elif avg_meth == "power":
                C_xi_P_str = rf"$C_\xi^P$"
                psd_str = r"PSD"
            else:
                raise ValueError(f"Invalid avg_meth={avg_meth}!")

            # psdnitudes
            p = 1 if plot_C_xi else 0 
            plt.subplot(2+p, 1, 1)
            plt.plot(f, psd, label=psd_str, color="k", alpha=0.5)
            # Mark picks
            plt.scatter(
                f[peak_idxs_PSD],
                psd[peak_idxs_PSD],
                color="orange",
                marker="x",
                s=s_pick_PSD,
                label=rf" > {prominence_PSD}dB in {psd_str}",
            )
            plot_sticknbase(f, psd, peak_idxs_PSD, prominence_PSD, stick_hw, stick_lw, "orange")
            if pp_type == "scipy" and plot_bases:
                bases_left, bases_right = peak_properties_PSD["left_bases"], peak_properties_PSD["right_bases"]
                plt.scatter(f[bases_left], psd[bases_left], color="green", s=s_bases)
                plt.scatter(f[bases_right], psd[bases_right], color="green", s=s_bases)

            if plot_diffs:
                # Mark ones that were in C but not psd
                plt.scatter(
                    f[C_not_PSD_idxs],
                    psd[C_not_PSD_idxs],
                    color="b",
                    marker="*",
                    s=s_pick_PSD,
                    label=rf"> {prominence_C}dB {C_xi_P_str}, not > {prominence_PSD}dB {psd_str}",
                )
                plot_sticknbase(f, psd, C_not_PSD_idxs, prominence_PSD, stick_hw, stick_lw, "b")
                
            # Set lims and labels
            plt.ylim(ymin_PSD, ymax_PSD)
            plt.xlim(xmin, xmax)
            plt.ylabel(f"{psd_str} [dB]", fontsize=12)
            plt.xlabel("Frequency [Hz]", fontsize=12)
            plt.legend()

            # C_xi_P
            plt.subplot(2+p, 1, 2)
            plt.plot(f, C_xi_P, label=rf"{C_xi_P_str}", color="k", alpha=0.5)
            # Mark Picks
            plt.scatter(
                f[peak_idxs_C],
                C_xi_P[peak_idxs_C],
                color="b",
                marker="*",
                s=s_pick_C,
                label=rf"> {prominence_C}dB in {C_xi_P_str}",
            )
            plot_sticknbase(f, C_xi_P, peak_idxs_C, prominence_C, stick_hw, stick_lw, "b")
            if pp_type == "scipy" and plot_bases:
                bases_left, bases_right = peak_properties_C["left_bases"], peak_properties_C["right_bases"]
                plt.scatter(f[bases_left], C_xi_P[bases_left], color="green", s=s_bases)
                plt.scatter(f[bases_right], C_xi_P[bases_right], color="green", s=s_bases)
            if plot_diffs:
                # Plot the diffs
                plt.scatter(
                    f[psd_not_C_idxs],
                    C_xi_P[psd_not_C_idxs],
                    color="orange",
                    marker="*",
                    s=s_pick_PSD,
                    label=rf"> {prominence_PSD}dB in {psd_str}, not > {prominence_C} in {C_xi_P_str}",
                )
                plot_sticknbase(f, C_xi_P, psd_not_C_idxs, prominence_C, stick_hw, stick_lw, "orange") # Use the C one cuz we wanna see why excluded
            plt.scatter(f[peak_idxs_C_manually_excluded], C_xi_P[peak_idxs_C_manually_excluded], color="red", s=s_pick_C)
            # Set lims and labels
            plt.ylim(ymin_C, ymax_C)
            plt.xlim(xmin, xmax)
            plt.ylabel(f"{psd_str} [dB]", fontsize=12)
            plt.xlabel("Frequency [Hz]", fontsize=12)
            plt.title(rf"{C_xi_P_str}", fontsize=12)
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
                    f[peak_idxs_PSD], C_xi_phi[peak_idxs_PSD], color="orange", marker="x", s=s_C_xi_PSD, zorder=2
                )
                plt.scatter(
                    f[peak_idxs_C_unthreshed], C_xi_phi[peak_idxs_C_unthreshed], color="blue", marker="*", s=s_C_xi_C, zorder=1
                )
                plt.scatter(
                    f[peak_idxs_C_removed_C_xi_thresh], C_xi_phi[peak_idxs_C_removed_C_xi_thresh], color="red", marker="*", s=s_C_xi_C, zorder=1
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
                fig_fp = os.path.join(dirs["pp_human"], subfolder, f"{wf_fn} [{meth_id} - {fig}].jpg")
                plt.savefig(fig_fp, dpi=500)
            if show_plots:
                plt.show()

            
        # Add to spreadsheet
        for psd_freq_idx in peak_idxs_PSD:
            rows_PSD.append({wf_fn:f[psd_freq_idx]})
        for C_freq_idx in peak_idxs_C:
            rows_C.append({wf_fn:f[C_freq_idx]})

    # Wrap up spreadsheet
    if output_spreadsheet:
        df_PSD = pd.DataFrame(rows_PSD)
        df_C = pd.DataFrame(rows_C)

        # Write to Excel with multiple sheets
        spreadsheet_fn = f"Additional Human Picked Peaks [{meth_id}].xlsx"
        ss_path = os.path.join(dirs["pp_human"], subfolder, spreadsheet_fn)
        with pd.ExcelWriter(ss_path, engine='openpyxl') as writer:
            df_PSD.to_excel(writer, index=False, sheet_name="PSD")
            df_C.to_excel(writer, index=False, sheet_name="C_xi_P")

        print(f"Saved Excel file as: {ss_path}")




# CRF


# case "biorxivflim": # Used for bioRxiv preprint except more strict flims
#     # Filtering parameters
#     hpf = None

#     # Coherence parameters
#     hop_C = 441
#     win_meth_C = {"method": "rho", "rho": 1.0, "win_type": "boxcar"}
#     # Pure magnitude parameters
#     hop_PSD = hop_C
#     win_PSD = "hann"
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
#     hop_PSD = 3072
#     win_PSD = "hann"
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
#     hop_PSD = 441
#     win_PSD = "hann"
#     fs = 44100
#     tau = 3072
#     xi = 665
#     flim = [0, 22499]
#     avg_meth="mag"


