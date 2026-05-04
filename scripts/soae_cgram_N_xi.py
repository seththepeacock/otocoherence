import os
import phaseco as pc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from collections import defaultdict
from scipy.signal import correlate, get_window, convolve, correlation_lags, hilbert
from helper_funcs import *

dirs = get_dirs()
os.chdir(dirs["oto"])

# Parameters

# ---WF---
speciess = ["Owl", "Anole", "Tokay", "Human"]
wf_idxs = range(4)
# speciess = ["Owl"]
# wf_idxs = [0]
wf_len_s = 60

# ---CGRAM---
mode = "phi"
tau_s = 0.15
hop_s = 0.01 # Defining it as a fraction of tau doesn't make sense since "effective" tau changes with xi
nfft = 2**13 # Next power of 2 for all fs
win_meth = {"method": "rho", "rho": 1.0, "win_type": "hann"}
# win_meth = {"method": "static", "win_type": "hann"}
xi_min_s = 0.001
xi_max_s = 0.05
flims = {'Anole':[1, 6], 'Human':[1, 10], 'Owl':[1, 12], 'Tokay':[1, 6]}
# flims = {'Anole':[0, 6], 'Human':[0, 10], 'Owl':[0, 12], 'Tokay':[0, 6]}
plot_xi_max_ss = {'Anole':50, 'Owl':50, 'Tokay':50, 'Human':200}
hpf_cf = 300
filter_meth = {'type':'kaiser', 'cf':hpf_cf, 'df':50, 'rip':100}

# T_xi Extraction
T_xi_type = "int"

# Significant T_xi extraction
T_xi_threshs = {"0.025":0.005, "0.05":0.01, "0.1":0.015}
fmin = 500
fmax = 15000

# Plot
dpi=300
plot=1
alpha_sig = 1
alpha_insig = 0.25
s_sig = 5
s_insig = 1
fsz = 16

for xi_max_s in [0.025]:
    rows = []
    T_xi_thresh = T_xi_threshs[str(xi_max_s)]
    cgram_id = f"{xi_max_s*1000:.0f}ms, delta_xi={xi_min_s*1000:.0f}ms, {pc.get_win_meth_str(win_meth)}, {get_filter_str(filter_meth)}, tau={tau_s*1000:.0f}ms, hop={hop_s*1000:.0f}ms, nfft={nfft}"
    T_xi_id = f"{T_xi_type} {xi_max_s*1000:.0f}ms, delta_xi={xi_min_s*1000:.0f}ms, {pc.get_win_meth_str(win_meth)}, {get_filter_str(filter_meth)}, tau={tau_s*1000:.0f}ms, hop={hop_s*1000:.0f}ms, nfft={nfft}"
    for wf_idx in wf_idxs: 
        for species in speciess:
        
            
            print(f"{species} {wf_idx}")
            # if species != "Human" and wf_idx > 0:
            #     continue
            # if species == "Human" and wf_idx in [1, 2]:
            #     continue
            # Get wf
            wf, wf_fn, fs = get_wf(species=species, wf_idx=wf_idx)
            wf = crop_wf(wf, fs, wf_len_s)
            # convert
            tau = int(round(tau_s*fs))
            hop = int(round(hop_s*fs))

            cgram = load_calc_colossogram(
                **{
                    "wf": wf,
                    "wf_idx": wf_idx,
                    "wf_fn": wf_fn,
                    "wf_len_s": wf_len_s,
                    "species": species,
                    "fs": fs,
                    "pkl_folder": dirs["pickles"],
                    "mode": mode,
                    "tau": tau,
                    "nfft": nfft, 
                    "xi_min_s": xi_min_s,
                    "xi_max_s": xi_max_s,
                    "hop": hop,
                    "filter_meth": filter_meth,
                    "win_meth": win_meth,
                    "demean": True,
                    "const_N_pd": False,
                    "scale": False,
                }
            )

            # Construct T_xi spectrum
            f = cgram["f"]
            T_xis = np.empty(len(f))
            for k in range(len(f)):
                if T_xi_type == "int":
                    T_xis[k] = get_T_xi_int(cgram["colossogram"][:, k], cgram["xis_s"])
                else:
                    raise ValueError()
                

            # Get mask for significance
            fmin_idx = np.argmin(np.abs(f-fmin))
            fmax_idx = np.argmin(np.abs(f-fmax))
            T_xis_idxs = np.arange(len(T_xis))
            sig_mask = (T_xis >= T_xi_thresh) & (T_xis_idxs > fmin_idx) & (T_xis_idxs < fmax_idx)

            if plot:
                plt.close('all')
                fmin_plot, fmax_plot = (np.array(flims[species]))*1000
                fmin_idx_plot, fmax_idx_plot = np.argmin(np.abs(f-fmin_plot)), np.argmin(np.abs(f-fmax_plot))
                f_plot = f[fmin_idx_plot:fmax_idx_plot]
                sig_mask_plot = sig_mask[fmin_idx_plot:fmax_idx_plot]
                T_xis_plot = T_xis[fmin_idx_plot:fmax_idx_plot]

                # Plot PSD 
                wf_filt = filter_wf_cgram(wf, fs, filter_meth)
                psd = pc.get_welch(wf=wf_filt, fs=fs, win="hann", tau=tau, nfft=nfft, hop=hop)[1]
                psd_plot = psd[fmin_idx_plot:fmax_idx_plot]
                psd_db_plot = 10*np.log10(psd_plot) 
                plt.ylabel(rf"PSD [dB]", color="orange", fontsize=fsz)
                plt.xlabel("Frequency [Hz]", fontsize=fsz)
                plt.plot(f_plot, psd_db_plot, color="orange")

                # Plot T_xi Spectrum
                plt.twinx()
                plt.scatter(f_plot[sig_mask_plot], T_xis_plot[sig_mask_plot], s=s_sig, alpha=alpha_sig, color="purple")
                plt.scatter(f_plot[~sig_mask_plot], T_xis_plot[~sig_mask_plot], s=s_insig, alpha=alpha_insig, color="purple")
                # plt.hlines(T_xi_thresh, 0, f[-1], color="purple")
                # plt.vlines([hpf_cf, fmin], np.min(T_xis_plot), np.max(T_xis_plot), color="red", lw=1)
                plt.ylabel(rf"$T_\xi^{{{T_xi_type}}}$ [{xi_max_s*1000:.0f}ms]", color="purple", fontsize=fsz)

                plt.title(f"{species} {wf_idx} [{wf_fn}]", fontsize=fsz)
                plt.tight_layout()
                fn_T_xi_spec = f"{species} {wf_idx} T_xi Spectrum [{T_xi_id}].jpg"
                plt.savefig(os.path.join(dirs["T_xi_specs"], fn_T_xi_spec), dpi=dpi)
                
                # Plot T_xi*PSD
                plt.close('all')
                T_xis_psd_plot = psd_plot * T_xis_plot
                T_xis_psd_plot = 10*np.log10(T_xis_psd_plot)
                plt.plot(f_plot, psd_db_plot, color="orange", alpha=0.5)
                plt.ylabel(rf"PSD [dB]", color="orange")
                plt.twinx()
                plt.plot(f_plot, T_xis_psd_plot, color="blue", alpha=0.5)
                plt.ylabel(rf"$T_\xi \cdot$PSD [dB]", color="blue")
                plt.title(f"{species} {wf_idx} [{wf_fn}]", fontsize=fsz)
                plt.tight_layout()
                fn_T_xi_psd = f"{species} {wf_idx} T_xi_PSD [{T_xi_id}].jpg"
                plt.savefig(os.path.join(dirs["T_xi_PSDs"], fn_T_xi_psd), dpi=dpi)
    
                


                # Plot Cgram
                plt.close('all')
                pc.plot_colossogram(cgram)
                plt.ylim(flims[species])
                fn_cgram = f"{species} {wf_idx} Colossogram [{cgram_id}].jpg"
                plt.savefig(os.path.join(dirs["cgrams"], fn_cgram), dpi=dpi)

            # Save data
            # T_xi_sig_idxs = np.flatnonzero(sig_mask)
            for idx in range(len(f)):
                f0 = f[idx]
                T_xi0 = T_xis[idx]
                row = {'species':species, 'wf_idx':wf_idx, 'f0':f0, 'mode':mode, 'T_xi':T_xi0, 'wf_fn':wf_fn, 'T_xi_type':T_xi_type}
                rows.append(row)
    df = pd.DataFrame(rows)
    fp_sheets = os.path.join(dirs["results"], f"soae_T_xi [cgram {T_xi_id}].xlsx")
    df.to_excel(fp_sheets, index=False)





