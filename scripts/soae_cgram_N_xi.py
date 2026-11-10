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
# speciess = ["Owl"]
# wf_idxs = [3]
wf_idxs = np.arange(4)


# ---CGRAM---
pcg = get_params_cgram()
wf_len_s = pcg['wf_len_s']
hpf_meth = pcg['hpf_meth']
mode = pcg['mode']
tau_s = pcg['tau_s']
hop_cgram_s = pcg['hop_cgram_s']
hop_psd_s = pcg['hop_psd_s']
nfft_cgram_psd = pcg['nfft']
win_meth_cgram = pcg['win_meth_cgram']
win_type_psd = pcg['win_type_psd']
xi_min_s = pcg['xi_min_s']
delta_xi_s = pcg['delta_xi_s']
T_xi_thresh = pcg["T_xi_thresh"]
cgram_id = pcg["cgram_id"]

# T_xi_meth = pcg['T_xi_meth']
T_xi_meth = "int"
# T_xi_id = pcg["T_xi_id"] # defined in loop


# Significant T_xi extraction
fmin = pcg['fmin']
fmax = np.inf

# Plot
dpi=300
plot=1
alpha_sig = 1
alpha_insig = 0.25
s_sig = 5
s_insig = 1
fsz = 16
flims = {'Anole':[1, 6], 'Human':[1, 10], 'Owl':[1, 12], 'Tokay':[1, 6]}
plot_xi_max_ss = {'Anole':50, 'Owl':50, 'Tokay':50, 'Human':200}

init_xi_max_s = pcg["init_xi_max_s"]
beta = pcg["beta"]
xi_max_s = pcg["xi_max_s"]

plot_cumsum = 1

if init_xi_max_s > xi_max_s:
    raise ValueError("init_xi_max_s can't be longer than xi_max_s!")
for T_xi_xi_max_s in [0.025]:
    rows = []
    T_xi_id = f"{T_xi_meth} {T_xi_xi_max_s*1000:.0f}ms"
    print(f"Running {T_xi_id}")
    for wf_idx in wf_idxs: 
        for species in speciess:
            if species == "Human":
                xi_max_s = 3.0
            else:
                xi_max_s = 0.5
            print(f"{species} {wf_idx}")

            # Get wf
            wf, wf_fn, fs = get_wf(species=species, wf_idx=wf_idx)
            wf_filt = filter_wf_cgram(wf, fs, hpf_meth) # Filter

            # convert
            tau_cgram_psd = int(round(tau_s*fs))
            hop_cgram = int(round(hop_cgram_s*fs))
            hop_psd = int(round(hop_psd_s*fs))

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
                    "tau": tau_cgram_psd,
                    "nfft": nfft_cgram_psd, 
                    "xi_min_s": xi_min_s,
                    "xi_max_s": xi_max_s,
                    "hop": hop_cgram,
                    "filter_meth": hpf_meth,
                    "win_meth": win_meth_cgram,
                    "demean": True,
                    "const_N_pd": False,
                    "scale": False,
                }
            )

            # Construct T_xi spectrum
            f = cgram["f"]
            T_xis = np.empty(len(f))
            # Identify which frequencies to plot
            fp_pp = os.path.join(dirs["results"], "picked_peaks.json")
            picked_peaks = get_picked_peaks(fp_pp, wf_fn)
            all_freqs = np.concatenate((picked_peaks, [500, 1000, 10000, 20000]))
            idxs = np.argmin(np.abs(f[:, None] - all_freqs[None, :]), axis=0)
            for k in range(len(f)):
                # Ignore these as we don't need them anyhow and they may need a longer cgram than we have
                if f[k] < pcg['fmin']:
                    T_xis[k] = -1
                    continue
                if T_xi_meth == "int":
                    T_xi_phi = get_T_xi_int(cgram["colossogram"][:, k], cgram["xis_s"], T_xi_xi_max_s=T_xi_xi_max_s, used_purposefully=True)
                    T_xis[k] = T_xi_phi
                elif T_xi_meth == "int_cor":
                    raise ValueError("sure you want to do that")
                    T_xi_phi, T_xi_phi_est = get_T_xi_int_cor(cgram["colossogram"][:, k], cgram["xis_s"], init_xi_max_s, beta)
                    T_xis[k] = T_xi_phi

                    if plot and (k in idxs):
                        # Convert from cgram language to ACF language
                        lags_s = cgram['xis_s']
                        acf_phi = cgram["colossogram"][:, k]

                        # Prep to plot T_xi calculation
                        xi_max_s_plot = max([T_xi_phi, init_xi_max_s, T_xi_phi_est * beta]) * 1.2
                        lags_max_plot_idx = np.argmin(np.abs(lags_s-xi_max_s_plot))
                        lags_ms_plot = lags_s[0:lags_max_plot_idx] * 1000
                        acf_phi_plot = acf_phi[0:lags_max_plot_idx]
                    
                        # Plot
                        plt.figure(k, figsize=(15, 8))
                        plt.suptitle(f"{species} {wf_idx} - {f[k]:.1f} Hz [{T_xi_id}]")
                        
                        # Plot the first round
                        plt.subplot(1, 2, 1)
                        plt.title(rf"$T_\xi^\text{{est}} =$ {T_xi_phi_est*1000:.3g} ms")
                        plt.plot(lags_ms_plot, acf_phi_plot, color='k')
                        plt.vlines(init_xi_max_s*1000, 0, 1, color='red', label=r'$\xi_\text{max}^\text{init}$')
                        idx = np.argmin(np.abs(lags_ms_plot-init_xi_max_s*1000))
                        x = lags_ms_plot[0:idx]
                        zeros = np.zeros(len(x))
                        plt.fill_between(x, zeros, acf_phi_plot[0:idx+1], color='gray', alpha=0.3)
                        plt.vlines(T_xi_phi_est*1000, 0, 1, color='purple', label=r'$T_{\xi}^\text{est}$', alpha=0.5)
                        plt.xlabel(r"$\xi$ [ms]")
                        plt.ylabel(r"$C_\xi$")
                        plt.legend(loc='upper right')

                        # Second Round
                        plt.subplot(1, 2, 2)
                        plt.title(rf"$T_\xi =$ {T_xi_phi*1000:.3g} ms ({100*(T_xi_phi-T_xi_phi_est) / T_xi_phi_est:.1f}%)")
                        plt.plot(lags_ms_plot, acf_phi_plot, color='k')
                        plt.vlines(T_xi_phi_est*1000, 0, 1, color='purple', label=r'$T_{\xi}^\text{est}$', alpha=0.5)
                        plt.vlines(T_xi_phi_est*1000*beta, 0, 1, color='red', label=r'$\xi_\text{max}=T_{\xi}^\text{est}*\beta$')
                        idx = np.argmin(np.abs(lags_ms_plot-T_xi_phi_est*1000*beta))
                        x = lags_ms_plot[0:idx]
                        zeros = np.zeros(len(x))
                        plt.fill_between(x, zeros, acf_phi_plot[0:idx+1], color='grey', alpha=0.3)
                        plt.vlines(T_xi_phi*1000, 0, 1, color='purple', label=r'$T_{\xi}$')
                        plt.xlabel(r"$\xi$ [ms]")
                        plt.ylabel(r"$C_\xi$")
                        plt.legend(loc='upper right')

                        # Save Fig
                        os.makedirs(os.path.join(dirs[f"T_xi_{T_xi_meth}"], f"Cgram {T_xi_id}"), exist_ok=True)
                        plt.savefig(os.path.join(dirs[f"T_xi_{T_xi_meth}"], f"Cgram {T_xi_id}", f"{species} {wf_idx} {f[k]:.0f} Hz - ACF [{T_xi_id}, {cgram_id}].jpg"))

                else:
                    raise ValueError("Only T_xi_meth='int_cor' implemented now to avoid errors!")
                

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
                psd = pc.get_welch(wf=wf_filt, fs=fs, win=win_type_psd, tau=tau_cgram_psd, nfft=nfft_cgram_psd, hop=hop_psd)[1]
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
                plt.ylabel(rf"$T_\xi^{{{T_xi_meth}}}$ [{init_xi_max_s*1000:.0f}ms]", color="purple", fontsize=fsz)

                plt.title(f"{species} {wf_idx} [{wf_fn}]", fontsize=fsz)
                plt.tight_layout()
                fn_T_xi_spec = f"{species} {wf_idx} T_xi Spectrum [{T_xi_id}].jpg"
                plt.savefig(os.path.join(dirs["T_xi_specs"], fn_T_xi_spec), dpi=dpi)
                
                # # Plot T_xi*PSD
                # plt.close('all')
                # T_xis_psd_plot = psd_plot * T_xis_plot
                # T_xis_psd_plot = 10*np.log10(T_xis_psd_plot)
                # plt.plot(f_plot, psd_db_plot, color="orange", alpha=0.5)
                # plt.ylabel(rf"PSD [dB]", color="orange")
                # plt.twinx()
                # plt.plot(f_plot, T_xis_psd_plot, color="blue", alpha=0.5)
                # plt.ylabel(rf"$T_\xi \cdot$PSD [dB]", color="blue")
                # plt.title(f"{species} {wf_idx} [{wf_fn}]", fontsize=fsz)
                # plt.tight_layout()
                # fn_T_xi_psd = f"{species} {wf_idx} T_xi_PSD [{T_xi_id}].jpg"
                # plt.savefig(os.path.join(dirs["T_xi_PSDs"], fn_T_xi_psd), dpi=dpi)

                # Plot Cgram
                plt.close('all')
                pc.plot_colossogram(cgram)
                xmin, _ = plt.xlim()
                xmax = init_xi_max_s * 1000
                plt.xlim(xmin, xmax)
                plt.ylim(flims[species])
                fn_cgram = f"{species} {wf_idx} Colossogram [{cgram_id}].jpg"
                plt.savefig(os.path.join(dirs["cgrams"], fn_cgram), dpi=dpi)
            if plot_cumsum:
                xis_s = cgram["xis_s"]
                for k in idxs:
                    C_xi = cgram["colossogram"][:, k]
                    cumsum = np.cumsum(C_xi)
                    if species == "Human":
                        xmax = 3000
                    else:
                        xmax = 500
                    xmax_idx = np.argmin(np.abs(xis_s-xmax/1000))
                    plt.close('all')
                    plt.plot(1000*xis_s[0:xmax_idx+1], cumsum[0:xmax_idx+1])
                    plt.xlabel(r"$\xi$ [ms]")
                    plt.ylabel(r"$\int_0^{\xi} C_\xi$")
                    plt.suptitle("Peak-Agnostic")
                    plt.title(f"{species} {wf_idx} - {f[k]:.1f} Hz")
                    folder = os.path.join(dirs[f"cumsums"], f"peak-agnostic")
                    os.makedirs(folder, exist_ok=True)
                    plt.savefig(os.path.join(folder, f"{species} {wf_idx} {f[k]:.0f} Hz - cumsum [peak-agnostic, {cgram_id}].jpg"))

            



                # Grab a 

            # Save data
            # T_xi_sig_idxs = np.flatnonzero(sig_mask)
            for idx in range(len(f)):
                f0 = f[idx]
                T_xi0 = T_xis[idx]
                psd0 = psd[idx]
                row = {'species':species, 'wf_idx':wf_idx, 'f0':f0, 'mode':mode, 'T_xi':T_xi0, 'PSD':psd0, 'wf_fn':wf_fn, 'T_xi_meth':T_xi_meth}
                rows.append(row)

    df = pd.DataFrame(rows)
    fp_sheets = os.path.join(dirs["results"], f"soae_T_xi [cgram {T_xi_id}, {cgram_id}].xlsx")
    df.to_excel(fp_sheets, index=False)





