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
speciess = ['Human', 'Anole', 'Tokay', 'Owl']
wf_idxs = range(4)

# Get variables from parameter dictionary
ppc = get_params_peakc()
wf_len_s = ppc['wf_len_s']
tau_s = ppc['tau_s']
nfft = ppc['nfft']
win_type = ppc['win_type']
hop_s = ppc['hop_s']
T_xi_meth = ppc['T_xi_meth']
T_xi_id = ppc['T_xi_id']
filt_id = ppc['filt_id']
init_xi_max_s = ppc['init_xi_max_s']
beta = ppc['beta']
lam = ppc['lam']

# ---Plots---
flims = {'Anole':[1, 6], 'Human':[1, 10], 'Owl':[1, 12], 'Tokay':[1, 6]}
figsize_psd = (8, 5)
figsize_ind = (5, 5)
ind_extra_bins = 50
pe_fit = [
    pe.Stroke(linewidth=10, foreground="black", alpha=1),
    pe.Normal(),
]
alpha_fit = 0.3
lw_fit = 5
plot_T_xi = True
plot = False


# Initialize spreadsheet rows
rows = []
for species in speciess:
    for wf_idx in wf_idxs:
    
        print(f"{species} {wf_idx}")
        # Get wf
        wf, wf_fn, fs = get_wf(species=species, wf_idx=wf_idx)
        fp_pp = os.path.join(dirs["results"], "picked_peaks.json")
        picked_peaks = get_picked_peaks(fp_pp, wf_fn)
        
        # Convert from fs
        tau = int(round(tau_s * fs))
        hop = int(round(hop_s * fs))

        # Get species specific flims
        flim = flims[species]

        # Get psd for fitting to
        f, psd = pc.get_welch(wf, fs, tau, hop=hop, win=win_type, nfft=nfft)
        psd_db = 10*np.log10(psd)
        f_khz = f / 1000

        # Plot PSD
        if plot:
            plt.close('all')
            plt.figure("full", figsize=figsize_psd)
            plt.plot(f_khz, psd_db, label='PSD', color='k', lw=5)
            plt.xlim(flims[species])
            plt.ylabel("PSD [dB]")
            plt.xlabel("Frequency [kHz]")
            plt.suptitle(wf_fn)

        # Get Lorentzian fits
        for f0_max_saved, color in zip(picked_peaks, get_colors('good')):
            f0_max_round = int(round(f0_max_saved)) # Useful for user-facing things

            fab = fit_and_bpf(wf, fs, f, psd, f0_max_saved, ppc)
            f_crop = fab['f_crop']
            crop_idxs = fab['crop_idxs']
            lorentz_fit = fab['lorentz_fit']

            # Conversions
            f_crop_khz = f_crop / 1000
            psd_crop_db = psd_db[crop_idxs[0]:crop_idxs[1]]
            lorentz_fit_db = 10*np.log10(lorentz_fit)

            # Plot crop and fit on full PSD
            # plt.plot(f_crop_khz, psd_crop_db, color='yellow')
            if plot:
                plt.figure("full")
                plt.plot(f_crop_khz, lorentz_fit_db, label=f"{f0_max_saved:.0f} Hz", color=color)
                
                
            # Crop to the fit plus some bins (only used for plotting)
            extra_bin_fact = 1
            bin_width = f[1]-f[0]
            bw_filt = fab['bw_filt']
            extra_bins = int(round((bw_filt)*extra_bin_fact/(bin_width)))
            
            # Do the crop
            crop_plus_slice = slice(crop_idxs[0] - extra_bins, crop_idxs[1] + extra_bins)
            f_khz_crop_plus = f_khz[crop_plus_slice]
            psd_crop_plus = psd[crop_plus_slice]

            # Compute psd on the filtered waveform
            psd_filt = pc.get_welch(fab['wf_filt'], fs, tau, hop=hop, win=win_type, nfft=nfft)[1]
            psd_filt_db = 10*np.log10(psd_filt)

            # Plot individual lorentzian fit
            if plot:
                plt.figure(f"{f0_max_round}", figsize=figsize_ind)
                plt.plot(f_khz_crop_plus, psd_crop_plus, label='PSD', color='k', lw=2, alpha=0.7)
                plt.plot(f_crop_khz, lorentz_fit, label="Lorentzian Fit", color='green', lw=5, alpha=0.3)
                plt.plot(f_khz, psd_filt, label="Filtered", color='purple')
                plt.ylabel("PSD")
                plt.legend()
                plt.xlim(f_khz_crop_plus[[0, -1]])
                plt.savefig(os.path.join(dirs["psd"], f"{species} {wf_idx} {f0_max_round} Hz [{filt_id}] - PSD"))


            
            # Calculate T_xi
            acf = fab['acf']
            acf_phi = fab['acf_phi']
            lags_s = fab['lags_s']
            lags_ms = lags_s * 1000

            # T_xi, T_xi_est = get_T_xi_int_cor(acf, lags_s, init_xi_max_s, beta)
            # T_xi_phi, T_xi_phi_est = get_T_xi_int_cor(acf_phi, lags_s, init_xi_max_s, beta)
            xi_crop_s = ppc["xi_crop_s"][species]
            T_xi_plot_subfolder = "peak-centric"
            T_xi_plot_fn = f"{species} {wf_idx} {f0_max_round:.0f} Hz - T_xi [{T_xi_id}, {filt_id}].jpg"
            T_xi = get_T_xi_kappa(lags_s, acf, xi_crop_s, lam=lam, subfolder=None, fn=None, plot=False)
            T_xi_phi = get_T_xi_kappa(lags_s, acf_phi, xi_crop_s, lam=lam, subfolder=T_xi_plot_subfolder, fn=T_xi_plot_fn, plot=plot_T_xi)

            # Prep to plot T_xi calculation (old way)
            if 0:
                xi_max_s_plot = max([T_xi_phi, init_xi_max_s, T_xi_est * beta]) * 1.2
                lags_max_plot_idx = np.argmin(np.abs(lags_s-xi_max_s_plot))
                lags_ms_plot = lags_s[0:lags_max_plot_idx] * 1000
                acf_plot = acf[0:lags_max_plot_idx]
                acf_phi_plot = acf_phi[0:lags_max_plot_idx]
            
                # Plot
                plt.figure(f"{f0_max_round} ACF", figsize=(15, 8))
                plt.suptitle(f"{species} {wf_idx} - {f0_max_round} Hz [{T_xi_id}]")
                
                # Plot the first round
                plt.subplot(1, 2, 1)
                plt.title(rf"$T_\xi^\text{{est}} =$ {T_xi_phi_est*1000:.3g} ms")
                plt.plot(lags_ms_plot, acf_phi_plot, color='k')
                plt.vlines(init_xi_max_s*1000, 0, 1, color='red', label=r'$\xi_\text{max}^\text{init}$')
                idx = np.argmin(np.abs(lags_ms_plot-init_xi_max_s*1000))
                x = lags_ms_plot[0:idx]
                zeros = np.zeros(len(x))
                plt.fill_between(x, zeros, acf_phi_plot[0:idx], color='gray', alpha=0.3)
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
                plt.fill_between(x, zeros, acf_phi_plot[0:idx], color='grey', alpha=0.3)
                plt.vlines(T_xi_phi*1000, 0, 1, color='purple', label=r'$T_{\xi}$')
                plt.xlabel(r"$\xi$ [ms]")
                plt.ylabel(r"$C_\xi$")
                plt.legend(loc='upper right')

                # Save Fig
                os.makedirs(os.path.join(dirs[f"T_xi_{T_xi_meth}"], f"Peakc {T_xi_id}"), exist_ok=True)
                plt.savefig(os.path.join(dirs[f"T_xi_{T_xi_meth}"], f"Peakc {T_xi_id}", f"{species} {wf_idx} {f0_max_round} Hz - ACF [{T_xi_id}, {filt_id}].jpg"))
                # plt.savefig(os.path.join(dirs[f"T_xi_{T_xi_meth}"], f"LONG XMAX Peakc {T_xi_id} {species} {wf_idx} {f0_max_round} Hz - ACF [].jpg"))

            row = {'species':species, 'wf_idx':wf_idx, 'f0_fit':fab['f0_fit'], 'mode':'W', 'T_xi':T_xi, 'gamma_L':fab['gamma_L'], 'a_L':fab['a_L'], 'wf_fn':wf_fn, 'f0_max':f0_max_saved, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_meth}
            row_phi = {'species':species, 'wf_idx':wf_idx, 'f0_fit':fab['f0_fit'], 'mode':'phi', 'T_xi':T_xi_phi, 'gamma_L':fab['gamma_L'], 'a_L':fab['a_L'], 'wf_fn':wf_fn, 'f0_max':f0_max_saved, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_meth}
            
            rows.append(row)
            rows.append(row_phi)


        # Save final full PSD figure
        plt.figure("full")
        # plt.legend()
        plt.title(filt_id)
        plt.savefig(os.path.join(dirs["psd"], f"{species} {wf_idx} Full PSD [{filt_id}]"))

    df = pd.DataFrame(rows)
    fp_sheets = os.path.join(dirs["results"], f"soae_T_xi [{T_xi_id}, {filt_id}].xlsx")
    df.to_excel(fp_sheets, index=False)
        


        
        



        