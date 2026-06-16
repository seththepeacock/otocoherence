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
filt_id = ppc['filt_id']
# T_xi_len is defined in the loop 



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


# ID
# match bpf_params['type']:
#     case 'kaiser':
#         if bpf_params["df"] < 1:
#             df = f"{bpf_params["df"]*100:.0f}p bw"
#         else:
#             df = f"{bpf_params["df"]}hz"
#         bpf_id = f"kaiser, df={df}, rip={bpf_params['rip']}db"
#     case 'exp':
#         bpf_id = f"exp, order={bpf_params['order']}hz"
# filt_id = f"bw={bw_filt_thresh*100:.0f}p max, {bpf_id}, crop={crop_bw}hz, tau={tau_s*1000:.0f}ms, hop_psd={hop_s*1000:.0f}ms, nfft={nfft}, win={win_type}"

for T_xi_len_s in [0.5, 0.025, 0.1]:
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

                fab = fit_and_bpf(wf, fs, f, psd, f0_max_saved, ppc, T_xi_len_s=T_xi_len_s)
                f_crop = fab['f_crop']
                crop_idxs = fab['crop_idxs']
                lorentz_fit = fab['lorentz_fit']

                # Conversions
                f_crop_khz = f_crop / 1000
                psd_crop_db = psd_db[crop_idxs[0]:crop_idxs[1]]
                lorentz_fit_db = 10*np.log10(lorentz_fit)

                # Plot crop and fit on full PSD
                # plt.plot(f_crop_khz, psd_crop_db, color='yellow')
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
                plt.figure(f"{f0_max_round}", figsize=figsize_ind)
                plt.plot(f_khz_crop_plus, psd_crop_plus, label='PSD', color='k', lw=2, alpha=0.7)
                plt.plot(f_crop_khz, lorentz_fit, label="Lorentzian Fit", color='green', lw=5, alpha=0.3)
                plt.plot(f_khz, psd_filt, label="Filtered", color='purple')
                plt.ylabel("PSD")
                plt.legend()
                plt.xlim(f_khz_crop_plus[[0, -1]])
                plt.savefig(os.path.join(dirs["psd"], f"{species} {wf_idx} {f0_max_round} Hz [{filt_id}] - PSD"))

                acf = fab['acf']
                acf_phi = fab['acf_phi']
                lags_s = fab['lags_s']
                lags_ms = lags_s * 1000

                if T_xi_meth == "exp":
                    # Crop to fit exponentials
                    acf_crop_slice = slice(np.argmax(acf < ppc['acf_exp_fit_max']), np.argmax(acf < ppc['acf_exp_fit_min']))
                    acf_phi_crop_slice = slice(np.argmax(acf_phi < ppc['acf_exp_fit_max']), np.argmax(acf_phi < ppc['acf_exp_fit_min']))
                    acf_crop = acf[acf_crop_slice]
                    acf_phi_crop = acf_phi[acf_phi_crop_slice]
                    
                    # Crop lags and convert to s and ms
                    lags_crop_s = lags_s[acf_crop_slice]
                    lags_crop_ms = lags_crop_s * 1000
                    lags_phi_crop_s = lags_s[acf_phi_crop_slice]
                    lags_phi_crop_ms = lags_phi_crop_s * 1000

                    # Fit exponentials
                    A_xi, T_xi, acf_fit = fit_exp(lags_crop_s, acf_crop)
                    A_xi_phi, T_xi_phi, acf_phi_fit = fit_exp(lags_phi_crop_s, acf_phi_crop)

                    # Plot ACF
                    def plotter():
                        plt.plot(lags_ms, acf, label=r"$P$", color='orange')
                        plt.plot(lags_ms, acf_phi, label=r"$\phi$", color='purple')
                        plt.plot(lags_crop_ms, acf_fit, color='orange', path_effects=None, lw=lw_fit, alpha=alpha_fit)
                        plt.plot(lags_phi_crop_ms, acf_phi_fit, color='purple', path_effects=None, lw=lw_fit, alpha=alpha_fit)
                        plt.xlabel(r"$\xi$ [ms]")
                        plt.ylabel(r"$C_\xi$")
                        plt.legend()
                    xmax_ms = (lags_crop_ms[-1])*2
                    row = {'species':species, 'wf_idx':wf_idx, 'f0_fit':fab['f0_fit'], 'mode':'W', 'T_xi':T_xi, 'A_xi':A_xi, 'gamma_L':fab['gamma_L'], 'a_L':fab['a_L'], 'fn':wf_fn, 'f0_max':f0_max_saved, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_meth}
                    row_phi = {'species':species, 'wf_idx':wf_idx, 'f0_fit':fab['f0_fit'], 'mode':'phi', 'T_xi':T_xi_phi, 'A_xi':A_xi_phi, 'gamma_L':fab['gamma_L'], 'a_L':fab['a_L'], 'wf_fn':wf_fn, 'f0_max':f0_max_saved, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_meth}
                    T_xi_id = f"exp_fit_bounds=({ppc['acf_exp_fit_min']}, {ppc['acf_exp_fit_max']}), max_lag={T_xi_len_s}s"
                elif T_xi_meth == "eta":
                    T_xi = get_T_xi_eta(acf, lags_s, eta=ppc['eta'], sig_thresh=ppc['sig_thresh_eta'])
                    T_xi_phi = get_T_xi_eta(acf_phi, lags_s, eta=ppc['eta'], sig_thresh=ppc['sig_thresh_eta'])
                    def plotter():
                        # Plot ACF
                        plt.plot(lags_ms, acf, label=r"$P$", color='orange')
                        plt.plot(lags_ms, acf_phi, label=r"$\phi$", color='purple')
                        plt.vlines(T_xi*1000, 0, 1, color='orange')
                        plt.vlines(T_xi_phi*1000, 0, 1, color='purple')
                        plt.xlabel(r"$\xi$ [ms]")
                        plt.ylabel(r"$C_\xi$")
                        plt.hlines(ppc['sig_thresh_eta'], 0, T_xi_len_s*1000, color='green')
                        plt.legend()
                    # if np.max([T_xi, T_xi_phi]) > 0.5:
                    #     xmax_ms = np.max([T_xi, T_xi_phi])*1.5*1000
                    # else:
                    #     xmax_ms = 0.5*1000
                    xmax_ms = np.max([T_xi, T_xi_phi])*2*1000
                    row = {'species':species, 'wf_idx':wf_idx, 'f0_fit':fab['f0_fit'], 'mode':'W', 'T_xi':T_xi, 'gamma_L':fab['gamma_L'], 'a_L':fab['a_L'], 'wf_fn':wf_fn, 'f0_max':f0_max_saved, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_meth}
                    row_phi = {'species':species, 'wf_idx':wf_idx, 'f0_fit':fab['f0_fit'], 'mode':'phi', 'T_xi':T_xi_phi, 'gamma_L':fab['gamma_L'], 'a_L':fab['a_L'], 'wf_fn':wf_fn, 'f0_max':f0_max_saved, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_meth}
                    T_xi_id = f"eta={ppc['eta']}, sig_thresh={ppc['sig_thresh_eta']}, max_lag={ppc['T_xi_len_s']}s"
                elif T_xi_meth == "int":
                    T_xi = get_T_xi_int(acf, lags_s)
                    T_xi_phi = get_T_xi_int(acf_phi, lags_s)
                    def plotter():
                        # Plot ACF
                        plt.plot(lags_ms, acf, label=r"$P$", color='orange')
                        plt.plot(lags_ms, acf_phi, label=r"$\phi$", color='purple')
                        plt.vlines(T_xi*1000, 0, 1, color='orange')
                        plt.vlines(T_xi_phi*1000, 0, 1, color='purple')
                        plt.xlabel(r"$\xi$ [ms]")
                        plt.ylabel(r"$C_\xi$")
                        plt.legend()
                    xmax_ms = np.max([T_xi, T_xi_phi])*2*1000
                    row = {'species':species, 'wf_idx':wf_idx, 'f0_fit':fab['f0_fit'], 'mode':'W', 'T_xi':T_xi, 'gamma_L':fab['gamma_L'], 'a_L':fab['a_L'], 'wf_fn':wf_fn, 'f0_max':f0_max_saved, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_meth}
                    row_phi = {'species':species, 'wf_idx':wf_idx, 'f0_fit':fab['f0_fit'], 'mode':'phi', 'T_xi':T_xi_phi, 'gamma_L':fab['gamma_L'], 'a_L':fab['a_L'], 'wf_fn':wf_fn, 'f0_max':f0_max_saved, 'f0_max_round':f0_max_round, 'T_xi_type':T_xi_meth}
                    T_xi_id = f"int {T_xi_len_s*1000:.0f}ms"
                else:
                    raise ValueError(f"{T_xi_meth} is not a valid T_xi_type")
                plt.figure(f"{f0_max_round} ACF", figsize=(12, 8))
                plt.suptitle(f"{species} {wf_idx} - {f0_max_round} Hz")
                plt.subplot(1, 2, 1)
                plotter()
                plt.subplot(1, 2, 2)
                plotter()
                plt.xlim(0, xmax_ms)
                plt.savefig(os.path.join(dirs[f"T_xi_{T_xi_meth}"], f"{species} {wf_idx} {f0_max_round} Hz - ACF [{T_xi_id}, {filt_id}].jpg"))

                
                rows.append(row)
                rows.append(row_phi)

                # Save final full figure
                plt.figure("full")
                plt.legend()
                plt.title(filt_id)
                plt.savefig(os.path.join(dirs["psd"], f"{species} {wf_idx} Full PSD [{filt_id}]"))

    df = pd.DataFrame(rows)
    fp_sheets = os.path.join(dirs["results"], f"soae_T_xi [{T_xi_id}, {filt_id}].xlsx")
    df.to_excel(fp_sheets, index=False)
        


        
        



        