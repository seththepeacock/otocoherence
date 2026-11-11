import phaseco as pc
from nddho_generator import nddho_generator
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import numpy as np
from helper_funcs import *
dirs = get_dirs()

# Fixed params shared with SOAE analysis
ppc = get_params_peakc()

init_xi_max_s = ppc["init_xi_max_s"]
beta = ppc["beta"]
T_xi_id = ppc["T_xi_id"]
lam = ppc["lam"]

plot_T_xi = True
plot = True

em = False
str_em = "EM - " if em else ""

# Loop Params
num_iters = 10

"NDDHO Parameters"

# qs = [50, 100, 150, 200, 250, 300]
# qs = [50, 75, 100, 125, 150, 175, 200]
f_ds = [1000, 5000, 10000]
qs = [20, 40, 60, 80, 100]

fs = 44100
# fs = 5000000
wf_len_s = 60

colors = np.concat((get_colors('good'), get_colors('bad'), (get_colors('good'))))

# Global folders
pkl_folder = dirs["pickles_NDDHO"]
psd_folder = dirs["psd_NDDHO"]

"Plotting Parameters"
make_plots = 1
fontsize = 8
output_spreadsheet = 1
plot_scatter = 1
figsize_ind = (5, 5)

"Generate and calculate autocoherence of NDDHO for various Q and f_d"

# Spreadsheet starts now
rows = []
for k_f_d, f_d in enumerate(f_ds):
    # plt.figure(figsize=(19.2 * 0.8, 12 * 0.8))
    for k_q, q in enumerate(qs):
        # Find gamma
        gamma = (f_d*2*np.pi) / np.sqrt(q**2-1/4)

        # # Start plot
        # N_cols = int(round((len(qs) / 2))) if len(qs) > 1 else 1
        # plt.subplot(2, N_cols, k_q + 1)
        for i, color in zip(range(num_iters), colors[0:num_iters]):
            plot = 1 if make_plots and i < 2 else 0
            print(
                f"Q={q} ({k_q+1}/{len(qs)}), f_d={f_d} ({k_f_d+1}/{len(f_ds)}), Iter {i+1}/{num_iters}"
            )


            "Filepaths"

            # NDDHO WF FP
            nddho_wf_id = f"{str_em}Q={q}, f_d={f_d}, len={wf_len_s}, fs={fs}, iter={i}"
            wf_fn = f"{nddho_wf_id} [NDDHO WF].pkl"

            wf_fp = os.path.join(pkl_folder, wf_fn)

            # Load/calc waveform
            if os.path.exists(wf_fp):
                print("Already got this wf, loading!")
                with open(wf_fp, "rb") as file:
                    wf_x = pickle.load(file)
            else:
                print(f"Generating NDDHO {wf_fn}")
                wf_x, wf_y = nddho_generator(f_d, q=q, fs=fs, t_max=wf_len_s, em=em)
                with open(wf_fp, "wb") as file:
                    pickle.dump(wf_x, file)
            
            # Demean
            wf_x -= np.mean(wf_x)


            # Convert from fs
            tau = int(round(ppc["tau_s"] * fs))
            hop = int(round(ppc["hop_s"] * fs))

            # Get psd for fitting to
            f, psd = pc.get_welch(wf_x, fs, tau, hop=hop, win=ppc["win_type"], nfft=ppc["nfft"])
            psd_db = 10*np.log10(psd)
            f_khz = f / 1000

            # Get Lorentzian fits

            fab = fit_and_bpf(wf_x, fs, f, psd, f_d, ppc)
            f_crop = fab['f_crop']
            crop_idxs = fab['crop_idxs']
            lorentz_fit = fab['lorentz_fit']

            # Conversions
            f_crop_khz = f_crop / 1000
            psd_crop_db = psd_db[crop_idxs[0]:crop_idxs[1]]
            lorentz_fit_db = 10*np.log10(lorentz_fit)


            # Crop to extra_bin_fact * filter bandwidth
            extra_bin_fact = 2
            bin_width = f[1]-f[0]
            bw_filt = fab['bw_filt']
            extra_bins = int(round((bw_filt)*extra_bin_fact/(bin_width)))
            f_d_idx = np.argmin(np.abs(f-f_d))
            
            # Do the crop
            crop_plus_slice = slice(f_d_idx - extra_bins, f_d_idx + extra_bins)
            f_crop_plus = f[crop_plus_slice]
            psd_crop_plus = psd[crop_plus_slice]

            # Compute psd on the filtered waveform
            psd_filt = pc.get_welch(fab['wf_filt'], fs, tau, hop=hop, win=ppc["win_type"], nfft=ppc["nfft"])[1]
            psd_filt_db = 10*np.log10(psd_filt)

            # Plot individual fits
            if plot:
                plt.close("all")
                plt.figure(figsize=figsize_ind)
                plt.title(f"Lorentzian FWHM = {2*fab['gamma_L']:.4g} Hz")
                plt.plot(f_crop_plus, psd_crop_plus, label='PSD', color='k', lw=2, alpha=0.7)
                plt.plot(f_crop, lorentz_fit, label="Lorentzian Fit", color='green', lw=10, alpha=0.4)
                plt.plot(f, psd_filt, label="Filtered", color='purple')
                plt.ylabel("PSD")
                plt.legend(loc="upper left")
                plt.xlim(f_crop_plus[[0, -1]])
                plt.savefig(os.path.join(psd_folder, f"NDDHO {f_d} Hz, Q={q}, i={i}, fs={fs} {str_em}[{ppc["filt_id"]}] - PSD"))

            # Get ACF and lags from fab
            acf = fab["acf"]
            acf_phi = fab["acf_phi"]
            lags_s = fab["lags_s"]

            # Get T_xi

            # T_xi, T_xi_est = get_T_xi_int_cor(acf, lags_s, init_xi_max_s, beta)
            # T_xi_phi, T_xi_phi_est = get_T_xi_int_cor(acf_phi, lags_s, init_xi_max_s, beta)
            xi_crop_s = ppc["xi_crop_s"]["NDDHO"]
            T_xi_plot_subfolder = "nddho"
            T_xi_plot_fn = f"{nddho_wf_id} - T_xi [{T_xi_id}, {ppc['filt_id']}].jpg"
            T_xi = get_T_xi_kappa(lags_s, acf, xi_crop_s, lam=lam, subfolder=None, fn=None, plot=False)
            T_xi_phi = get_T_xi_kappa(lags_s, acf_phi, xi_crop_s, lam=lam, subfolder=T_xi_plot_subfolder, fn=T_xi_plot_fn, plot=plot_T_xi)

            N_xi = T_xi * f_d
            N_xi_phi = T_xi_phi * f_d
            row = {'Q':q, 'CF':f_d, 'f0_fit':fab['f0_fit'], 'mode':'W', 'N_xi':N_xi, 'T_xi':T_xi, 'gamma_L':fab['gamma_L'], 'a_L':fab['a_L'], 'wf_fn':wf_fn, 'f0_max':f_d, 'f0_max_round':f_d, 'T_xi_type':ppc["T_xi_meth"]}
            row_phi = {'Q':q, 'CF':f_d, 'f0_fit':fab['f0_fit'], 'mode':'phi', 'N_xi':N_xi_phi, 'T_xi':T_xi_phi, 'gamma_L':fab['gamma_L'], 'a_L':fab['a_L'], 'wf_fn':wf_fn, 'f0_max':f_d, 'f0_max_round':f_d, 'T_xi_type':ppc["T_xi_meth"]}
            for r in [row, row_phi]:
                r['NDDHO Params'] = nddho_wf_id
                r['gamma'] = gamma
                r['iter'] = i
                r['Undamped CF'] = np.sqrt(f_d**2 + (gamma / (4*np.pi))**2)
            
            if 0:
                # Prep to plot T_xi calculation
                xi_max_s_plot = max([T_xi_phi, init_xi_max_s, T_xi_est * beta]) * 1.2
                lags_max_plot_idx = np.argmin(np.abs(lags_s-xi_max_s_plot))
                lags_ms_plot = lags_s[0:lags_max_plot_idx] * 1000
                acf_plot = acf[0:lags_max_plot_idx]
                acf_phi_plot = acf_phi[0:lags_max_plot_idx]
            
                # Plot
                plt.figure(f"{f_d} ACF", figsize=(15, 8))
                plt.suptitle(f"{f_d} Hz - Lorentzian FWHM = {2*fab['gamma_L']:.4g} Hz [{T_xi_id}]")
                
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
                os.makedirs(f"{dirs[f"T_xi_{ppc["T_xi_meth"]}_NDDHO"]} {T_xi_id}", exist_ok=True)
                plt.savefig(os.path.join(f"{dirs[f"T_xi_{ppc["T_xi_meth"]}_NDDHO"]} {T_xi_id}", f"f_d={f_d}, Q={q}, iter={i} {str_em}- ACF [{ppc["T_xi_id"]}, {ppc["filt_id"]}, {nddho_wf_id}].jpg"))            
            rows.append(row)
            rows.append(row_phi)

# Outside of f0 loop now
if output_spreadsheet:
    # Save parameter data as xlsx
    df_fitted_params = pd.DataFrame(rows)
    spreadsheet_fn = os.path.join(dirs["results"], rf"{str_em}NDDHO N_xi Data [{ppc["T_xi_id"]}, {ppc["filt_id"]}].xlsx")
    print(f"Saving to {spreadsheet_fn}")
    df_fitted_params.to_excel(spreadsheet_fn, index=False)

