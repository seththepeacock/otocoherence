import os
import phaseco as pc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from collections import defaultdict
from scipy.signal import correlate, get_window, convolve, correlation_lags, hilbert

"Directories"
dirs = {}
dirs["oto"] = r"c:\\Users\\setht\\Dropbox\\Citadel\\GitHub\\otocoherence"
# Get subfolders
for subfolder in ["scripts", "results", "pickles"]:
    dirs[subfolder] = os.path.join(dirs["oto"], subfolder)
# subsubdirs
for results_subfolder in ["psd", "acf"]:
    dirs[results_subfolder] = os.path.join(dirs["results"], results_subfolder)
for dir in dirs.values():
    os.makedirs(dir, exist_ok=True)
os.chdir(dirs["scripts"])
from helper_funcs import *
os.chdir(dirs["oto"])

# Parameters

# ---WF---
speciess = ['Anole', 'Tokay', 'Human', 'Owl']
wf_idxs = range(4)
# speciess = ['Anole']
# wf_idxs = [0]
wf_len_s = 60

# ---PSD---
tau_s_psd = 0.5
win_type_psd = 'hann'
hop_s_psd = 0.25

# ---Plots---
flims = {'Anole':[1, 6], 'Human':[1, 10], 'Owl':[1, 12], 'Tokay':[1, 6]}
bw_crop = 200
figsize_psd = (8, 5)
figsize_ind = (5, 5)
ind_extra_bins = 50
pe_fit = [
    pe.Stroke(linewidth=10, foreground="black", alpha=1),
    pe.Normal(),
]
alpha_fit = 0.3
lw_fit = 5

# ---Filtering---
bw_mult = 7
df_mult = 7
rip = 50

# ---ACF---
max_lag_s = 0.1
acf_fit_min = 0.1
acf_fit_max = 0.9

# ---Filtering---
def filter_wf(wf, gamma, f0, bw_mult, df_mult, rip):
    bw = gamma*bw_mult
    cf = (f0-bw/2, f0+bw/2)
    df = gamma*df_mult
    wf_filt = kaiser_filter(wf, fs, cf, df, rip)
    return wf_filt

# Initialize spreadsheet rows
rows = []

# ID
fn_id = f"bw={bw_mult}gamma, df={df_mult}gamma, rip={rip}db, bw_crop={bw_crop}Hz"

for species in speciess:
    for wf_idx in wf_idxs:
        # Get wf
        wf, wf_fn, fs, gpf, bpf = get_wf(species=species, wf_idx=wf_idx)
        wf = crop_wf(wf, fs, wf_len_s)
        
        # Convert from fs
        tau_psd = int(round(tau_s_psd * fs))
        hop_psd = int(round(hop_s_psd * fs))

        # Get species specific
        flim = flims[species]

        # Get psd for plotting
        f, psd = pc.get_welch(wf, fs, tau_psd, hop=hop_psd, win=win_type_psd)
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
        for f0, color in zip(gpf, get_colors('good')):
            # Crop axes
            f0_exact = f[np.argmin(np.abs(f-f0))]
            crop_idxs = [np.argmin(np.abs(f-(f0_exact-bw_crop/2))), np.argmin(np.abs(f-(f0_exact+bw_crop/2)))+1]
            f_crop = f[crop_idxs[0]:crop_idxs[1]]
            psd_crop = psd[crop_idxs[0]:crop_idxs[1]]
            # Fit Lorentzian
            f0_fit, y0_l, gamma_l, A_l, lorentz_fit = fit_lorentzian(f_crop, psd_crop)
            # Conversions
            f_crop_khz = f_crop / 1000
            psd_crop_db = psd_db[crop_idxs[0]:crop_idxs[1]]
            lorentz_fit_db = 10*np.log10(lorentz_fit)

            # Plot crop and fit on full PSD
            # plt.plot(f_crop_khz, psd_crop_db, color='yellow')
            plt.figure("full")
            plt.plot(f_crop_khz, lorentz_fit_db, label=f"{f0} Hz", color=color)
            
            # Crop to the fit plus some bins (only used for plotting)
            extra_bin_fact = 4
            extra_bins = int(round(gamma_l*extra_bin_fact/(f[1]-f[0])))
            crop_plus_slice = slice(crop_idxs[0] - extra_bins, crop_idxs[1] + extra_bins)
            f_khz_crop_plus = f_khz[crop_plus_slice]
            psd_crop_plus = psd[crop_plus_slice]

            # Filter and demean
            wf_filt = filter_wf(wf, gamma_l, f0_fit, bw_mult, df_mult, rip)
            wf_filt = wf_filt - np.mean(wf_filt)
            psd_filt = pc.get_welch(wf_filt, fs, tau_psd, hop=hop_psd, win=win_type_psd)[1]
            psd_filt_db = 10*np.log10(psd_filt)

            # Plot individual fits
            plt.figure(f"{f0}", figsize=figsize_ind)
            plt.plot(f_khz_crop_plus, psd_crop_plus, label='PSD', color='k', lw=2, alpha=0.7)
            plt.plot(f_crop_khz, lorentz_fit, label="Lorentzian Fit", color='green', lw=5, alpha=0.3)
            plt.plot(f_khz, psd_filt, label="Filtered", color='purple')
            plt.ylabel("PSD")
            plt.legend()
            plt.xlim(f_khz_crop_plus[[0, -1]])
            plt.savefig(os.path.join(dirs["psd"], f"{species} {wf_idx} {f0} Hz [{fn_id}] - PSD"))

            # Get analytic signal
            wf_filt_h = hilbert(wf_filt)
            wf_filt_h_phi = wf_filt_h / np.abs(wf_filt_h)
            acf = correlate(wf_filt_h, wf_filt_h, mode='full', method='auto')
            acf_phi = correlate(wf_filt_h_phi, wf_filt_h_phi, mode='full', method='auto')
            
            # Get lags and crop both to positive lags
            N = len(wf_filt_h)
            lags = correlation_lags(N, N, mode='full')

            # Crop acf
            max_lag_idx = np.argmin(np.abs(lags-max_lag_s*fs))
            acf_slice = slice(N-1, max_lag_idx+1) # Final lag is max_lag_s (inclusive)
            lags = lags[acf_slice]
            acf = acf[acf_slice]
            acf_phi = acf_phi[acf_slice]

            # Normalize acf 
            num_terms = N-lags
            var = np.abs(acf[0])/N
            acf = np.abs(acf)/(num_terms*var)
            acf_phi = np.abs(acf_phi) / num_terms

            # Convert lags to ms
            lags_ms = lags * 1000 / fs

            # Crop to fit exponentials
            acf_crop_slice = slice(np.argmin(np.abs(acf-acf_fit_max)), np.argmin(np.abs(acf-acf_fit_min)))
            acf_phi_crop_slice = slice(np.argmin(np.abs(acf_phi-acf_fit_max)), np.argmin(np.abs(acf_phi-acf_fit_min)))
            # Crop acfs
            acf_crop = acf[acf_crop_slice]
            acf_phi_crop = acf_phi[acf_phi_crop_slice]
            # Crop lags and convert to s and ms
            lags_crop = lags[acf_crop_slice]
            lags_crop_s = lags_crop / fs
            lags_crop_ms = lags_crop_s * 1000
            lags_phi_crop = lags[acf_phi_crop_slice]
            lags_phi_crop_s = lags_phi_crop / fs
            lags_phi_crop_ms = lags_phi_crop_s * 1000
            # Fit exponentials
            A_xi, T_xi, acf_fit = fit_exp(lags_crop_s, acf_crop)
            A_xi_phi, T_xi_phi, acf_phi_fit = fit_exp(lags_phi_crop_s, acf_phi_crop)

            # Plot ACF
            plt.figure(f"{f0} ACF")
            plt.plot(lags_ms, acf, label=r"$P$", color='orange')
            plt.plot(lags_ms, acf_phi, label=r"$\phi$", color='purple')
            plt.plot(lags_crop_ms, acf_fit, color='orange', path_effects=None, lw=lw_fit, alpha=alpha_fit)
            plt.plot(lags_phi_crop_ms, acf_phi_fit, color='purple', path_effects=None, lw=lw_fit, alpha=alpha_fit)
            
            plt.legend()
            plt.savefig(os.path.join(dirs["acf"], f"{species} {wf_idx} {f0} Hz [{fn_id}] - ACF"))
            row = {'species':species, 'wf_idx':wf_idx, 'f0_fit':f0_fit, 'mode':'P', 'T_xi':T_xi, 'A_xi':A_xi, 'gamma_l':gamma_l, 'A_l':A_l, 'fn':wf_fn, 'f0_exact':f0_exact, 'f0_guess':f0}
            row_phi = {'species':species, 'wf_idx':wf_idx, 'f0_fit':f0_fit, 'mode':'phi', 'T_xi':T_xi_phi, 'A_xi':A_xi_phi, 'gamma_l':gamma_l, 'A_l':A_l, 'wf_fn':wf_fn, 'f0_exact':f0_exact, 'f0_guess':f0}
            rows.append(row)
            rows.append(row_phi)

        # Save final full figure
        plt.figure("full")
        plt.legend()
        plt.title(fn_id)
        plt.savefig(os.path.join(dirs["psd"], f"{species} {wf_idx} Full PSD [{fn_id}]"))
        
        


        
        



        