import numpy as np
from helper_funcs import *
import phaseco as pc
from phaseco import *
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import find_peaks

dirs = get_dirs()

rows = []
for species in ['Tokay','Anole', 'Human', 'Owl',]:
    for wf_idx in range(4):
        "Get waveform"
        wf, wf_fn, fs = get_wf(species=species, wf_idx=wf_idx)
        good_peak_freqs = get_peak_guesses(species=species, wf_idx=wf_idx)
        bad_peak_freqs = []
        print(f"Processing {species} {wf_idx} ({fs}Hz)")
        wf = crop_wf(wf, fs, wf_len_s)

        "PARAMETERS"
        plot = 0
        check_guesses = 0
        ppc = get_params_peakc()
        wf_len_s = ppc["wf_len_s"]
        tau_s = ppc["tau_s_psd"]
        nfft = ppc["nfft_psd"]
        hop_s = ppc["hop_s_psd"]
        tau, hop = int(round(tau_s*fs)), int(round(hop_s*fs))
        win_type = ppc["win_type_psd"]


        max_khzs = {
                        'Anole': 6,
                        'Tokay': 6,
                        'Human': 10,
                        'V Sim Human': 10,
                        'Owl': 12
                    }
                    
        max_khz = max_khzs[species]
        # Get peak bin indices
        fn_id = rf"{species} {wf_idx}, $\tau={tau / fs *1000:.0f}$ms, wf_length={wf_len_s:.3f}s"
        f, psd = pc.get_welch(wf=wf, fs=fs, tau=tau, hop=hop, win=win_type, nfft=nfft)
        
        # Guesses
        # peak_guesses = np.concatenate((good_peak_freqs, bad_peak_freqs))
        peak_guesses = good_peak_freqs

        psd_db = 10*np.log10(psd)
        peak_indices, _ = find_peaks(psd_db, prominence=1, distance=5)

        # # Inspect prominence values of detected peaks
        # prominences = peak_prominences(10*np.log10(psd), peak_indices)[0]

        # # Print them out
        # for i, prom in zip(peak_indices, prominences):
        #     print(f"Peak at f={f[i]:.1f} Hz has prominence {prom:.2f} dB")

        
        f = np.array(f)
        if plot:
            plt.close('all')
            plt.figure(figsize=(10, 5))
            plt.suptitle(fn_id)
            plt.title(rf"Power Spectral Density")
            plt.plot(f, psd_db, label='PSD', c='k')
            
        # for kind, c, peak_guesses in zip(['good', 'bad'], ['g', 'r'], [good_peak_freqs, bad_peak_freqs]):
            # if kind == 'good':
            #     print("Good peak freqs:")
            # elif kind == 'bad':
            #     print('Bad peak freqs:')
        c="g"
        picked_peaks = []
        for peak_guess in peak_guesses:
            # Find index of nearest true peak
            nearest_idx = np.argmin(np.abs(f[peak_indices] - peak_guess))
            
            f_picked_peak_idx = peak_indices[nearest_idx]
            peak = f[f_picked_peak_idx]
            picked_peaks.append(peak)
            if check_guesses:
                if f"{peak:.0f}" != f"{peak_guess}":
                    raise ValueError(f"You didn't copy and paste correctly for {species} {wf_idx} at {peak_guess}Hz, should be {peak:.0f}Hz!")
            if plot:
                plt.scatter(f[f_picked_peak_idx], psd_db[f_picked_peak_idx], marker='x', c=c, s=30)
        picked_peaks = np.array(picked_peaks)
        peak_str = ""
        for peak_freq in picked_peaks:
            peak_str += f"{peak_freq:.0f}, "
        print(peak_str)
        rows.append({"species":species, "wf_idx":wf_idx, "peaks":picked_peaks, "wf_fn":wf_fn})
        if plot:
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD [dB]")  
            plt.legend()
            plt.xlim(0, max_khz*1000)
            plt.tight_layout()
            plt.show()
        

df = pd.DataFrame(rows)
df.to_json(os.path.join(dirs["results"], "picked_peaks.json"), orient="records")

if check_guesses:
    print(f"All picks copied correctly!")