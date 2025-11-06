import numpy as np
from helper_funcs import *
import phaseco as pc
from phaseco import *
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# for species in ['Anole', 'Human', 'Owl', 'Tokay',]:
for species in ['Tokay']:
    for wf_idx in range(4):
        "Get waveform"
        wf, wf_fn, fs, good_peak_freqs, bad_peak_freqs = get_wf(species=species, wf_idx=wf_idx)
        print(f"Processing {species} {wf_idx} ({fs}Hz)")
        wf_len_s = 60
        wf = crop_wf(wf, fs, wf_len_s)

        "PARAMETERS"
        plot = 1
        check_guesses = 1
        tau = 2**14
        win_type = 'hann'
        # Everyone can use the same tau; slightly less finegrained for owl but it's already way oversampled
        
        max_khzs = {
                        'Anole': 6,
                        'Tokay': 6,
                        'Human': 10,
                        'V Sim Human': 10,
                        'Owl': 12
                    }
                    
        max_khz = max_khzs[species]
        # Get peak bin indices
        fig_folder = r'N_xi Fits/Auto Peak Picks'
        fn_id = rf"{species} {wf_idx}, $\tau={tau / fs *1000:.0f}$ms, wf_length={wf_len_s:.3f}s"
        f, psd = pc.get_welch(wf=wf, fs=fs, tau=tau, win=win_type)
        
        # Guesses
        peak_guesses = np.concatenate((good_peak_freqs, bad_peak_freqs))

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
            
        for kind, c, peak_guesses in zip(['good', 'bad'], ['g', 'r'], [good_peak_freqs, bad_peak_freqs]):
            peak_guesses = peak_guesses
            picked_peaks = []
            if kind == 'good':
                print("Good peak freqs:")
            elif kind == 'bad':
                print('Bad peak freqs:')
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
            
        if plot:
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("PSD [dB]")  
            plt.legend()
            plt.xlim(0, max_khz*1000)
            plt.tight_layout()
            plt.show()
        


if check_guesses:
    print(f"All picks copied correctly!")