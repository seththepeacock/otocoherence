# otocoherence

Analysis code for "Spontaneous Otocoherence in the Active Ear." (Peacock et al. 2025).

## Dependencies
The bulk of the calculations rely on our package [phaseco](https://seththepeacock.github.io/phaseco/).

## Data
All spontaneous otoacoustic emission (SOAE) data can be downloaded [here](https://www.dropbox.com/scl/fo/x6svdaiyrwiej4gqky97w/AGTnF23tGA1Mqh_9Amqrceo?rlkey=2xkgmifr3981njduacx2z7e8e&st=ciuzk2t0&dl=0). To run the analysis, put the individual files directly in the `/data` folder.

## Scripts
All analysis code is in `/scripts`. `soae_N_xi.py` will analyze all SOAE waveforms using two windowing methods and generate `.pkl` files of the analysis in `/pickles` for future use in plots; `nddho_N_xi.py` will simulate noise driven damped harmonic oscillator waveforms (using the function in `nddho_generator.py`) and analyze them similarly. `figures_main.ipynb` (and `figures_SI.ipynb`)  will generate all figures.


## Links
- [Preprint](https://seththepeacock.github.io)
- [phaseco](https://seththepeacock.github.io/phaseco/)
- [SOAE Data](https://www.dropbox.com/scl/fo/x6svdaiyrwiej4gqky97w/AGTnF23tGA1Mqh_9Amqrceo?rlkey=2xkgmifr3981njduacx2z7e8e&st=ciuzk2t0&dl=0)
