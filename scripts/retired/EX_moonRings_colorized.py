#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EX_moonRings.py

Purpose: Inter-radial rings of moon-like shapes....
(artistic version of EX_PhaseLockingValue.py)

Created on Mon May  4 17:03:49 2026
@author: pumpkin
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

def mix_colors(rgb1, rgb2, weight=0.5):
    """
    Mixes two RGB tuples.
    weight: 0.0 results in rgb2, 1.0 results in rgb1.
    """
    r = int(rgb1[0] * weight + rgb2[0] * (1 - weight))
    g = int(rgb1[1] * weight + rgb2[1] * (1 - weight))
    b = int(rgb1[2] * weight + rgb2[2] * (1 - weight))
    color = np.array((r, g, b)) / 255 # Normalize to [0, 1]
    return color

rgb1 = (42, 1, 52) # dark purple
rgb2 = (193, 198, 252) # light periwinkle

# ========= [User Params]  ==================
ringNum= 7  # number of coecentric rings to plot {7}
ringNoise= 0.1  # scale factor to widen rings {0.1}
ringDirection= 1  # boolean to randomize ring centroid direction {1}


# ==== more technical params.
# --- wf1
f1= 1300   # wf1 sin. freq [Hz] {1300}
A1= 1.0      # wf1 amplitude of sinusoid {1}
An1= 0.4      # wf1 noise amplitude {0.4}
phi1= 0    # wf1 phase offset {0}
# --- wf2
f2= 1300   # wf2 sin. freq [Hz] {1300}
A2= 1.0      # wf2 amplitude of sinusoid {1}
An2= 0.4      # wf2 noise amplitude {0.4}
phi2= 0*np.pi    # wf2 phase offset {0}
# -----
SR= 44100;         # sample rate [Hz] {44100}
Npts= 8192*16;     # length of fft window (# of points) {8192*16}

# ====================================


# ====================================
# ==== bookeeping
# --- create a freq. array (for FFT bin labeling)
freq= np.arange(0,(Npts+1)/2,1)    
freq= SR*freq/Npts
df = SR/Npts   # freq. bin width
t= np.linspace(0,(Npts-1)/SR,Npts)   # time array

# =======================================================================
# ==== set up plot visualize
plt.close("all")

# +++ FIG.2 - Polar plot of phase diffs
figP, axP = plt.subplots()
axP.axis('off')
axP = figP.add_subplot(111, projection='polar')
#axP.plot(deltaPhi,r,'ko',alpha=0.01,ms=2) 
axP.set_rticks([1])
axP.axis('off')

# ====== loop through each randomized ring

for n in range(ringNum):
    # ---
    if (ringDirection==1):
        phi1= 2*np.pi*np.random.rand(1)
    # --- create noisy noise ;-)
    noiseT1= np.random.normal(0,1,Npts)
    noiseT2= np.random.normal(0,1,Npts)
      # --- create waveforms      
    wf1= (An1*noiseT1) + A1*np.sin(f1*2*np.pi*t+ phi1)
    wf2= (An2*noiseT2) + A2*np.sin(f2*2*np.pi*t+ phi2)
    
    # ======= Hilbert transform --> extract inst. phases
    AS1= hilbert(wf1)   # compute the analytic signal
    env1= np.abs(AS1)   # extract the envelope
    phase1= np.angle(AS1) # extract inst. phase
    AS2= hilbert(wf2)   # 
    env2= np.abs(AS2)   # 
    phase2= np.angle(AS2)
    # ======= Compute phase diff. and PLV (as well as RMS+RMSD for each)
    deltaPhi= phase1-phase2  # compute phase diff
    # --- create some jittery pseudo-unit radial vectors for phase diff. polar plot;
    # incl. some jitter to improve visualization
    r = n+ np.ones(len(deltaPhi))+ ringNoise*np.random.randn(len(deltaPhi))
    # get random color somewhere rgb1 and rgb2
    color = mix_colors(rgb1, rgb2, weight=np.random.rand())
    axP.plot(deltaPhi,r,marker="o",color=color,alpha=0.01,ms=2) 

plt.show()
# =======================================================================

"""
Note froms EX_PhaseLockingValue.py

Purpose: Demonstrate utility of the PLV (Phase Locking Value) via a 
Hilbert transform as per Aydore et al 2013 to assess a "vector strength"
measure relating two noisy sinusoids

Notes
o Easiest to understand when f1~f2
o Useful observation is that when f1=f2 and one sinusoid is relatively
quite noisy, PLVs seem to suggest the fraction of time over
which the phase diff. is coherent (i.e., constant). You can see this 
visually in the unwrapped phase difference (sloped when incoherent,
flat when coherent). Not sure how firm this is, just an observation.
o Note that rectifying one of the waveforms (wf1) such that half the time 
it has one phase diff re wf2 and the other half that phase diff is +pi has
little effect on the PLV, presumably  because the Hilbert transform is not
sensitive to such (i.e., the phase is unaffected while the change manifests
in the envelope)  
"""

