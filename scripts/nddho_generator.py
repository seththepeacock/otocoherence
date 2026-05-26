import numpy as np
from scipy.linalg import expm
from tqdm import tqdm




def nddho_generator(f_d, gamma=None, q=None, fs=44100, t_max=60, seed=None):
    """
    Generate a damped noise-driven harmonic oscillator (NDDHO) waveform
    using the exact discrete-time update scheme of Nørrelykke & Flyvbjerg.

    The system obeys
        x'' + gamma x' + k x = η(t),
    where η(t) is normalized unit white noise with
        ⟨η(t) η(t')⟩ = δ(t - t').

    The damping gamma and stiffness k are chosen to realize the requested
    quality factor Q and damped natural frequency f_d.



    Parameters
    ----------
    q : float
        Quality factor.
    f_d : float
        Damped natural frequency (in Hz).
    fs : float
        Sampling frequency (in Hz).
    t_max: float
        Length of waveform to generate (in sec).

    Returns
    -------
    x : ndarray
        Position time series (waveform).
    v : ndarray
        Velocity time series.
    """

    # Define variables
    dt = 1.0 / fs
    omega_d = 2 * np.pi * f_d  # Damped critical frequency
    n_samples = round(t_max * fs)

    # We're assuming m = 1 for simplicity (note k = omega_0**2 / m = omega_0**2, so we'll just use omega_0**2)
    # omega_d^2 = omega_0^2 - gamma^2 / 4 and gamma = omega_0 / q ==> omega_0 = omega_d / np.sqrt(1 - 1 / (4 * q**2))

    # Handle q vs gamma to define the other 
    if (gamma is not None and q is not None) or (gamma is None and q is None):
        raise ValueError("Must have exactly one of gamma and Q!")
    elif q is not None:
        omega_0 = omega_d / np.sqrt(1 - 1 / (4 * q**2))
        gamma = omega_0 / q
        # We also have gamma = omega_d / np.sqrt(q**2-1/4)
    elif gamma is not None:
        omega_0 = np.sqrt(gamma**2+omega_d**2)
        q = omega_0 / gamma
        

    
    if q <= 1/2:
        raise ValueError(f"Can't handle the underdamped case Q={q} <= 1/2!")
    
    # Since their driving force is F_therm(t) = sqrt(2 * k_B * T * gamma) eta(t)
    # and we just want eta(t), we want kBT = 1/(2*gamma)
    kBT = 1 / (2 * gamma)
    # Then we define D (Einstein's relation)
    D = kBT / gamma
    # Note that now sqrt(2D) / tau = (1/gamma) * gamma = 1
    omega = np.sqrt(omega_0**2 - (gamma / 2) ** 2)  # damped critical frequency

    # Define variables for equivalence with paper
    tau = 1 / gamma

    # Drift matrix M (Eq. 5)
    M = np.array([[0, -1], [omega_0**2, 1 / tau]])

    # Exact update: Matrix exponential
    Phi = expm(-M * dt)

    # Variances/covariances of increments (Eqs. 15–17)
    expfac = np.exp(-dt / tau)
    sig_xx2 = (D / (4 * omega**2 * omega_0**2 * tau**3)) * (
        4 * omega**2 * tau**2
        + expfac
        * (
            np.cos(2 * omega * dt)
            - 2 * omega * tau * np.sin(2 * omega * dt)
            - 4 * omega_0**2 * tau**2
        )
    )
    sig_vv2 = (D / (4 * omega**2 * tau**3)) * (
        4 * omega**2 * tau**2
        + expfac
        * (
            np.cos(2 * omega * dt)
            + 2 * omega * tau * np.sin(2 * omega * dt)
            - 4 * omega_0**2 * tau**2
        )
    )
    sig_xv2 = (D / (omega**2 * tau**2)) * expfac * np.sin(omega * dt) ** 2

    # Preallocate
    x = np.empty(n_samples)
    v = np.empty(n_samples)

    # make rng
    rng = np.random.default_rng(seed=seed)

    # sample initial state from stationary Gaussian (zero-mean, cov from Eq.23)
    cov_stationary = np.array([[kBT / omega_0**2, 0.0], [0.0, kBT]])
    ic = rng.multivariate_normal(mean=[0.0, 0.0], cov=cov_stationary)

    # Set initial conditions
    x[0] = ic[0]
    v[0] = ic[1]

    # And set as state
    state = np.array([x[0], v[0]])

    # Correlated Gaussian increments (Eqs. 13–14)
    xis = rng.normal(size=(n_samples-1))
    zetas = rng.normal(size=(n_samples-1))

    sig_xx = np.sqrt(sig_xx2)
    dxs = sig_xx * xis
    dvs = (sig_xv2 / sig_xx) * xis + np.sqrt(sig_vv2 - (sig_xv2**2) / sig_xx2) * zetas
    ds = np.stack((dxs, dvs), axis=1)

    # Iterate
    for j in tqdm(range(1, n_samples), mininterval=1.0):
        # Correlated Gaussian increments (Eqs. 13–14)
        state = Phi @ state + ds[j-1]  # (Eq. 7)
        x[j], v[j] = state

    return x, v

