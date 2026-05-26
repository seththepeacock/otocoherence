"Init"
from helper_funcs import *
import helper_funcs
import importlib
importlib.reload(helper_funcs)

from collections import defaultdict
import numpy as np
import pandas as pd

dirs = get_dirs(root="C:\\Users\\setht\\Dropbox\\Citadel\\GitHub\\otocoherence")
dpi = 500

# Get data
def get_data(T_xi_types):
    data = defaultdict(lambda: defaultdict(dict))

    for T_xi_type in T_xi_types:
        # Read df
        df = pd.read_excel(os.path.join(
            dirs["results"], rf"soae_T_xi [{T_xi_type}].xlsx"
        ))

        dfs = {}
        dfs["P"] = df[df["mode"]=="P"]
        dfs["phi"] = df[df["mode"]=="phi"]
        has_P = not (len(dfs["P"]) == 0)

        # Gather everything into lists
        if "cgram" in T_xi_type:
            prop_list = ["T_xi", "species", "wf_idx", "f0", "PSD"]
            N_xi_multiplier = "f0"
        else:
            prop_list = ["T_xi", "species", "wf_idx", "f0_max", "f0_fit"]
            N_xi_multiplier = "f0_max"

        for mode in ["phi", "P"]:
            if not has_P and mode == "P":
                continue
            for prop in prop_list:
                data[T_xi_type][mode][prop] = np.array(dfs[mode][prop])
  
            # Calculate derived data
            data[T_xi_type][mode]["N_xi"] = data[T_xi_type][mode]["T_xi"] * data[T_xi_type][mode][N_xi_multiplier]
            data[T_xi_type][mode]["inv_T_xi"] = 1/(data[T_xi_type][mode]["T_xi"])
        if has_P:
            data[T_xi_type]["T_xi_diffs"] = data[T_xi_type]["P"]["T_xi"] - data[T_xi_type]["phi"]["T_xi"]
            data[T_xi_type]["N_xi_diffs"] = data[T_xi_type]["P"]["N_xi"] - data[T_xi_type]["phi"]["N_xi"]

    return data

# Define universal variables

# Get peak centric variables from parameter dictionary
ppc = get_params_peakc()
pcg = get_params_cgram()


filt_id_peakc = ppc['filt_id']
meth_id_cgram = pcg['meth_id']

# Plotting parameters
markA='.'  # C_tau
markB='+'  # C_omega
markC='s'  # C_xi 
fontsize_ac = 14
dpi=500
fontsize_cgram = 40
labelpad = 20
fontsize_ticks = 20

lcc_kwargs = {
    "wf_len_s": pcg['wf_len_s'],
    "filter_meth": pcg['hpf_meth'],
    "mode":pcg['mode'],
    "xi_min_s": pcg['xi_min_s'],
    "win_meth": pcg['win_meth_cgram'],
    "nfft": pcg['nfft'],
    "pkl_folder": dirs["pickles"],
}

# Load noise floor
nf_fp = os.path.join(dirs["data"], 'testSOAEsupp1.txt')   # file name
nf_data = np.loadtxt(nf_fp)
f_nf = nf_data[:,0]
mags_nf = nf_data[:,1]

###

"Init"
from helper_funcs import *
import helper_funcs
import importlib
importlib.reload(helper_funcs)

from collections import defaultdict
import numpy as np
import pandas as pd

dirs = get_dirs(root="C:\\Users\\setht\\Dropbox\\Citadel\\GitHub\\otocoherence")
dpi = 500

# Get data
def get_data(T_xi_types):
    data = defaultdict(lambda: defaultdict(dict))

    for T_xi_type in T_xi_types:
        # Read df
        df = pd.read_excel(os.path.join(
            dirs["results"], rf"soae_T_xi [{T_xi_type}].xlsx"
        ))

        dfs = {}
        dfs["P"] = df[df["mode"]=="P"]
        dfs["phi"] = df[df["mode"]=="phi"]
        has_P = not (len(dfs["P"]) == 0)

        # Gather everything into lists
        if "cgram" in T_xi_type:
            prop_list = ["T_xi", "species", "wf_idx", "f0", "PSD"]
            N_xi_multiplier = "f0"
        else:
            prop_list = ["T_xi", "species", "wf_idx", "f0_max", "f0_fit"]
            N_xi_multiplier = "f0_max"

        for mode in ["phi", "P"]:
            if not has_P and mode == "P":
                continue
            for prop in prop_list:
                data[T_xi_type][mode][prop] = np.array(dfs[mode][prop])
  
            # Calculate derived data
            data[T_xi_type][mode]["N_xi"] = data[T_xi_type][mode]["T_xi"] * data[T_xi_type][mode][N_xi_multiplier]
            data[T_xi_type][mode]["inv_T_xi"] = 1/(data[T_xi_type][mode]["T_xi"])
        if has_P:
            data[T_xi_type]["T_xi_diffs"] = data[T_xi_type]["P"]["T_xi"] - data[T_xi_type]["phi"]["T_xi"]
            data[T_xi_type]["N_xi_diffs"] = data[T_xi_type]["P"]["N_xi"] - data[T_xi_type]["phi"]["N_xi"]

    return data

# Define universal variables

# Get peak centric variables from parameter dictionary
ppc = get_params_peakc()
pcg = get_params_cgram()


filt_id_peakc = ppc['filt_id']
meth_id_cgram = pcg['meth_id']

# Plotting parameters
markA='.'  # C_tau
markB='+'  # C_omega
markC='s'  # C_xi 
fontsize_ac = 14
dpi=500
fontsize_cgram = 40
labelpad = 20
fontsize_ticks = 20

lcc_kwargs = {
    "wf_len_s": pcg['wf_len_s'],
    "filter_meth": pcg['hpf_meth'],
    "mode":pcg['mode'],
    "xi_min_s": pcg['xi_min_s'],
    "win_meth": pcg['win_meth_cgram'],
    "nfft": pcg['nfft'],
    "pkl_folder": dirs["pickles"],
}

# Load noise floor
nf_fp = os.path.join(dirs["data"], 'testSOAEsupp1.txt')   # file name
nf_data = np.loadtxt(nf_fp)
f_nf = nf_data[:,0]
mags_nf = nf_data[:,1]

"Fig? (T_xi_exp)"



# Init plot
fig = plt.figure(figsize=(10, 10))
for k, (species, wf_idx, f0) in enumerate([("Human", 3, 904.0), ("Human", 0, 3220.0)]):

    # Get waveform
    wf, wf_fn, fs = get_wf(
        species=species,
        wf_idx=wf_idx,
    )


    color1="#07586E"
    color2="#8C60B3"
    fsz=40
    labelpad=20
    lw = 5
    lw_fit=10
    lw_stroke=5
    alpha_fit = 0.5
    alpha_fit_stroke =1
    zorder_fit = 1
    pe_stroke_fit = [
        pe.Stroke(linewidth=lw_fit + lw_stroke, foreground="black", alpha=alpha_fit_stroke),
        pe.Normal(),
    ]

    "Dynamic windowing"
    tau_cgram = int(round(pcg['tau_s'] * fs))
    hop_cgram = int(round(pcg['hop_cgram_s'] * fs))
    xi_max_s = 1.0

    # Load Colossogram
    cgram_dict = load_calc_colossogram(
        **(
            lcc_kwargs
            | {
                "xi_max_s": xi_max_s,
                "species": species,
                "fs": fs,
                "tau": tau_cgram,
                "hop": hop_cgram,
                "wf": wf,
                "wf_idx": wf_idx,
                "wf_fn": wf_fn,
                "f0s":np.array([f0]),
            }
        )
    )

    # Fitting Parameters
    N_xi, N_xi_dict = pc.get_N_xi(
        cgram_dict,
        f0,
    )
    C_xi = N_xi_dict["colossogram_slice"]
    xis_s = N_xi_dict["xis_s"]
    exp_fit = N_xi_dict["fitted_decay"]
    xis_exp_s = N_xi_dict["xis_s_fit_crop"]


    # MAKE PLOT
    plt.subplot(2, 2, 1 + k)
    plt.plot(xis_s*1000, C_xi, lw=lw, color=color2)
    plt.plot(xis_exp_s*1000, exp_fit, color=color2, lw=lw_fit, path_effects=pe_stroke_fit, alpha=alpha_fit, zorder=zorder_fit) 
    # pc.plot_N_xi_fit(N_xi_dict, color=color1, plot_noise_floor=False, s_signal=50, lw_fit=20, lw_stroke=5)
    # plt.xlabel(r"$\xi$ [ms]", labelpad=labelpad, fontsize=fsz)
    plt.ylabel(r"$C_\xi^\phi$", labelpad=labelpad, fontsize=fsz)
    plt.title("")
    plt.tick_params('both', labelsize=fsz-15)
    plt.xlim(0, 850)


    "Peak-Centric"
    plt.subplot(2, 2, 2 + 2*k)
    # Get ppc params for this fs
    tau_ppc, hop_ppc = int(round(ppc["tau_s"] * fs)), int(round(ppc["hop_s"] * fs)) 
    # Get PSD For fitting to
    f, psd = pc.get_welch(wf, fs, tau_ppc, hop=hop_ppc, win=ppc["win_type"], nfft=ppc["nfft"])
    fab = fit_and_bpf(wf, fs, f, psd, f0, ppc)
    C_xi = fab["acf_phi_full"]
    xis_s = fab["lags_full_s"]
    crop_slice = slice(np.argmax(C_xi < ppc['acf_exp_fit_max']), np.argmax(C_xi < ppc['acf_exp_fit_min']))
    C_xi_exp_crop, xis_exp_s = C_xi[crop_slice], xis_s[crop_slice]
    _, _, exp_fit = fit_exp(xis_exp_s, C_xi_exp_crop)


    # Plot
    plt.plot(xis_s*1000, C_xi, lw=lw, color=color2)
    plt.plot(xis_exp_s*1000, exp_fit, color=color2, lw=lw_fit, path_effects=pe_stroke_fit, alpha=alpha_fit, zorder=zorder_fit) 
    plt.xlabel(r"$\xi$ [ms]", labelpad=labelpad, fontsize=fsz)
    # plt.ylabel(r"$C_\xi^\phi$", labelpad=labelpad, fontsize=fsz)
    plt.title("")
    # plt.legend(fontsize=fsz-20)
    plt.tick_params('both', labelsize=fsz-15)
    plt.xlim(0, 850)




        

    plt.tight_layout()
    plt.savefig(os.path.join(dirs["figs"], f'Fig S1 (exp fit).jpg'),dpi=dpi, bbox_inches='tight')