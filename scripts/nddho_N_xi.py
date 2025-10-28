import phaseco as pc
from nddho_generator import nddho_generator
import matplotlib.pyplot as plt
import pickle
import os
import pandas as pd
import numpy as np
from helper_funcs import *

# Fixed params
stop_fit = "frac"
stop_fit_frac = 0.1

# Loop Params
A_consts = [False]
num_iters = 10

# NDDHO params
f_ds = [1000, 2000, 3000]
qs = [25, 30, 35, 40, 45, 50]

# Coherence params
pws = [False]
rho_bw_hops = [(1.0, 50, ('s', 0.01)), (None, 'gamma', ('int', 1))] # (rho, bandwidth, (hop_units, hop))
win_type = 'flattop'
demean = True

colors = get_colors('good')

# Global folders
pkl_folder = os.path.join('pickles', 'NDDHO')
os.makedirs(pkl_folder, exist_ok=True)
tau_pkl_folder = 'pickles'
os.makedirs(tau_pkl_folder, exist_ok=True)

"Plotting Parameters"
gen_plots = 1
show_plots = 0
fontsize = 8
output_spreadsheet = 1
plot_scatter = 1

"Generate and calculate autocoherence of NDDHO for various Q and f_d"
for pw in pws:
    for rho, bw_type, hop_thing in rho_bw_hops:
        for A_const in A_consts:
            # Spreadsheet starts now
            rows = []
            for f_d_idx, f_d in enumerate(f_ds):
                plt.figure(figsize=(19.2 * 0.8, 12 * 0.8))
                for k_q, q in enumerate(qs):
                    # Find gamma
                    gamma = (f_d*2*np.pi) / np.sqrt(q**2-1/4)

                    # Deal with windowing method
                    if rho is None:
                        win_meth = {'method':'static', 'win_type':win_type}
                        
                    else:
                        win_meth = {'method':'rho', 'rho':rho, 'win_type':win_type}
                    # And bandwidth
                    if bw_type == 'gamma':
                        bw = gamma/2
                    else:
                        bw = bw_type
                    # Get strings
                    win_meth_str = pc.get_win_meth_str(win_meth)
                    bw_str = "BW=0.5gamma" if bw_type == 'gamma' else f"BW={bw_type}Hz"
                    relevant_comp_str = f"PW={pw}, {win_meth_str}, {bw_str}, A_const={A_const}"
                    
                    # Start plot
                    N_cols = int(round((len(qs) / 2))) if len(qs) > 1 else 1
                    plt.subplot(2, N_cols, k_q + 1)
                    for iter, color in zip(range(num_iters), colors[0:num_iters]):
                        print(
                            f"Q={q} ({k_q+1}/{len(qs)}), f_d={f_d} ({f_d_idx+1}/{len(f_ds)}), Iter {iter+1}/{num_iters}"
                        )

                        "NDDHO Parameters"

                        fs = 44100
                        wf_len_s = 60

                        "Coherence Parameters"
                        nfft = 2**14
                        const_N_pd = 0
                        tau = get_precalc_tau_from_bw(bw, fs, win_meth['win_type'], tau_pkl_folder)
                        hop = get_hop_from_hop_thing(hop_thing, tau, fs)
                        

                        # f0s_cgram = None
                        f0s_cgram = np.array([f_d])

                        xi_max_s = 15 / gamma

                        # delta_xi_s = max(xi_max_s / 100, 1/fs)
                        delta_xi_s = xi_max_s / 100
                        xi_min_s = delta_xi_s

                        xis = {
                            "xi_min_s": xi_min_s,
                            "delta_xi_s": delta_xi_s,
                            "xi_max_s": xi_max_s,
                        }

                        "Filepaths"
                        
                        # NDDHO WF FP
                        wf_id = f"Q={q}, f_d={f_d}, len={wf_len_s}, fs={fs}, iter={iter}"
                        wf_fn = f"{wf_id} [NDDHO WF].pkl"

                        wf_fp = os.path.join(pkl_folder, wf_fn)

                        # Colossogram FP
                        const_N_pd_str = "N_pd=const" if const_N_pd else "N_pd=max"
                        f0s_str = (
                            ""
                            if f0s_cgram is None
                            else f"f0s={np.array2string(f0s_cgram, formatter={'float' : lambda x: "%.0f" % x})}, "
                        )
                        nfft_str = "" if nfft is None else f"nfft={nfft}, "
                        delta_xi_str = "" if delta_xi_s == 0.001 else f"delta_xi={delta_xi_s*1000:.2g}ms, "
                        cgram_id = rf"PW={pw}, {win_meth_str}, hop={hop}, tau={tau}, xi_max={xi_max_s*1000:.0f}ms, DM={demean}, {delta_xi_str}{nfft_str}{f0s_str}{const_N_pd_str}"
                        cgram_fn = f"{cgram_id}, {wf_id} [COLOSSOGRAM].pkl"
                        cgram_fp = os.path.join(pkl_folder, cgram_fn)

                        # Load/calc waveform
                        if os.path.exists(wf_fp):
                            print("Already got this wf, loading!")
                            with open(wf_fp, "rb") as file:
                                wf_x = pickle.load(file)
                        else:
                            print(f"Generating NDDHO {wf_fn}")
                            wf_x, wf_y = nddho_generator(f_d, q=q, fs=fs, t_max=wf_len_s)
                            with open(wf_fp, "wb") as file:
                                pickle.dump(wf_x, file)

                        # Load/calc colossogram
                        if os.path.exists(cgram_fp):
                            print("Already got this colossogram, loading!")
                            with open(cgram_fp, "rb") as file:
                                cgram_dict = pickle.load(file)

                        else:
                            print(f"Calculating Colossogram ({cgram_id})")
                            # Demean
                            if demean:
                                wf_x -= np.mean(wf_x)
                            cgram_dict = pc.get_colossogram(
                                wf_x,
                                fs,
                                xis=xis,
                                hop=hop,
                                tau=tau,
                                win_meth=win_meth,
                                pw=pw,
                                nfft=nfft,
                                const_N_pd=const_N_pd,
                                f0s=f0s_cgram,
                                return_dict=True,
                            )
                            with open(cgram_fp, "wb") as file:
                                pickle.dump(cgram_dict, file)
                        
                        xis_s = cgram_dict["xis_s"]
                        f = cgram_dict["f"]
                        colossogram = cgram_dict["colossogram"]
                        cgram_dict['pw']=pw
                        # plt.close('all')
                        # pc.plot_colossogram(xis_s, f, colossogram, pw=pw)
                        # plt.show()
                        N_xi, N_xi_dict = pc.get_N_xi(
                            cgram_dict,
                            f_d,
                            A_const=A_const,
                            stop_fit=stop_fit,
                            stop_fit_frac=stop_fit_frac,
                        )

                        # Unpack dictionary
                        f0_bin_center = N_xi_dict["f0_exact"]
                        N_xi = N_xi_dict["N_xi"]
                        N_xi_std = N_xi_dict["N_xi_std"]
                        T_xi = N_xi_dict["T_xi"]
                        T_xi_std = N_xi_dict["T_xi_std"]
                        A_xi = N_xi_dict["A_xi"]
                        A_xi_std = N_xi_dict["A_xi_std"]
                        mse = N_xi_dict["mse"]
                        decayed_idx = N_xi_dict["decayed_idx"]

                        if output_spreadsheet:
                            rows.append(
                                {
                                    "Q": q,
                                    "CF": f_d,
                                    "N_xi": N_xi,
                                    "N_xi_std": N_xi_std,
                                    "T_xi": T_xi,
                                    "T_xi_std": T_xi_std,
                                    "A_xi": A_xi,
                                    "A_xi_std": A_xi_std,
                                    "MSE": mse,
                                    "Iter": iter,
                                    "DFT Freq Bin": f0_bin_center,
                                    "Undamped CF": np.sqrt(f_d**2 + (gamma / (4*np.pi))**2),
                                    "gamma": gamma,
                                    "NDDHO Params": wf_fn,
                                }
                            )

                        if gen_plots:
                            pc.plot_N_xi_fit(
                                N_xi_dict, color=color, plot_noise_floor=False
                            )
                            plt.legend(fontsize=fontsize / 2)
                            plt.title(rf"$Q={q:.0f}$, $\gamma={gamma:.1f}$", fontsize=fontsize)

                # Outside of q/iter loop now (but inside f0 loop)
                if gen_plots:
                    plt.suptitle(
                        rf"f0={f_d}Hz, PW={pw}, A_const={A_const}, {win_meth_str}, hop={hop/fs*1000:.1f}ms, tau={(tau/fs)*1000:.0f}ms, wf_len={wf_len_s}s, fs={fs}Hz",
                        fontsize=fontsize,
                    )
                    plt.tight_layout()
                    results_folder = os.path.join('results', 'nddho', f'NDDHO Results [{relevant_comp_str}]')
                    os.makedirs(results_folder, exist_ok=True)
                    plots_folder = os.path.join(results_folder, 'Fits Varying Q')
                    os.makedirs(plots_folder, exist_ok=True)
                    plot_fp = os.path.join(plots_folder, f"f0={f_d}Hz, {relevant_comp_str}, [FITS VARYING Q].jpg")
                    plt.savefig(plot_fp, dpi=300)
                if show_plots:
                    plt.show()
            # Outside of f0 loop now
            if output_spreadsheet:
                # Save parameter data as xlsx
                df_fitted_params = pd.DataFrame(rows)
                spreadsheet_fn = os.path.join(results_folder, rf"NDDHO Q N_xi Data [{relevant_comp_str}].xlsx")
                print(f"Saving to {spreadsheet_fn}")
                df_fitted_params.to_excel(spreadsheet_fn, index=False)


"Make scatterplot of N_xi vs Q"
if plot_scatter:
    # Plotting Params
    s=25
    fontsize = 10
    marker='*'
    plot_mse = False
    nrows = 2 if plot_mse else 1
    ncols = 3
    fig_size = (ncols*5, nrows*5)
    show_plot = 1

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#126290",
    ]

    for A_const in A_consts:
        for rho, bw_type, hop_thing in rho_bw_hops:
            for pw in pws:
                plt.close('all')
                plt.figure(figsize=fig_size)
                for (f_d_i, f_d), color in zip(enumerate(f_ds), colors[0 : len(f_ds)]):
                    # Deal with windowing method
                    if rho is None:
                        win_meth = {'method':'static', 'win_type':win_type}
                    else:
                        win_meth = {'method':'rho', 'rho':rho, 'win_type':win_type}
                    
                    win_meth_str = pc.get_win_meth_str(win_meth)
                    bw_str = "BW=0.5gamma" if bw_type == 'gamma' else f"BW={bw_type}Hz"
                    relevant_comp_str = f"PW={pw}, {win_meth_str}, {bw_str}, A_const={A_const}"

                    results_folder = os.path.join('results', 'nddho', f'NDDHO Results [{relevant_comp_str}]')
                    os.makedirs(results_folder, exist_ok=True)
                    spreadsheet_fp = os.path.join(results_folder, f"NDDHO Q N_xi Data [{relevant_comp_str}].xlsx")
                    # print("ANALYSIS FROM ", spreadsheet_fp)
                    df = pd.read_excel(spreadsheet_fp)
                    num_iters = df["Iter"].max() + 1
                    qs = df["Q"].unique()

                    for iter in range(num_iters):
                        for q_k, q in enumerate(qs):
                            # Deal with bandwidth

                            row = df.loc[
                                (df["CF"] == f_d) & (df["Q"] == q) & (df["Iter"] == iter)
                            ]

                            N_xi = row["N_xi"].iloc[0]
                            T_xi = row["T_xi"].iloc[0]
                            mse = row["MSE"].iloc[0]
                            # Only add one label per f0
                            label = f"UCF={f_d}Hz" if q_k == 0 and iter == 0 else None

                            # N_xi vs gamma Plot
                            # plt.subplot(nrows, ncols, f0_i + 1)
                            plt.scatter(q, N_xi, label=label, c=color, marker=marker, s=s)
                            plt.legend()
                            plt.xlabel(r"Q", fontsize=fontsize)
                            plt.ylabel(r"$N_\xi$", fontsize=fontsize)
                            # plt.ylim(0, 35)
                            if plot_mse:
                                # MSE vs gamma Plot
                                plt.subplot(nrows, ncols, f_d_i + 1 + 2)
                                plt.scatter(q, mse, label=label, c=color, marker=marker, s=s)
                                plt.legend()
                
                
                if plot_mse:
                    plt.subplot(nrows, ncols, 3)
                    plt.title(r"MSE vs $Q$", fontsize=fontsize)
                    plt.xlabel(r"$Q$", fontsize=fontsize)
                    plt.ylabel(r"MSE", fontsize=fontsize)
                    plt.legend()
                    plt.subplot(nrows, ncols, 4)
                    plt.title(r"MSE vs $Q$", fontsize=fontsize)
                    plt.xlabel(r"$Q$", fontsize=fontsize)
                    plt.ylabel(r"MSE", fontsize=fontsize)
                    plt.legend()

                # Adjust subplots
                sp_space = 0.25
                plt.subplots_adjust(wspace=sp_space, hspace=sp_space)

                # Save Figure
                scatter_fp = os.path.join(results_folder, f'N_xi vs Q [{relevant_comp_str}].jpg')
                # plt.tight_layout()
                plt.suptitle(rf'$N_\xi$ vs $Q$ [{relevant_comp_str}]')
                if show_plots:
                    plt.show()
                plt.savefig(scatter_fp, dpi=300)