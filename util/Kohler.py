import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from util.data import resolve_monotone


def plot_rho(Ts, Hs, rho_xxs, cutoff, resu_dir):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.1, 0.9, int(np.max(Ts)) + 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, axs = plt.subplots(1, 1, figsize = (8,7))
    for T, rho_xx in zip(Ts, rho_xxs):
        plt.plot(Hs[cutoff:], rho_xx[cutoff:], '-', color = colors[int(T)], linewidth = 3)
    axs.set_xlabel(r'$B$ [T]', fontsize = 20)
    axs.set_ylabel(r'$\rho_{xx}$ [$\Omega$m]', fontsize = 20)
    axs.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    axs.set_xlim((np.min(Hs[cutoff:]), np.max(Hs[cutoff:])))
    axs.yaxis.get_offset_text().set_size(20)

    cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 0, vmax = np.max(Ts))))
    cb.ax.set_title(r'$T$ (K)', fontsize = 20)
    cb.ax.tick_params(length = 5, width = 1.5, labelsize = 20)

    f.tight_layout()
    f.savefig(os.path.join(resu_dir, 'rho_vs_H.png'))
    plt.close()


def plot_MR(Ts, Hs, MRs, cutoff, resu_dir):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.1, 0.9, int(np.max(Ts)) + 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, axs = plt.subplots(1, 1, figsize = (8,7))
    for T, MR in zip(Ts, MRs):
        plt.plot(Hs[cutoff:], MR[cutoff:], '-', color = colors[int(T)], linewidth = 3)
    axs.set_xlabel(r'$B$ [T]', fontsize = 20)
    axs.set_ylabel(r'$MR$ [%]', fontsize = 20)
    axs.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    axs.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
    axs.set_xlim((np.min(Hs[cutoff:]), np.max(Hs[cutoff:])))
    axs.yaxis.get_offset_text().set_size(20)

    cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 0, vmax = np.max(Ts))))
    cb.ax.set_title(r'$T$ (K)', fontsize = 20)
    cb.ax.tick_params(length = 5, width = 1.5, labelsize = 20)

    f.tight_layout()
    f.savefig(os.path.join(resu_dir, 'MR_vs_H.png'))
    plt.close()


def plot_MRK(Ts, Hs, MRs, rho_xx0s, cutoff, resu_dir):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.1, 0.9, int(np.max(Ts)) + 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, axs = plt.subplots(1, 1, figsize = (8,7))
    for T, MR, rho_xx0 in zip(Ts, MRs, rho_xx0s):
        plt.plot(Hs[cutoff:]/rho_xx0, MR[cutoff:], '-', color = colors[int(T)], linewidth = 3)
    axs.set_xlabel(r'$B/\rho_{0}$ [T$\Omega^{-1}$m$^{-1}$]', fontsize = 20)
    axs.set_ylabel(r'$MR$ [%]', fontsize = 20)
    axs.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    axs.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
    axs.yaxis.get_offset_text().set_size(20)
    axs.set_xscale('log')
    axs.set_yscale('log')

    cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 0, vmax = np.max(Ts))))
    cb.ax.set_title(r'$T$ (K)', fontsize = 20)
    cb.ax.tick_params(length = 5, width = 1.5, labelsize = 20)

    f.tight_layout()
    f.savefig(os.path.join(resu_dir, 'MR_vs_H_rho0.png'))
    plt.close()


def MR_shift(xs0, ys0, xs, ys):
    ymin = max(np.min(ys0), np.min(ys))
    ymax = min(np.max(ys0), np.max(ys))
    mask = (ys < ymax) * (ys > ymin)
    mask0 = (ys0 < ymax) * (ys0 > ymin)
    x_interp = PchipInterpolator(*resolve_monotone(ys[mask > 0], xs[mask > 0]), extrapolate = False)
    return x_interp(ys0[mask0 > 0]) / xs0[mask0 > 0]


def plot_MREK(Ts, Hs, MRs, rho_xx0s, cutoff, resu_dir):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.1, 0.9, int(np.max(Ts)) + 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, axs = plt.subplots(1, 1, figsize = (8,7))
    nTs = []
    for T, MR, rho_xx0 in zip(Ts, MRs, rho_xx0s):
        rho0nTs = MR_shift(Hs[cutoff:], MRs[-1][cutoff:], Hs[cutoff:], MR[cutoff:])
        rho0nT = np.mean(rho0nTs[~np.isnan(rho0nTs)])
        nTs.append(rho0nT/rho_xx0)
        plt.plot(Hs[cutoff:]/rho0nT, MR[cutoff:], '-', color = colors[int(T)], linewidth = 3)
    axs.set_xlabel(r'$B/\rho_{0}n_T$ [$T\cdot\Omega^{-1}\cdot m^{-1}$]', fontsize = 20)
    axs.set_ylabel(r'$MR$ [%]', fontsize = 20)
    axs.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    axs.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
    axs.yaxis.get_offset_text().set_size(20)
    axs.set_xscale('log')
    axs.set_yscale('log')

    cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 0, vmax = np.max(Ts))))
    cb.ax.set_title(r'$T$ (K)', fontsize = 20)
    cb.ax.tick_params(length = 5, width = 1.5, labelsize = 20)

    f.tight_layout()
    f.savefig(os.path.join(resu_dir, 'MR_vs_H_rho0nT.png'))
    plt.close()

    f, axs = plt.subplots(1, 1, figsize = (8,7))
    plt.plot(Ts, nTs, '-o', linewidth = 3,  markersize=10)
    axs.set_xlabel(r'$T$ [K]', fontsize = 20)
    axs.set_ylabel(r'$n_T$ []', fontsize = 20)
    axs.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    axs.yaxis.get_offset_text().set_size(20)

    f.tight_layout()
    f.savefig(os.path.join(resu_dir, 'nT_vs_T.png'))
    plt.close()