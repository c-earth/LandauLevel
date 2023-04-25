import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from extract import resolve_monotone


def plot_rho(Ts, Hs, rho_xxs, cutoff, resu_dir):
    colors = plt.cm.jet(np.linspace(0, 1, int(np.max(Ts)) + 1))

    plt.subplots(1, 1, figsize = (8,7))
    for T, rho_xx in zip(Ts, rho_xxs):
        plt.plot(Hs[cutoff:], rho_xx[cutoff:], '-', color = colors[int(T)])
    
    plt.xlabel(r'$H$ [$T$]')
    plt.ylabel(r'$\rho_{xx}$ [$\Omega\cdot m$]')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(resu_dir, 'rho_vs_H.png'))


def plot_MR(Ts, Hs, MRs, cutoff, resu_dir):
    colors = plt.cm.jet(np.linspace(0, 1, int(np.max(Ts)) + 1))

    plt.subplots(1, 1, figsize = (8,7))
    for T, MR in zip(Ts, MRs):
        plt.plot(Hs[cutoff:], MR[cutoff:], '-', color = colors[int(T)])
    
    plt.xlabel(r'$H$ [$T$]')
    plt.ylabel(r'$MR$ [%]')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(resu_dir, 'MR_vs_H.png'))


def plot_MRK(Ts, Hs, MRs, rho_xx0s, cutoff, resu_dir):
    colors = plt.cm.jet(np.linspace(0, 1, int(np.max(Ts)) + 1))

    plt.subplots(1, 1, figsize = (8,7))
    for T, MR, rho_xx0 in zip(Ts, MRs, rho_xx0s):
        plt.plot(Hs[cutoff:]/rho_xx0, MR[cutoff:], '-', color = colors[int(T)])
    
    plt.xlabel(r'$H/\rho_{0}$ [$T\cdot\Omega^{-1}\cdot m^{-1}$]')
    plt.ylabel(r'$MR$ [%]')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(resu_dir, 'MR_vs_H_rho0.png'))


def MR_shift(xs0, ys0, xs, ys):
    ymin = max(np.min(ys0), np.min(ys))
    ymax = min(np.max(ys0), np.max(ys))
    mask = (ys < ymax) * (ys > ymin)
    mask0 = (ys0 < ymax) * (ys0 > ymin)
    x_interp = PchipInterpolator(*resolve_monotone(ys[mask > 0], xs[mask > 0]), extrapolate = False)
    return x_interp(ys0[mask0 > 0]) / xs0[mask0 > 0]


def plot_MREK(Ts, Hs, MRs, rho_xx0s, cutoff, resu_dir):
    colors = plt.cm.jet(np.linspace(0, 1, int(np.max(Ts)) + 1))

    plt.subplots(1, 1, figsize = (8,7))
    nTs = []
    for T, MR, rho_xx0 in zip(Ts, MRs, rho_xx0s):
        rho0nTs = MR_shift(Hs[cutoff:], MRs[-1][cutoff:], Hs[cutoff:], MR[cutoff:])
        rho0nT = np.mean(rho0nTs[~np.isnan(rho0nTs)])
        plt.plot(Hs[cutoff:]/rho0nT, MR[cutoff:], '-', color = colors[int(T)])
        nTs.append(rho0nT/rho_xx0)
    plt.xlabel(r'$H/\rho_{0}n_T$ [$T\cdot\Omega^{-1}\cdot m^{-1}$]')
    plt.ylabel(r'$MR$ [%]')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(resu_dir, 'MR_vs_H_rho0nT.png'))

    plt.subplots(1, 1, figsize = (8,7))
    plt.plot(Ts, nTs)
    plt.xlabel(r'$T$ [K]')
    plt.ylabel(r'$n_T$ []')
    plt.savefig(os.path.join(resu_dir, 'nT_vs_T.png'))