import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import Divider, Size

from scipy.interpolate import PchipInterpolator
from util.data import resolve_monotone

q = 1.602E-19

def read_params(filename):
    with open(filename, 'r') as f:
        params = []
        for line in f.readlines()[1:]:
            params.append([float(x) for x in line.split('\t')])
    return np.array(params)

def change_axes_size(sizes, fig, axs):
    oldw, oldh = fig.get_size_inches()
    l = fig.subplotpars.left
    r = fig.subplotpars.right
    t = fig.subplotpars.top
    b = fig.subplotpars.bottom
    h = fig.subplotpars.hspace
    w = fig.subplotpars.vspace



def fix_axes_size_incm(axew, axeh, axs = None, axc = None):
    if not axs:
        fig = plt.gcf()
        ax = plt.gca()
    else:
        fig = axs.figure
        ax = axs
    #obtain the current ratio values for padding and fix size
    oldw, oldh = fig.get_size_inches()
    print(oldw, oldh)
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    h = ax.figure.subplotpars.hspace
    w = ax.figure.subplotpars.wspace
    print(l, r, t, b, h, w)

    #work out what the new  ratio values for padding are, and the new fig size.
    neww = axew+oldw*(1-r+l+w) + 0.24
    newh = axeh+oldh*(1-t+b)
    newr = neww - (1-r)*oldw
    newl = l*oldw
    newt = newh - (1-t)*oldh
    newb = b*oldh

    #right(top) padding, fixed axes size, left(bottom) pading
    hori = [Size.Fixed(newl), Size.Fixed(axew), Size.Fixed(w*oldw), Size.Fixed(0.24), Size.Fixed(newr)]
    vert = [Size.Fixed(newb), Size.Fixed(axeh), Size.Fixed(newt)]

    divider = Divider(fig, (0.0, 0.0, 1., 1.), hori, vert, aspect=False)
    # the width and height of the rectangle is ignored.

    ax.set_axes_locator(divider.new_locator(nx=1, ny=1))
    axc.set_axes_locator(divider.new_locator(nx=3, ny=1))
    print(neww, newh)
    #we need to resize the figure now, as we have may have made our axes bigger than in.
    fig.set_size_inches(neww,newh)

def plot_rho(Ts, Bs, rho_xxs, cutoff, resu_dir):
    base = 1000
    colors = mpl.colormaps['gnuplot'](np.log(np.linspace(base**(0.1), base**(0.9), int(np.max(Ts))))/np.log(base))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, (axs, axc) = plt.subplots(1, 2, figsize = (8,7), gridspec_kw={"width_ratios":[0.97, 0.03]})
    print(axs.figure.subplotpars.right, axc.figure.subplotpars.right)
    for T, rho_xx in zip(np.flip(Ts, axis = 0), np.flip(rho_xxs, axis = 0)):
        axs.plot(Bs[cutoff:], rho_xx[cutoff:], '-', color = colors[int(T)-2], linewidth = 3)

    axs.set_xlabel(r'$B$ [T]', fontsize = 20)
    axs.set_ylabel(r'$\rho_{xx}$ [$\Omega$m]', fontsize = 20)
    axs.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    axs.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
    axs.set_xlim((np.min(Bs[cutoff:]), np.max(Bs[cutoff:])))
    axs.yaxis.get_offset_text().set_size(20)

    cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 2, vmax = np.max(Ts)+2)), cax = axc)
    cb.ax.set_title(r'$T$ (K)', fontsize = 20)
    cb.ax.tick_params(length = 5, width = 1.5, labelsize = 20, which = 'both', direction = 'out')
    cb.ax.set_yscale('log')
    cb.ax.set_yticks([10, 100])
    cb.ax.set_yticklabels([10, 100])

    f.tight_layout()
    # fix_axes_size_incm(8, 7, axs, axc)
    # fix_axes_size_incm(0.24, 7, axc)
    f.savefig(os.path.join(resu_dir, 'rho_vs_B.png'))
    plt.close()


def plot_MR(Ts, Bs, MRs, cutoff, resu_dir):
    base = 1000
    colors = mpl.colormaps['gnuplot'](np.log(np.linspace(base**(0.1), base**(0.9), int(np.max(Ts))))/np.log(base))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, (axs, axc) = plt.subplots(1, 2, figsize = (8,7), gridspec_kw={"width_ratios":[0.97, 0.03]})
    for T, MR in zip(np.flip(Ts, axis = 0), np.flip(MRs, axis = 0)):
        axs.plot(Bs[cutoff:], MR[cutoff:], '-', color = colors[int(T)-2], linewidth = 3)
    axs.set_xlabel(r'$B$ [T]', fontsize = 20)
    axs.set_ylabel(r'$MR$ [%]', fontsize = 20)
    axs.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    axs.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
    axs.set_xlim((np.min(Bs[cutoff:]), np.max(Bs[cutoff:])))
    axs.yaxis.get_offset_text().set_size(20)

    cb = f.colorbar(plt.cm.ScalarMappable(cmap = cmap, norm = plt.Normalize(vmin = 2, vmax = np.max(Ts)+2)), cax = axc)
    cb.ax.set_title(r'$T$ (K)', fontsize = 20)
    cb.ax.tick_params(length = 5, width = 1.5, labelsize = 20, which = 'both', direction = 'out')
    cb.ax.set_yscale('log')
    cb.ax.set_yticks([10, 100])
    cb.ax.set_yticklabels([10, 100])

    f.tight_layout()
    fix_axes_size_incm(8, 7, axs, axc)

    f.savefig(os.path.join(resu_dir, 'MR_vs_B.png'))
    plt.close()


def plot_MRK(Ts, Bs, MRs, rho_xx0s, cutoff, resu_dir):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.1, 0.9, int(np.max(Ts)) + 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, axs = plt.subplots(1, 1, figsize = (8,7))
    for T, MR, rho_xx0 in zip(Ts, MRs, rho_xx0s):
        plt.plot(Bs[cutoff:]/rho_xx0, MR[cutoff:], '-', color = colors[int(T)], linewidth = 3)
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
    f.savefig(os.path.join(resu_dir, 'MR_vs_B_rho0.png'))
    plt.close()


def MR_shift(xs0, ys0, xs, ys):
    ymin = max(np.min(ys0), np.min(ys))
    ymax = min(np.max(ys0), np.max(ys))
    mask = (ys < ymax) * (ys > ymin)
    mask0 = (ys0 < ymax) * (ys0 > ymin)
    x_interp = PchipInterpolator(*resolve_monotone(ys[mask > 0], xs[mask > 0]), extrapolate = False)
    return x_interp(ys0[mask0 > 0]) / xs0[mask0 > 0]


def plot_MREK(Ts, Bs, MRs, rho_xx0s, cutoff, resu_dir, params_file = None):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.1, 0.9, int(np.max(Ts)) + 1))
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_gnuplot', colors)

    f, axs = plt.subplots(1, 1, figsize = (8,7))
    nTs = []
    for T, MR, rho_xx0 in zip(Ts, MRs, rho_xx0s):
        rho0nTs = MR_shift(Bs[cutoff:], MRs[-1][cutoff:], Bs[cutoff:], MR[cutoff:])*rho_xx0s[-1]
        rho0nT = np.mean(rho0nTs[~np.isnan(rho0nTs)])
        nTs.append(rho0nT/rho_xx0)
        plt.plot(Bs[cutoff:]/rho0nT, MR[cutoff:], '-', color = colors[int(T)], linewidth = 3)
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
    f.savefig(os.path.join(resu_dir, 'MR_vs_B_rho0nT.png'))
    plt.close()

    f, axs = plt.subplots(1, 1, figsize = (8,7))
    plt.plot(Ts, nTs, 'r-o', linewidth = 3,  markersize=10, label = 'Extended Kohler')
    axs.set_xlabel(r'$T$ [K]', fontsize = 20)
    axs.set_ylabel(r'$n_T$ [normalized at 300K]', fontsize = 20)
    axs.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    axs.yaxis.get_offset_text().set_size(20)
    plt.legend(fontsize = 20)

    f.tight_layout()
    f.savefig(os.path.join(resu_dir, 'nT_vs_T_ExtendedKohler.png'))
    plt.close()

    if params_file:
        # from two-band model
        params = read_params(params_file)
        ts = params[:, 0]
        uhs = params[:, 1]
        nhs = params[:, 3]
        ues = params[:, 5]
        nes = params[:, 7]
        nts = q * (uhs * nhs + ues * nes) ** (3/2) / (uhs ** 3 * nhs + ues ** 3 *nes) ** (1/2)

        f, axs = plt.subplots(1, 1, figsize = (8,7))
        plt.plot(ts, nts, 'b-o', linewidth = 3,  markersize=10, label = 'Two Band')
        axs.set_xlabel(r'$T$ [K]', fontsize = 20)
        axs.set_ylabel(r'$n_T$ [Cm$^{-3}$]', fontsize = 20)
        axs.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
        axs.yaxis.get_offset_text().set_size(20)
        plt.legend(fontsize = 20)

        f.tight_layout()
        f.savefig(os.path.join(resu_dir, 'nT_vs_T_TwoBand.png'))
        plt.close()

        f, axs = plt.subplots(1, 1, figsize = (8,7))
        plt.plot(Ts, nTs, 'r-o', linewidth = 3,  markersize=10, label = 'Extended Kohler')
        plt.plot(ts, nts/nts[-1], 'b-o', linewidth = 3,  markersize=10, label = 'Two Band')
        axs.set_xlabel(r'$T$ [K]', fontsize = 20)
        axs.set_ylabel(r'$n_T$ [normalized at 300K]', fontsize = 20)
        axs.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
        axs.yaxis.get_offset_text().set_size(20)
        plt.legend(fontsize = 20)

        f.tight_layout()
        f.savefig(os.path.join(resu_dir, 'nT_vs_T_compare.png'))
        plt.close()