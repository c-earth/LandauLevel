import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ptick

from scipy.optimize import curve_fit
from util.data import running_average


def poly(x, *ps):
    out = 0
    for i in range(len(ps)):
        out += ps[i] * x ** i
    return out


def dpoly(x, *ps):
    out = 0
    for i in range(len(ps)):
        if i == 0:
            continue
        else:
            out += i * ps[i] * x ** (i - 1)
    return out


def pieces_poly(xm, *bspss):
    x = xm[:-1]
    m = int(xm[-1])
    bs = np.sort(np.concatenate([[0, np.inf], bspss[:m-1]]))
    pss = np.array(bspss[m-1:]).reshape((m, -1))

    fnb = 0
    dfnb = 0
    out = np.zeros(x.shape)

    for i, (bi, ps, bf) in enumerate(zip(bs[:-1], pss, bs[1:])):
        if i == 0:
            out += (bi <= x) * (x < bf) * poly(x-bi, *ps)
            if bf != np.inf:
                fnb = poly(bf-bi, *ps)
                dfnb = dpoly(bf-bi, *ps)
        else:
            out += (bi <= x) * (x < bf) * poly(x-bi, *np.concatenate([[fnb, dfnb], ps[2:]]))
            if bf != np.inf:
                fnb = poly(bf-bi, *np.concatenate([[fnb, dfnb], ps[2:]]))
                dfnb = dpoly(bf-bi, *np.concatenate([[fnb, dfnb], ps[2:]]))
    return out


def subbg_po(Ts, Bs, MRs, po_power, T_max, resu_dir):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.3, 0.9, int(T_max) + 1))
    Bs_out = np.copy(Bs)
    Ts_out = []
    MRs_out = []

    for T, MR in zip(Ts, MRs):
        if T > T_max:
            break
        ps, _ = curve_fit(poly, Bs, MR, p0 = np.ones(po_power + 1), sigma = np.sqrt(MR + 1))
        bg = poly(Bs, *ps)
        Ts_out.append(T)
        MRs_out.append(MR - bg)

        f, ax = plt.subplots(2, 1, figsize = (8,7))

        ax[0].plot(Bs, MR, '.', color = colors[int(T)], linewidth = 3, label = f'{T} K')
        ax[0].plot(Bs, bg, ':', color = 'k', linewidth = 3, label = 'background')
        ax[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
        ax[0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
        ax[0].set_xlabel(r'$B$ [T]', fontsize = 20)
        ax[0].yaxis.offsetText.set_visible(False)
        ax[0].figure.canvas.draw()
        ax[0].set_ylabel(r'$MR$'+ f'[{ax[0].yaxis.get_major_formatter().get_offset()} %]', fontsize = 20)
        ax[0].legend(fontsize = 20)

        ax[1].plot(Bs, MR - bg, '-', color = colors[int(T)], linewidth = 3, label = f'{T} K')
        ax[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
        ax[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
        ax[1].set_xlabel(r'$B$ [T]', fontsize = 20)
        ax[1].yaxis.offsetText.set_visible(False)
        ax[1].figure.canvas.draw()
        ax[1].set_ylabel(r'$\Delta MR$'+ f'[{ax[1].yaxis.get_major_formatter().get_offset()} %]', fontsize = 20)
        ax[1].legend(fontsize = 20)

        f.subplots_adjust(hspace = 0)
        f.savefig(os.path.join(resu_dir, f'subbg_po_{T}K.png'))
        plt.close()
    return np.array(Ts_out), Bs_out, np.stack(MRs_out)


def subbg_pp(Ts, Bs, MRs, pp_power, pieces, T_max, resu_dir):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.3, 0.9, int(T_max) + 1))
    Bs_out = np.copy(Bs)
    Ts_out = []
    MRs_out = []

    for T, MR in zip(Ts, MRs):
        if T > T_max:
            break
        bounds = [[1] * (pieces - 1) + [-np.inf] * (pieces * (pp_power + 1)), [8] * (pieces - 1) + [np.inf] * (pieces * (pp_power + 1))]
        bspss, _ = curve_fit(pieces_poly, np.concatenate([Bs, [pieces]]), MR, p0 = 4*np.ones(pieces - 1 + pieces * (pp_power + 1)), bounds = bounds, maxfev = 10000, sigma = np.sqrt(MR + 1))
        bg = pieces_poly(np.concatenate([Bs, [pieces]]), *bspss)
        Ts_out.append(T)
        MRs_out.append(MR - bg)

        f, ax = plt.subplots(2, 1, figsize = (8,7))

        ax[0].plot(Bs, MR, '-', color = colors[int(T)], linewidth = 3, label = f'{T} K')
        ax[0].plot(Bs, bg, ':', color = 'k', linewidth = 3, label = 'background')
        ax[0].vlines(bspss[:(pieces - 1)], 0, np.max(MR), linestyle = '--', color = 'k')
        ax[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
        ax[0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
        ax[0].set_xlabel(r'$B$ [T]', fontsize = 20)
        ax[0].yaxis.offsetText.set_visible(False)
        ax[0].figure.canvas.draw()
        ax[0].set_ylabel(r'$MR$'+ f'[{ax[0].yaxis.get_major_formatter().get_offset()} %]', fontsize = 20)
        ax[0].legend(fontsize = 20)
        

        ax[1].plot(Bs, MR - bg, '-', color = colors[int(T)], linewidth = 3, label = f'{T} K')
        ax[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
        ax[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
        ax[1].set_xlabel(r'$B$ [T]', fontsize = 20)
        ax[1].yaxis.offsetText.set_visible(False)
        ax[1].figure.canvas.draw()
        ax[1].set_ylabel(r'$\Delta MR$'+ f'[{ax[1].yaxis.get_major_formatter().get_offset()} %]', fontsize = 20)
        ax[1].legend(fontsize = 20)

        f.subplots_adjust(hspace = 0)
        f.savefig(os.path.join(resu_dir, f'subbg_pp_{T}K.png'))
        plt.close()
    return np.array(Ts_out), Bs_out, np.stack(MRs_out)


def subbg_de(Ts, Bs, MRs, avg_window, T_max, resu_dir):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.3, 0.9, int(T_max)))
    Ts_out = []
    MRs_out = []

    for T, MR in zip(Ts, MRs):
        if T > T_max:
            break
        Bs_out = np.copy(Bs)
        MR_out = np.copy(MR)
        Bs_out, MR_out = running_average(Bs_out, MR_out, avg_window)

        MR_out = (MR_out[2:] + MR_out[:-2] - 2 * MR_out[1:-1])/(Bs_out[2:]/2 - Bs_out[:-2]/2) ** 2
        Bs_out = Bs_out[1:-1]
        Ts_out.append(T)
        MRs_out.append(MR_out)

        f, ax = plt.subplots(2, 1, figsize = (8,7))

        ax[0].plot(Bs, MR, '-', color = colors[int(T)], linewidth = 3, label = f'{T} K')
        ax[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
        ax[0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
        ax[0].set_xlabel(r'$B$ [T]', fontsize = 20)
        ax[0].yaxis.offsetText.set_visible(False)
        ax[0].figure.canvas.draw()
        ax[0].set_ylabel(r'$MR$'+ f'[{ax[0].yaxis.get_major_formatter().get_offset()} %]', fontsize = 20)
        ax[0].legend(fontsize = 20)

        ax[1].plot(Bs_out, MR_out, '-', color = colors[int(T)], linewidth = 3, label = f'{T} K')
        ax[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
        ax[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
        ax[1].set_xlabel(r'$B$ [T]', fontsize = 20)
        ax[1].yaxis.offsetText.set_visible(False)
        ax[1].figure.canvas.draw()
        ax[1].set_ylabel(r'$\partial^2 MR/\partial B^2$'+ f'[{ax[1].yaxis.get_major_formatter().get_offset()} %]', fontsize = 20)
        ax[1].legend(fontsize = 20)
        
        f.subplots_adjust(hspace = 0)
        f.savefig(os.path.join(resu_dir, f'subbg_de_{T}K.png'))
        plt.close()
    return np.array(Ts_out), Bs_out, np.stack(MRs_out)