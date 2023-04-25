import os
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit


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
            fnb = poly(bf-bi, *ps)
            dfnb = dpoly(bf-bi, *ps)
        else:
            out += (bi <= x) * (x < bf) * poly(x-bi, *np.concatenate([[fnb, dfnb], ps[2:]]))
            fnb = poly(bf-bi, *np.concatenate([[fnb, dfnb], ps[2:]]))
            dfnb = dpoly(bf-bi, *np.concatenate([[fnb, dfnb], ps[2:]]))
    
    return out


def subbg_poly(Ts, Hs, MRs, po_power, T_max, resu_dir):
    colors = plt.cm.jet(np.linspace(0, 1, T_max + 1))
    Hs_out = np.copy(Hs)
    MRs_out = []

    for T, MR in zip(Ts, MRs):
        if T > T_max:
            break
        ps, _ = curve_fit(poly, Hs, MR, p0 = np.ones(po_power + 1), sigma = np.sqrt(MR + 1))
        bg = poly(Hs, *ps)
        MRs_out.append(MR - bg)

        f, ax = plt.subplots(2, 1, figsize = (12,7))

        ax[0].plot(Hs, MR, '-', color = colors[int(T)])
        ax[0].plot(Hs, bg, '--', color = 'k')
        ax[0].set_xlabel(r'$H$ [$T$]')
        ax[0].set_ylabel(r'$MR$ [%]')

        ax[1].plot(Hs, MR - bg, '-', color = colors[int(T)])
        ax[1].set_xlabel(r'$H$ [$T$]')
        ax[1].set_ylabel(r'$\Delta MR$ [%]')

        plt.savefig(os.path.join(resu_dir, f'subbg_poly_{T}K.png'))
    return Hs_out, np.stack(MRs_out)


def subbg_pieces_poly(Ts, Hs, MRs, pp_power, pieces, T_max, resu_dir):
    colors = plt.cm.jet(np.linspace(0, 1, T_max + 1))
    Hs_out = np.copy(Hs)
    MRs_out = []

    for T, MR in zip(Ts, MRs):
        if T > T_max:
            break
        bspss, _ = curve_fit(pieces_poly, np.concatenate([Hs, [pieces]]), MR, p0 = np.ones(pieces - 1 + pieces * (pp_power + 1)), sigma = np.sqrt(MR + 1), maxfev = 10000)
        bg = pieces_poly(np.concatenate([Hs, [pieces]]), *bspss)
        MRs_out.append(MR - bg)

        f, ax = plt.subplots(2, 1, figsize = (12,7))

        ax[0].plot(Hs, MR, '-', color = colors[int(T)])
        ax[0].plot(Hs, bg, '--', color = 'k')
        ax[0].set_xlabel(r'$H$ [$T$]')
        ax[0].set_ylabel(r'$MR$ [%]')

        ax[1].plot(Hs, MR - bg, '-', color = colors[int(T)])
        ax[1].set_xlabel(r'$H$ [$T$]')
        ax[1].set_ylabel(r'$\Delta MR$ [%]')

        plt.savefig(os.path.join(resu_dir, f'subbg_pieces_poly_{T}K.png'))
    return Hs_out, np.stack(MRs_out)


def subbg_derivative(Ts, Hs, MRs, de_power, T_max, resu_dir):
    colors = plt.cm.jet(np.linspace(0, 1, T_max + 1))
    MRs_out = []

    for T, MR in zip(Ts, MRs):
        if T > T_max:
            break
        Hs_out = np.copy(Hs)
        MR_out = np.copy(MR)
        for i in range(1, de_power + 1):
            MR_out = (MR_out[1:] - MR_out[:-1]) / (Hs_out[1:] - Hs_out[:-1])
            Hs_out = (Hs_out[1:] + Hs_out[:-1])/2
        MRs_out.append(MR)

        f, ax = plt.subplots(2, 1, figsize = (12,7))

        ax[0].plot(Hs, MR, '-', color = colors[int(T)])
        ax[0].set_xlabel(r'$H$ [$T$]')
        ax[0].set_ylabel(r'$MR$ [%]')

        ax[1].plot(Hs_out, MR_out, '-', color = colors[int(T)])
        ax[1].set_xlabel(r'$H$ [$T$]')
        ax[1].set_ylabel(r'$\Delta MR$ [%]')

        plt.savefig(os.path.join(resu_dir, f'subbg_derivative_{T}K.png'))
    return Hs_out, np.stack(MRs_out)