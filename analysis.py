import os
import pickle as pkl
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from extract import resolve_monotone

data_file = 'D:/python_project/LandauLevel/data/MR_S4_extracted'
resu_dir = 'D:/python_project/LandauLevel/resu/MR_S4/'

with open(data_file, 'rb') as f:
    data = pkl.load(f)
temperatures = np.sort(list(data.keys()))
idx_min = 0
idx_max = len(data[temperatures[0]][0]) - 1
for temperature in temperatures:
    rho_xx = data[temperature][1]
    nans = np.isnan(rho_xx)
    got_min = False
    got_max = False
    for i, nan in enumerate(nans):
        if not got_min:
            if not nan:
                idx_min = max(idx_min, i)
                got_min = True
        else:
            if not got_max:
                if nan:
                    idx_max = min(idx_max, i - 1)
                    got_max = True
                elif i == len(nans) - 1:
                    idx_max = min(idx_max, i)

for temperature in temperatures:
    data[temperature][0] = data[temperature][0][idx_min: idx_max + 1]
    data[temperature][1] = data[temperature][1][idx_min: idx_max + 1]


def plot_rho(data, temperatures, co):
    colors = plt.cm.jet(np.linspace(0, 1, int(max(temperatures)) + 1))

    plt.figure()
    for temperature in temperatures:
        plt.plot(data[temperature][0][co:], data[temperature][1][co:], '-', color = colors[int(temperature)])
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def plot_MR(data, temperatures, co):
    colors = plt.cm.jet(np.linspace(0, 1, int(max(temperatures)) + 1))

    plt.figure()
    for temperature in temperatures:
        plt.plot(data[temperature][0][co:], 100*(data[temperature][1][co:]/data[temperature][1][0]-1), '-', color = colors[int(temperature)])
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def plot_MRK(data, temperatures, co):
    colors = plt.cm.jet(np.linspace(0, 1, int(max(temperatures)) + 1))

    plt.figure()
    for temperature in temperatures:
        plt.plot(data[temperature][0][co:]/data[temperature][1][0], 100*(data[temperature][1][co:]/data[temperature][1][0]-1), '-', color = colors[int(temperature)])
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

def plot_MREK(data, temperatures, co):
    colors = plt.cm.jet(np.linspace(0, 1, int(max(temperatures)) + 1))

    plt.figure()
    nTs = []
    for temperature in temperatures:
        rho0nTs = MR_shift(data[temperatures[-1]][0][co:], 100*(data[temperatures[-1]][1][co:]/data[temperatures[-1]][1][0]-1), data[temperature][0][co:], 100*(data[temperature][1][co:]/data[temperature][1][0]-1))
        rho0nT = np.mean(rho0nTs[~np.isnan(rho0nTs)])
        plt.plot(data[temperature][0][co:]/rho0nT, 100*(data[temperature][1][co:]/data[temperature][1][0]-1), '-', color = colors[int(temperature)])
        nTs.append(rho0nT/data[temperature][1][0])
    plt.yscale('log')
    plt.xscale('log')
    plt.show()
    plt.figure()
    plt.plot(temperatures, nTs)
    plt.show()

def MR_shift(xs0, ys0, xs, ys):
    ymin = max(np.min(ys0), np.min(ys))
    ymax = min(np.max(ys0), np.max(ys))
    mask = (ys < ymax) * (ys > ymin)
    mask0 = (ys0 < ymax) * (ys0 > ymin)
    x_interp = PchipInterpolator(*resolve_monotone(ys[mask > 0], xs[mask > 0]), extrapolate = False)
    return x_interp(ys0[mask0 > 0]) / xs0[mask0 > 0]

def subbg_poly(data, temperatures, n):
    colors = plt.cm.jet(np.linspace(0, 1, int(max(temperatures)) + 1))

    plt.figure()
    for temperature in temperatures:
        H = data[temperature][0]
        rho_xx = data[temperature][1]
        p, _ = curve_fit(poly, H, rho_xx, p0 = np.ones(n), sigma = np.sqrt(rho_xx + 1) - 1)
        plt.plot(H, rho_xx, '-', color = colors[int(temperature)])
        plt.plot(H, poly(H, *p), '--', color = colors[int(temperature)])     
    plt.show()

def subbg_pieces_poly(data, temperatures, m, n):
    colors = plt.cm.jet(np.linspace(0, 1, int(max(temperatures)) + 1))

    plt.figure()
    for temperature in temperatures:
        H = data[temperature][0]
        rho_xx = data[temperature][1]
        p, _ = curve_fit(pieces_poly, np.concatenate([H, [m]]), rho_xx, p0 = np.ones((n-1) * m + 1), sigma = np.sqrt(rho_xx + 1) - 1)
        plt.plot(H, rho_xx, '-', color = colors[int(temperature)])
        plt.plot(H, pieces_poly(np.concatenate([H, [m]]), *p), '--', color = colors[int(temperature)])
    plt.show()

def subbg_derivative(data, temperatures):
    return

def poly(x, *ps):
    out = 0
    for i in range(len(ps)):
        out += ps[i] * x ** i
    return out

def dpoly(x, *ps):
    out = 0
    for i in range(len(ps)):
        out += i * ps[i] * x ** (i - 1)
    return out

def pieces_poly(xp, *ps):
    x = xp[:-1]
    m = int(xp[-1])
    n = int((len(ps) - 1) / m + 1)
    bs = np.sort(ps[:m - 1])
    bs = np.concatenate([[0], bs, [np.inf]])
    fnb = 0
    dfnb = 0
    pst = ps[m - 1:]
    pss = []
    for i in range(m):
        if i == 0:
            pss.append(pst[:n])
            pst[n:]
        else:
            pss.append(pst[:n-2])
            pst[n-2:]

    for bi, ps, bf in zip(bs[:-1], pss, bs[1:]):
        if x <= bf:
            if fnb == 0:
                return poly(x-bi, *ps)
            else:
                return poly(x-bi, *np.concatenate([[fnb, dfnb], ps]))
        else:
            if fnb == 0:
                fnb = poly(bf-bi, *ps)
                dfnb = dpoly(bf-bi, *ps)
            else:
                fnb = poly(bf-bi, *np.concatenate([[fnb, dfnb], ps]))
                dfnb = dpoly(bf-bi, *np.concatenate([[fnb, dfnb], ps]))

subbg_pieces_poly(data, temperatures, 2, 2)