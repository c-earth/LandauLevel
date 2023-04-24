import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import PchipInterpolator
from extract import resolve_monotone

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