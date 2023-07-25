import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.interpolate import PchipInterpolator
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit
from util.data import resolve_monotone


def FFT(Ts_sub, Hs_sub, MRs_sub, T_max, q_min, q_max, subbg, resu_dir):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.3, 0.9, int(T_max) + 1))
    iHs_sub = 1/Hs_sub
    iH = np.linspace(np.min(iHs_sub), np.max(iHs_sub), 10000)
    MRs_iH = []
    qs = []
    MRs_iH_fft = []

    
    for T, MR in zip(Ts_sub, MRs_sub):
        f, ax = plt.subplots(2, 1, figsize = (8,7))
        MR_iH = PchipInterpolator(*resolve_monotone(iHs_sub, MR), extrapolate = False)
        MR_iH = MR_iH(iH)

        MR_iH_fft = np.fft.rfft(MR_iH)
        sample_rate = len(iH)/(np.max(iH)-np.min(iH))
        q = np.fft.rfftfreq(len(iH), d = 1./sample_rate)
        MRs_iH.append(MR_iH)
        qs.append(q)
        MRs_iH_fft.append(MR_iH_fft)

        ax[0].plot(iH, MR_iH, '-', color = colors[int(T)], linewidth = 3, label = f'{T} K')
        ax[1].plot(q, np.abs(MR_iH_fft), '-', color = colors[int(T)], linewidth = 3, label = f'{T} K')
        
        ax[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
        ax[0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
        ax[0].set_xlabel(r'$B^{-1}$ [T$^{-1}$]', fontsize = 20)
        ax[0].set_ylabel(r'$\Delta MR$ [%]', fontsize = 20)
        ax[0].yaxis.get_offset_text().set_size(20)
        ax[0].set_xscale('log')
        ax[0].legend(fontsize = 20, loc = 'upper right')

        ax[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
        ax[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
        ax[1].set_xlabel(r'$q$ [T]', fontsize = 20)
        ax[1].set_ylabel(r'$\Delta MR$ [%]', fontsize = 20)
        ax[1].yaxis.get_offset_text().set_size(20)
        ax[1].set_ylim((0, 1.2 * np.max(np.abs(np.stack(MRs_iH_fft))[:, q > 1])))
        ax[1].set_xlim((q_min, q_max))
        ax[1].legend(fontsize = 20, loc = 'upper right')

        f.tight_layout()
        f.savefig(os.path.join(resu_dir, f'FFT_{subbg}_{T}K.png'))
        plt.close()
        
    return iH, np.stack(MRs_iH), np.stack(qs), np.stack(MRs_iH_fft)


def separate(p):
    args = int(len(p)/3.)
    A = p[:args]
    X0 = p[args:2*args]
    S = p[2*args:]
    return A, X0, S


def lorentz(x, *p):
    A, X0, S = separate(p)
    y = np.zeros(len(x))
    for a,x0,s in zip(A, X0, S):
        y += a/(1+(2.*(x0-x)/s)**2)
    return y


def FFT_peaks(q, MR_iH_fft, r_min, r_max, T_max, resu_dir, T, subbg):
    
    MR_iH_fft = MR_iH_fft[(q >= r_min)*(q <= r_max)]
    q = q[(q >= r_min)*(q <= r_max)]
    

    yA = np.abs(MR_iH_fft).astype(np.float64)
    ymax = np.max(yA)

    maxinds = argrelextrema(yA, np.greater)[0]

    ypeaks = yA[maxinds]
    maxinds = maxinds[ypeaks/ymax > 0.2]
    ypeaks = yA[maxinds]
    qpeaks = q[maxinds]

    p0 = np.concatenate([ypeaks, qpeaks, np.sqrt(np.gradient(qpeaks))])
    numb = len(ypeaks)
    bounds = [[0]*3*numb, [np.max(np.abs(MR_iH_fft))]*numb+[np.max(q)]*numb+[np.inf]*numb]
    p, _ = curve_fit(lorentz, q, yA, p0, bounds = bounds, maxfev = 50000)
    y_amps, y_poss, y_widthes = separate(p)
    accept = np.argsort(y_amps)[-3:]
    y_amps = y_amps[accept]
    y_poss = y_poss[accept]
    y_widthes = y_widthes[accept]

    colors = mpl.colormaps['gnuplot'](np.linspace(0.3, 0.9, int(T_max) + 1))
    f, ax = plt.subplots(1, 1, figsize = (8,7))
    ax.plot(q, np.abs(MR_iH_fft), '-', color = colors[int(T)], linewidth = 3, label = f'{T} K')
    for y_amp, y_pos, y_width in zip(y_amps, y_poss, y_widthes):
        ax.plot(q, lorentz(q, y_amp, y_pos, y_width), ':', color = 'k', linewidth = 3)
    ax.tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    ax.ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
    ax.set_xlabel(r'$q$ [T]', fontsize = 20)
    ax.set_ylabel(r'$\Delta MR$ [%]', fontsize = 20)
    ax.yaxis.get_offset_text().set_size(20)
    ax.legend(fontsize = 20, loc = 'upper right')
    
    f.savefig(os.path.join(resu_dir, f'peaks_{subbg}_{T}K.png'))
    plt.close()
    return y_poss, y_widthes

def signal_filter(iHs, q, MR_iH_fft, qpeaks, y_widthes, T_max, resu_dir, T, subbg):
    colors = mpl.colormaps['gnuplot'](np.linspace(0.3, 0.9, int(T_max) + 1))
    f, ax = plt.subplots(2, 1, figsize = (8, 7), gridspec_kw = {'height_ratios':[3,5]})

    ixplot_max = 1.4
    numb = len(qpeaks)
    idxs = np.argsort(qpeaks)
    for i, (qpeak, y_width) in enumerate(zip(qpeaks[idxs], y_widthes[idxs])):
        yfft_copy = np.copy(MR_iH_fft)
        yfft_copy *= lorentz(q, 2/y_width/np.pi, qpeak, y_width)
        yfilter = np.fft.irfft(yfft_copy)

        num_osc = int(qpeak/2) + 2
        maxinds = argrelextrema(yfilter, np.greater)[0][0:num_osc]
        mininds = argrelextrema(yfilter, np.less)[0][0:num_osc]
        slope = qpeak

        n_max0 = np.round(slope*(iHs[maxinds[0]]))
        n_max = np.arange(n_max0,n_max0+len(mininds),1)
        if iHs[mininds[0]]<iHs[maxinds[0]]:
            n_min0 = n_max0-0.5
        else:
            n_min0 = n_max0+0.5
        n_min = np.arange(n_min0, n_min0+len(mininds), 1)
        ixn = np.concatenate([iHs[maxinds], iHs[mininds]])
        n = np.concatenate([n_max,n_min])

        ixn_plot = np.linspace(0,ixplot_max,50)
        p = np.polyfit(ixn,n,1)
        yplot = np.polyval(p,ixn_plot)
        print('slope: ',p[0],' intercept: ', p[1])

        color = np.concatenate([colors[int(T)][:3], np.array([(i + 1)/numb])])

        ax[0].plot(iHs, yfilter, linewidth = 3, color = color)
        ax[0].scatter(iHs[mininds], yfilter[mininds], s = 50, facecolor = color, edgecolors = color, linewidth = 3, zorder = 3)
        ax[0].scatter(iHs[maxinds], yfilter[maxinds], facecolor = 'w', edgecolors = color, s = 60, linewidth = 3, zorder = 3)

        ax[1].plot(ixn_plot, yplot, linestyle = '--', linewidth = 3, color = color, label = f'{round(qpeak, 2)} T')
        ax[1].scatter(iHs[mininds], n_min, s = 50, facecolor = color, edgecolors = color, linewidth = 3, zorder = 3)
        ax[1].scatter(iHs[maxinds], n_max, s = 60, facecolor = 'w', edgecolors = color, linewidth = 3, zorder = 3)

    ax[0].set_ylabel(r'$\mathrm{\mathsf{\Delta}}$MR (%)', fontsize = 20)
    ax[0].set_xlim([0, ixplot_max])
    ax[0].set_ylim([3*np.min(yfilter),3*np.max(yfilter)])
    ax[0].fill_between([0, np.min(iHs)], 3*np.min(yfilter),3*np.max(yfilter), color = 'grey', alpha = 0.5)
    ax[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax[0].locator_params(axis='y', nbins=5)
    ax[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    ax[0].yaxis.get_offset_text().set_size(20)

    ax[1].set_xlabel(r'1/B [T$^{-1}$]', fontsize = 20)
    ax[1].set_xlim([0, ixplot_max])
    ax[1].set_ylim([-0.5, 15.5])
    ax[1].fill_between([0, np.min(iHs)], -0.5, 15.5, color = 'grey', alpha = 0.5)
    ax[1].set_ylabel('Landau level', fontsize = 20)
    ax[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 20)
    ax[1].yaxis.get_offset_text().set_size(20)
    ax[1].legend(title = f'{T} K', title_fontsize = 20, fontsize = 20, loc = 'upper right')

    f.tight_layout()
    f.savefig(os.path.join(resu_dir, f'SdH_{subbg}_{T}K.png'), bbox_inches = 'tight', dpi = 400)
    plt.close()