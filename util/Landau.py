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

    f, ax = plt.subplots(2, 1, figsize = (8,7))
    for T, MR in zip(Ts_sub, MRs_sub):
        MR_iH = PchipInterpolator(*resolve_monotone(iHs_sub, MR), extrapolate = False)
        MR_iH = MR_iH(iH)

        MR_iH_fft = np.fft.rfft(MR_iH)
        sample_rate = len(iH)/(np.max(iH)-np.min(iH))
        q = np.fft.rfftfreq(len(iH), d = 1./sample_rate)
        MRs_iH.append(MR_iH)
        qs.append(q)
        MRs_iH_fft.append(MR_iH_fft)

        ax[0].plot(iH, MR_iH, '-', color = colors[int(T)], linewidth = 3, label = f'{T} K')
        ax[1].plot(q, np.abs(MR_iH_fft), '-', color = colors[int(T)], linewidth = 3)
        
    ax[0].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 14)
    ax[0].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
    ax[0].set_xlabel(r'$H^{-1}$ [T$^{-1}$]', fontsize = 14)
    ax[0].set_ylabel(r'$\Delta MR$ [%]', fontsize = 14)
    ax[0].yaxis.get_offset_text().set_size(14)
    ax[0].set_xscale('log')
    ax[0].legend()

    ax[1].tick_params(which = 'both', direction = 'in', top = False, right = False, length = 5, width = 1.5, labelsize = 14)
    ax[1].ticklabel_format(axis = 'y', style = 'sci', scilimits = (0, 0), useMathText=True)
    ax[1].set_xlabel(r'$q$ [T]', fontsize = 14)
    ax[1].set_ylabel(r'$\Delta MR$ [%]', fontsize = 14)
    ax[1].yaxis.get_offset_text().set_size(14)
    ax[1].set_ylim((0, 1.2 * np.max(np.abs(np.stack(MRs_iH_fft))[:, q > 1])))
    ax[1].set_xlim((q_min, q_max))

    f.tight_layout()
    f.savefig(os.path.join(resu_dir, f'FFT_{subbg}.png'))
    plt.close()
    return iH, np.stack(MRs_iH), np.stack(qs), np.stack(MRs_iH_fft)


def separate(p):
    args = int(len(p)/3.)
    A = p[:args]
    X0 = p[args:2*args]
    S = p[2*args:]
    return A,X0,S


def lorentz(x,*p):
    A,X0,S = separate(p)
    y = np.zeros(len(x))
    for a,x0,s in zip(A,X0,S):
        y += a/(1+(2.*(x0-x)/s)**2)
    return y


def FFT_peaks(q, yfft, Fmin, Fmax):

    yfft = yfft[(q >= Fmin)*(q <= Fmax)]
    q = q[(q >= Fmin)*(q <= Fmax)]
    

    yA = np.abs(yfft).astype(np.float64)
    ymax = np.max(yA)

    maxinds = argrelextrema(yA,np.greater)[0]

    ypeaks = yA[maxinds]
    maxinds = maxinds[ypeaks/ymax > 0.5]
    ypeaks = yA[maxinds]
    qpeaks = q[maxinds]

    p0 = np.concatenate([ypeaks,qpeaks,np.sqrt(np.gradient(qpeaks))])
    p, _ = curve_fit(lorentz,q,yA,p0)
    y_amps,y_pos,y_width = separate(p)


    # Reject peaks that have been poorly fit (probably too broad)
    accept = np.ones(len(ypeaks)).astype(bool)
    for i in range(len(ypeaks)):
        if np.abs(qpeaks[i]-y_pos[i])>1:
            accept[i] = False
    y_amps = y_amps[accept]
    y_pos = y_pos[accept]
    y_width = y_width[accept]

    # Return parameters for windowing
    qpeaks = qpeaks[accept]
    window_size = np.sqrt(y_width)

    return qpeaks, window_size

def signal_filter(ix,y,q,yfft,qpeaks,window_size,label, T):
    # Inverse Fourier transform narrow window around each peak and generate Landau level plots
    f3,ax3 = plt.subplots(2,1,figsize=(8,7),gridspec_kw = {'height_ratios':[3,5]})

    ixplot_max = 0.7   # User can modify plotting range if desired
    for i in range(len(qpeaks)):
        yfft_copy = np.copy(yfft)
        qmin = qpeaks[i]-window_size[i]/3.
        qmax = qpeaks[i]+window_size[i]/3.
        yfft_copy[(q<qmin) | (q>qmax)]=0
        yfilter = np.fft.irfft(yfft_copy)

        num_osc = int(qpeaks[i]/2.)+2       # Cut off how many period of oscillation are plotted using a reasonable rule
        maxinds = argrelextrema(yfilter,np.greater)[0][0:num_osc]
        mininds = argrelextrema(yfilter,np.less)[0][0:num_osc]
        slope = qpeaks[i]

        # Make all maxima at integers and minima at half-integers
        n_max0 = np.round(slope*(ix[maxinds[0]]))
        n_max = np.arange(n_max0,n_max0+len(mininds),1)
        if ix[mininds[0]]<ix[maxinds[0]]:
            n_min0 = n_max0-0.5
        else:
            n_min0 = n_max0+0.5
        n_min = np.arange(n_min0,n_min0+len(mininds),1)
        ixn = np.concatenate([ix[maxinds],ix[mininds]])
        n = np.concatenate([n_max,n_min])

        ixn_plot = np.linspace(0,ixplot_max,50)
        p = np.polyfit(ixn,n,1)
        yplot = np.polyval(p,ixn_plot)
        print('slope: ',p[0],' intercept: ',p[1], ' QL (T): ',1/ixn_plot[np.argmin(np.abs(yplot-1))], ' EQL (T): ',1/ixn_plot[np.argmin(np.abs(yplot-0.5))])

        ax3[0].plot(ix,yfilter,linewidth=2)
        ax3[0].scatter(ix[mininds],yfilter[mininds],s=50,linewidth=2,zorder=3)
        ax3[0].scatter(ix[maxinds],yfilter[maxinds],facecolor='w',s=60,linewidth=2,zorder=3)
        ax3[1].plot(ixn_plot,yplot,linestyle='--',linewidth=2)
        ax3[1].scatter(ix[mininds],n_min,s=50,linewidth=2,zorder=3)
        ax3[1].scatter(ix[maxinds],n_max,s=60,facecolor='w',linewidth=2,zorder=3)

    ax3[0].set_title(label)
    ax3[0].set_ylabel(r'$\mathrm{\mathsf{\Delta}}$MR (%)')
    ax3[0].set_xlim([0,ixplot_max])
    ax3[0].set_ylim([3*np.min(yfilter),3*np.max(yfilter)])
    ax3[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax3[0].locator_params(axis='y', nbins=5)
    ax3[0].tick_params(direction='in',top=False,right=False,length=5,width=1.5)

    ax3[1].set_xlabel(r'1/B (T$^\mathrm{-1}$)')
    ax3[1].set_xlim([0,ixplot_max])
    ax3[1].set_ylim([-0.5,22.5])
    ax3[1].set_ylabel('Landau level')
    ax3[1].tick_params(direction='in',top=False,right=False,length=5,width=1.5)

    f3.tight_layout()
    f3.savefig('SdH_'+label+'_'+str(T)+'K.png',bbox_inches='tight',dpi=400)
    plt.close()