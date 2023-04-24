import numpy as np
import matplotlib as mtl
from scipy.interpolate import interp1d
import matplotlib.colors as colors
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt

xmin = 0.05 # Lower bound on field to consider (quality of background subtraction may affect this number)
xmax = 9.   # Upper bound on field to consider

Fmin = 1.5  # Lower bound on frequency to cut off tail of 0 frequency component (user should change this accordingly)
Fmax = 70   # Upper bound on frequency to exclude small high frequency components

f1,ax1 = plt.subplots(1,1,figsize=(8,7))
f2,ax2 = plt.subplots(1,1,figsize=(8,7))

def truncate_colormap(cmap, minval = 0., maxval = 1., n = 100):
    new_cmap = colors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n = cmap.name, a = minval, b = maxval), cmap(np.linspace(minval, maxval, n)))
    return new_cmap

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

def FFT(x,y):
    ix = 1./x
    f = interp1d(ix, y, kind = 'linear', fill_value = 'extrapolate')
    ix = np.linspace(np.min(ix), np.max(ix), 10000)
    y = f(ix)

    yfft = np.fft.rfft(y)
    sample_rate = len(ix)/(np.max(ix)-np.min(ix))
    q = np.fft.rfftfreq(len(y), d = 1./sample_rate)

    return ix, y, q, yfft

def FFT_peaks(q, yfft, color, label):
    i0 = int(Fmin/np.max(q)*len(q))
    q = q[i0:]
    yfft = yfft[i0:]

    yA = np.abs(yfft).astype(np.float64)
    ymax = np.max(yA)

    maxinds = argrelextrema(yA,np.greater)[0]

    ypeaks = yA[maxinds]
    maxinds = maxinds[ypeaks/ymax > 0.5]
    ypeaks = yA[maxinds]
    qpeaks = q[maxinds]

    p0 = np.concatenate([ypeaks,qpeaks,np.sqrt(np.gradient(qpeaks))])
    p,cov = curve_fit(lorentz,q,yA,p0)
    y_amps,y_pos,y_width = separate(p)

    ax2.plot(q,yA,'-',color=color,linewidth=2,label=label)
    ax2.scatter(qpeaks,ypeaks,color=color,linewidth=2)

    # Reject peaks that have been poorly fit (probably too broad)
    accept = np.ones(len(ypeaks)).astype(np.bool)
    for i in range(len(ypeaks)):
        if np.abs(qpeaks[i]-y_pos[i])>1:
            accept[i] = False
    y_amps = y_amps[accept]
    y_pos = y_pos[accept]
    y_width = y_width[accept]

    for i in range(len(y_amps)):
        ax2.fill_between(q,y1=lorentz(q,y_amps[i],y_pos[i],y_width[i]),y2=np.zeros(len(q)),linestyle='-',color=color,linewidth=2,alpha=0.3)

    # Return parameters for windowing
    qpeaks = qpeaks[accept]
    window_size = np.sqrt(y_width)

    return qpeaks,window_size

def signal_filter(ix,y,q,yfft,qpeaks,window_size,title, label):
    # Inverse Fourier transform narrow window around each peak and generate Landau level plots
    f3,ax3 = plt.subplots(2,1,figsize=(8,7),gridspec_kw = {'height_ratios':[3,5]})
    cmap = truncate_colormap(mtl.cm.get_cmap('CMRmap'),0.12,0.72)
    norm = mtl.colors.Normalize(vmin=0, vmax=len(qpeaks)-0.5)

    ixplot_max = 0.7   # User can modify plotting range if desired
    for i in range(len(qpeaks)):
        yfft_copy = np.copy(yfft)
        color = cmap(norm(i))
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

        ax3[0].plot(ix,yfilter,color=color,linewidth=2)
        ax3[0].scatter(ix[mininds],yfilter[mininds],facecolor=color,edgecolor=color,s=50,linewidth=2,zorder=3)
        ax3[0].scatter(ix[maxinds],yfilter[maxinds],facecolor='w',edgecolor=color,s=60,linewidth=2,zorder=3)
        ax3[1].plot(ixn_plot,yplot,linestyle='--',color=color,linewidth=2)
        ax3[1].scatter(ix[mininds],n_min,s=50,facecolor=color,edgecolor=color,linewidth=2,zorder=3)
        ax3[1].scatter(ix[maxinds],n_max,s=60,facecolor='w',edgecolor=color,linewidth=2,zorder=3)

    ax3[0].set_title(label)
    ax3[0].set_ylabel(r'$\mathrm{\mathsf{\Delta}}$MR (%)')
    ax3[0].set_xlim([0,ixplot_max])
    ax3[0].set_ylim([3*np.min(yfilter),3*np.max(yfilter)])
    ax3[0].ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax3[0].locator_params(axis='y', nbins=5)
    ax3[0].tick_params(direction='in',top=False,right=False,length=5,width=1.5)
    x_fill = np.linspace(0,1./xmax,10)
    ax3[0].fill_between(x_fill,y1=[10*np.min(yfilter)]*len(x_fill),y2=[10*np.max(yfilter)]*len(x_fill),color='slategray',alpha=0.2,zorder=-1)

    ax3[1].set_xlabel(r'1/B (T$^\mathrm{-1}$)')
    ax3[1].set_xlim([0,ixplot_max])
    ax3[1].set_ylim([-0.5,22.5])
    ax3[1].set_ylabel('Landau level')
    ax3[1].tick_params(direction='in',top=False,right=False,length=5,width=1.5)
    ax3[1].fill_between(x_fill,y1=[-50]*len(x_fill),y2=[50]*len(x_fill),color='slategray',alpha=0.2,zorder=-1)

    f3.tight_layout()
    f3.savefig('SdH_'+label+'.png',bbox_inches='tight',dpi=400)