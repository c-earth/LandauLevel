import pickle as pkl
import numpy as np
import matplotlib as mtl
import matplotlib.pyplot as plt

from util.Kohler import plot_rho, plot_MR, plot_MRK, plot_MREK
from util.back_ground import subbg_poly, subbg_pieces_poly, subbg_derivative
from util.Landau import truncate_colormap, FFT

data_file = 'D:/python_project/LandauLevel/data/MR_S4_extracted'
resu_dir = 'D:/python_project/LandauLevel/results/MR_S4/'

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

xmin = 0.05 # Lower bound on field to consider (quality of background subtraction may affect this number)
xmax = 9.   # Upper bound on field to consider

Fmin = 1.5  # Lower bound on frequency to cut off tail of 0 frequency component (user should change this accordingly)
Fmax = 70   # Upper bound on frequency to exclude small high frequency components

f1,ax1 = plt.subplots(1,1,figsize=(8,7))
f2,ax2 = plt.subplots(1,1,figsize=(8,7))

cmap = truncate_colormap(mtl.cm.get_cmap('CMRmap'),0.12,0.72)
norm = mtl.colors.Normalize(vmin=np.min(temperatures), vmax=np.max(temperatures))

data_subbg = subbg_poly(data, temperatures, 4)
for t in range(len(temperatures)):
    label = str(np.round(temperatures[t],1))+'K'
    color = cmap(norm(temperatures[t]))
    xc,yc = data_subbg[temperatures[t]]
    xc = xc[1:]
    yc = yc[1:]

    ix,y,q,yfft = FFT(xc,yc)
    # qpeaks,window_size = FFT_peaks(q,yfft,color,label)
    # signal_filter(ix,y,q,yfft,qpeaks,window_size,label)


ax1.set_xlabel('B (T)')
ax1.set_ylabel('MR (%)')
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax1.tick_params(direction='in',top=False,right=False,length=5,width=1.5)
ax1.set_xlim([0,9])
ax1.set_ylim([0,1.5E5])
ax1.legend(loc='upper left')

ax2.set_xlabel('Frequency (T)')
ax2.set_ylabel('Amplitude (a.u.)')
ax2.set_xlim([Fmin,Fmax])
ax2.set_ylim(bottom=0)
ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax2.tick_params(direction='in',top=False,right=False,length=5,width=1.5)
ax2.locator_params(axis='y', nbins=6)
ax2.legend(loc='upper right')

f1.tight_layout()
f2.tight_layout()
f1.savefig('Background.png',bbox_inches='tight',dpi=400)
f2.savefig('FFT.png',bbox_inches='tight',dpi=400)

plt.show()