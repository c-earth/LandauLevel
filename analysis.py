import os
import pickle as pkl
import numpy as np

from util.Kohler import plot_rho, plot_MR, plot_MRK, plot_MREK
from util.back_ground import subbg_poly, subbg_pieces_poly, subbg_derivative
# from util.Landau import FFT, FFT_peaks, signal_filter

# # MR_S2
# data_file = 'D:/python_project/LandauLevel/data/MR_S2_extracted.pkl'
# resu_dir = 'D:/python_project/LandauLevel/results/MR_S2/'

# MR_S4
data_file = 'D:/python_project/LandauLevel/data/MR_S4_extracted.pkl'
resu_dir = 'D:/python_project/LandauLevel/results/MR_S4/'


# create analysis result folder
if not os.path.exists(resu_dir):
    os.mkdir(resu_dir)


# load extracted data
with open(data_file, 'rb') as f:
    Ts, Hs, rho_xxs = pkl.load(f)


# calculate magnetic resistances from resistivities
MRs = (rho_xxs / rho_xxs[:, :1] - 1) * 100


# Kohler's rule analysis
cutoff = 10
plot_rho(Ts, Hs, rho_xxs, cutoff, resu_dir)
plot_MR(Ts, Hs, MRs, cutoff, resu_dir)
plot_MRK(Ts, Hs, MRs, rho_xxs[:, :1], cutoff, resu_dir)
plot_MREK(Ts, Hs, MRs, rho_xxs[:, :1], cutoff, resu_dir)


# background subtraction
po_power = 5
pieces = 2
pp_power = 4
de_power = 2
T_max = 20
Hs_po, MRs_po = subbg_poly(Ts, Hs, MRs, po_power, T_max, resu_dir)
Hs_pp, MRs_pp = subbg_pieces_poly(Ts, Hs, MRs, pp_power, pieces, T_max, resu_dir)
Hs_de, MRs_de = subbg_derivative(Ts, Hs, MRs, de_power, T_max, resu_dir)

# data_subbg = subbg_poly(data, temperatures, 4)
# for t in range(len(temperatures)):
#     label = str(np.round(temperatures[t],1))+'K'
#     xc,yc = data_subbg[temperatures[t]]
#     xc = xc[1:]
#     yc = yc[1:]

#     ix,y,q,yfft = FFT(xc,yc)
#     qpeaks,window_size = FFT_peaks(q,yfft, label)
#     signal_filter(ix,y,q,yfft,qpeaks,window_size,label)