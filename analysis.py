import os
import pickle as pkl
import numpy as np

from util.Kohler import plot_rho, plot_MR, plot_MRK, plot_MREK
from util.back_ground import subbg_po, subbg_pp, subbg_de
from util.Landau import FFT#, FFT_peaks, signal_filter

# MR_S2
data_file = 'D:/python_project/LandauLevel/data/MR_S2_extracted.pkl'
resu_dir = 'D:/python_project/LandauLevel/results/MR_S2/'
po_power = 7
pieces = 2
pp_power = 4
avg_window = 10

# # MR_S4
# data_file = 'D:/python_project/LandauLevel/data/MR_S4_extracted.pkl'
# resu_dir = 'D:/python_project/LandauLevel/results/MR_S4/'
# po_power = 5
# pieces = 2
# pp_power = 3
# avg_window = 10


# select background options
po = False
pp = False
de = True


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
T_max = 20
QO_data = dict()
if po:
    QO_data['po'] = subbg_po(Ts, Hs, MRs, po_power, T_max, resu_dir)
if pp:
    QO_data['pp'] = subbg_pp(Ts, Hs, MRs, pp_power, pieces, T_max, resu_dir)
if de:
    QO_data['de'] = subbg_de(Ts, Hs, MRs, avg_window, T_max, resu_dir)


for subbg, (Ts_sub, Hs_sub, MRs_sub) in QO_data.items():
    iHs, MRs_iH, qs, MRs_iH_fft = FFT(Ts_sub, Hs_sub[Hs_sub > 0], MRs_sub[:, Hs_sub > 0], subbg, resu_dir)
#     signal_filter(ix,y,q,yfft,qpeaks,window_size,label)