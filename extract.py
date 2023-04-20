import os
import pandas as pd
import numpy as np
import pickle as pkl
from scipy.interpolate import PchipInterpolator

# MR_S2
data_dir = 'D:/python_project/LandauLevel/data/MR_S2/'
T = 0.85    # thickness
W = 1.76    # width
L = 1.33    # length
ch = 2      # \pho_{xx} channel

# # MR_S4
# data_dir = 'D:/python_project/LandauLevel/data/MR_S4/'
# T = 0.61    # thickness
# W = 1.81    # width
# L = 0.70    # length
# ch = 1      # Rxx_{xx} channel

resu_name = data_dir[:-1] + '_extracted'

def resolve_monotone(Xs, Ys):
    idx = np.argsort(Xs)
    Xs_tmp = np.array(Xs)[idx]
    Ys_tmp = np.array(Ys)[idx]
    Xs_out = []
    Ys_out = []
    vx = 0
    vys = []
    for x, y in zip(Xs_tmp, Ys_tmp):
        if vys == []:
            vx = x
            vys.append(y)
        elif x == vx:
            vys.append(y)
        else:
            Xs_out.append(vx)
            Ys_out.append(sum(vys)/len(vys))
            vx = x
            vys = [y]
    if vys != []:
        Xs_out.append(vx)
        Ys_out.append(sum(vys)/len(vys))
    return np.array(Xs_out), np.array(Ys_out)

resu = {}
for file_name in os.listdir(data_dir):
    data = pd.read_csv(os.path.join(data_dir, file_name))
    data.rename(columns = {'Temperature (K)': 'Temperature', 
                           'Field (Oe)': 'Field', 
                           f'Resistance Ch{ch} (Ohms)': 'rho_xx'},
                inplace = True)
    
    data['Temperature'] = np.round(data['Temperature'], 0)  # round temperature to integer
    data['Field'] = data['Field']/10000                     # convert field unit from Gauss to Tesla
    data['rho_xx'] = data['rho_xx'] * W * T / (L * 1000)    # calculate sample resistivity

    rho_xx_0 = np.average(data.loc[(np.abs(data['Field']) <= max(0.05, np.min(np.abs(data['Field']))))]['rho_xx'])
    rho_xx_pos = data.loc[(data['Field'] > 0)].sort_values(by = 'Field', ascending = True)
    rho_xx_neg = data.loc[(data['Field'] < 0)].sort_values(by = 'Field', ascending = False)
    
    H = np.linspace(0, 9, 1000)
    pos_interp = PchipInterpolator(*resolve_monotone(rho_xx_pos['Field'], rho_xx_pos['rho_xx']), extrapolate = False)
    rho_xx_pos_new = pos_interp(H)
    # rho_xx_pos_new = np.interp(H, rho_xx_pos['Field'], rho_xx_pos['rho_xx'])
    neg_interp = PchipInterpolator(*resolve_monotone(np.abs(rho_xx_neg['Field']), rho_xx_neg['rho_xx']), extrapolate = False)
    rho_xx_neg_new = neg_interp(H)
    # rho_xx_neg_new = np.interp(H, np.abs(rho_xx_neg['Field']), rho_xx_neg['rho_xx'])
    
    rho_xx= (rho_xx_pos_new + rho_xx_neg_new)/2
    resu[data['Temperature'][0]] = [H[~np.isnan(rho_xx)], rho_xx[~np.isnan(rho_xx)], rho_xx_0]

with open(resu_name, 'wb') as f:
    pkl.dump(resu, f)