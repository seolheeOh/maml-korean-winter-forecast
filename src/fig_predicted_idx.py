import os, pathlib, sys
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import BoundaryNorm

main_dir = '/home/doseol1129/Tool/MAML/seolhee/'

import sys
sys.path.append(main_dir+'src/')
from utils import correlation, mse, rmse

train_iter = 100
train_update = 10

meta_exp = 'train-itr'+str(train_iter)+'-up'+str(train_update)

opath = main_dir+'src/data/'

exp_name = 'kortemp_DJF'
oname = 'fcst_MAML_DJF'

# Load ground truth data
ipath = main_dir+'dataset/'
f = Dataset(ipath+'lab_DJF.nc','r')
lab = f.variables['p'][50:]
f.close()

tdim = len(lab)
lab = lab.reshape(tdim)
em = np.zeros((60,tdim))

# Load predicted result
for e, ens in enumerate(range(1,61)):
    ipath = main_dir+'output/'+exp_name+'/'+meta_exp+'/'
    f = Dataset(ipath+'fcst_ens0'+str(ens)+'.nc','r')
    fvar = f.variables['p'][:]
    f.close()

    meta = fvar[:]
    del fvar

    meta = meta.reshape(tdim)
    em[e] = meta

# Ensemble mean
meta = np.mean(em,axis=0)
meta = meta.reshape(tdim)

# Compute anomaly
meta = meta-np.mean(meta)
meta = np.array(meta)

# Calculate skills
meta_cor = np.array(correlation(meta,lab))
meta_rmse = np.array(rmse(meta,lab))
print(meta_cor)

# Draw figure
fig, ax = plt.subplots(1,1, figsize=(8, 3))

tdim = np.arange(tdim)
labels = np.arange(2000, 2021)
target_name = 'a.DJF (2000/01-2020/21)'

fline = ax.plot(tdim, lab, 'black', label = 'Ground truth')
gline = ax.plot(tdim, meta, 'red', label = 'MAML (Cor.:'+str(np.round(meta_cor,2))+', RMSE : '+str(np.round(meta_rmse[0],3))+')')

plt.setp(fline,linewidth=3,marker='o',markersize=0,zorder=9)
plt.setp(gline,linewidth=1,marker='o',markersize=5,zorder=9)
ax.tick_params(axis='both', labelsize=7)

minor_yticks = np.arange(-3.5,3.51,0.5)
major_yticks = np.arange(-3,3.1,1)
ytick_labels = [str(i) for i in major_yticks]

ax.grid(axis='y', which = 'major', linestyle = '--')
ax.grid(axis='x', which = 'major', linestyle = '--')

ax.set_xticks(tdim,labels,fontsize=11, rotation=30)

ax.set_yticks(major_yticks)
ax.set_yticks(minor_yticks, minor = True)
ax.set_yticklabels(ytick_labels, fontsize=11)

ax.legend(loc='upper left', prop={'size':9}, ncol=3)
ax.set_title(target_name, fontsize= 13, loc='left', pad=5)

plt.tight_layout(rect=(0,0,1,1))

plt.show()
