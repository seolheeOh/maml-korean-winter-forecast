from netCDF4 import Dataset
import numpy as np
import os, math

def detrend_idx (idx):

    idx = idx.reshape(-1)
    tdim = len(idx)

    t = np.arange(1,tdim+1)
    t = t-np.mean(t)

    xx = np.mean(t*t)
    xy = np.mean(t*idx)

    reg = xy/xx
    treg = t*reg

    idx = idx-treg
    return idx, reg

def correlation (idx1, idx2):

    idx1 = idx1-np.mean(idx1,axis=0)
    idx2 = idx2-np.mean(idx2,axis=0)

    xx = np.mean(idx1*idx1,axis=0)
    yy = np.mean(idx2*idx2,axis=0)
    xy = np.mean(idx1*idx2,axis=0)

    cor = xy/np.sqrt(xx*yy)

    sig = cor*np.sqrt(len(idx1)-2)/np.sqrt(1-cor**2)
#    print('p-value :', sig)

    return cor

def rmse (idx1, idx2):
#    idx1 = idx1-np.mean(idx1,axis=0)
#    idx2 = idx2-np.mean(idx2,axis=0)

    rmse = np.sqrt(np.mean(pow(idx1-idx2,2)))
    rmse = np.array(rmse).reshape(1)

    return rmse

def mse (idx1, idx2):
#    idx1 = idx1-np.mean(idx1,axis=0)
#    idx2 = idx2-np.mean(idx2,axis=0)

    mse = np.mean(pow(idx1-idx2,2))
    mse = np.array(mse).reshape(1)

    return mse

def aave(dat, lon_st, lon_nd, lat_st, lat_nd,
        lon_init=0.0,  lon_int=2.5,
        lat_init=-90.0, lat_int=2.5,
        equal_area_treat=True, fill_value=-9.99e+08):
    
    print(lon_init, lon_int, lat_init, lat_int)
#    lat_st = lat_st+lat_int
#    lat_nd = lat_nd+lat_int
    x_st = int( math.floor(( ( lon_st - lon_init ) / lon_int ) + 0.5 ))
    x_nd = int( math.floor(( ( lon_nd - lon_init ) / lon_int ) + 1 + 0.5))
    y_st = int( math.floor(( ( lat_st - lat_init ) / lat_int )+ 0.5 ))
    y_nd = int( math.floor(( ( lat_nd - lat_init ) / lat_int ) + 1 +0.5 ))
    
    n_dim = len(dat.shape)
    print(x_st, x_nd, y_st, y_nd)
    
    if n_dim < 2:
        print('The input must be at least two-dimensional.')
        
    # get dimension
    o_dim = dat.shape[:-2]
    ydim, xdim = dat.shape[-2:]
    
    # crop
    dat = dat.reshape(-1,ydim,xdim)
    crop = dat[:,y_st:y_nd,x_st:x_nd]
    
    # get crop dimension       
    ydim2, xdim2 = crop.shape[1:]
    n_grid = crop.shape[1] * crop.shape[2]
    
    # mask missing values
    crop = np.ma.masked_equal(crop, fill_value)
    
    if equal_area_treat == True:
        # make latitude list
        lat_list = np.arange(lat_st,lat_st+lat_int*(ydim2), lat_int)
        # cosine theta
        weight = np.cos(np.deg2rad(lat_list))
        weight = weight.reshape(ydim2, 1)
        weight = np.expand_dims(np.repeat(weight, xdim2, axis=1), axis=0)
        
        # mask missing values
#        weight[crop[0].mask==True] = fill_value
        weight = np.ma.masked_equal(weight, fill_value)
        
        crop = np.nansum(crop*weight,axis=(1,2))/np.nansum(weight,axis=(1,2))
        
        if len(crop.shape) == 1 and crop.shape[0] == 1:   
            crop = np.array(crop)
        else:
            crop = crop.reshape(o_dim)
        return crop



