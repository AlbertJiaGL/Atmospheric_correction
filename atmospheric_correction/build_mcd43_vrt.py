#! /usr/bin/env python
import os
import gdal
from glob import glob
import numpy as np
mcd43_dir = '/data/nemesis/MCD43/'
flist = np.array(glob(mcd43_dir + 'MCD43A1*.006.*hdf'))
all_dates = np.array([i.split('/')[-1] .split('.')[1][1:9] for i in flist])
udates = np.unique(all_dates)
temp1 = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Band_Mandatory_Quality_%s'
temp2 = 'HDF4_EOS:EOS_GRID:"%s":MOD_Grid_BRDF:BRDF_Albedo_Parameters_%s'
for i in udates:
    fnames =  flist[all_dates==i]
    #print flist[mask]
    for temp in [temp1, temp2]:
        for band in ['Band1','Band2','Band3','Band4','Band5','Band6','Band7', 'vis', 'nir', 'shortwave']:
            bs = []
            for fname in fnames:
                bs.append(temp%(fname, band))
            if not os.path.exists('MCD43/' + '%s/'%i):
                os.mkdir('MCD43/' + '%s/'%i)
            gdal.BuildVRT('MCD43/' + '%s/'%i + '_'.join(['MCD43', i, bs[0].split(':')[-1]])+'.vrt', bs).FlushCache()
        
