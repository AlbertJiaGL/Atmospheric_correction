#!/usr/bin/env python
import gdal
import numpy as np
import pylab as plt
bands = ['B01',
         'B02',
         'B03',
         'B04',
         'B05',
         'B06',
         'B07',
         'B08',
         'B8A',
         'B09',
         'B10',
         'B11',
         'B12']

wv = [435.0,
      486.0,
      560.0,
      666.0,
      704.0,
      740.0,
      781.0,
      841.0,
      864.0,
      944.0,
      1378.0,
      1609.0,
      2185.0]

band_ratios = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 6, 2, 2]

def reader(fname, aoi, band_ratio):
    xoff, yoff, xsize, ysize = (np.array(aoi) / band_ratio).astype(int)
    data = gdal.Open(fname).ReadAsArray(xoff, yoff, xsize, ysize)/10000.
    data = np.repeat(np.repeat(data, band_ratio, axis=0), band_ratio, axis=1)
    return data

def read_aoi(dire, pixel=None):
    if pixel is None:
        pixel = np.random.choice(10980, 2)
    cx, cy = pixel
    xoff, yoff   = max(cx-60, 0),         max(cy-60, 0)
    xsize, ysize = min(120 + xoff, 10980) - xoff, min(120 +  yoff, 10980) - yoff
    aoi = xoff, yoff, xsize, ysize
    dats = []
    for i, j in enumerate(bands):
        fname = dire+'/'+j+'.jp2'
        dats.append(reader(fname, aoi, band_ratios[i]))
    return dats

def spec_viewer(dire):
    for i in range(20):
        dats = read_aoi(dire)
        fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(16, 8))
        ax[0].plot(wv, [i[60, 60] for i in dats], '--o', ms=2) 
        ax[0].set_ylim(0, 0.6)
        ax[0].set_xlim(400, 2500)
        ax[1].imshow(np.minimum(np.array(dats[3:0:-1]).transpose(1,2,0)/0.25, 1.))   
    plt.show()
if __name__ == '__main__':
    spec_viewer('./S2_data/49/S/DT/2018/1/1/0/')

