import os
import sys
sys.path.insert(0, '.')
sys.path.insert(0, 'util')
import gdal
import requests
import numpy as np
from glob import glob
from get_modis import get_modisfiles
from grab_brdf import get_hv
from multi_process import parmap
from datetime import datetime, timedelta
from get_tile_lat_lon import get_tile_lat_lon

def downloader(url_root, fname, file_dir):
    new_url = url_root + fname
    new_req = requests.get(new_url, stream=True)
    print('downloading %s' % fname)
    with open(os.path.join(file_dir, fname), 'wb') as fp:
        for chunk in new_req.iter_content(chunk_size=1024):
            if chunk:
                fp.write(chunk)

def down_s2_emus(emus_dir):
    url = 'http://www2.geog.ucl.ac.uk/~ucfafyi/emus/'
    req = requests.get(url)
    for line in req.text.split():
        if '.pkl' in line:
            fname   = line.split('"')[1].split('<')[0]
            if 'S2' in fname:
                downloader(url, fname, emus_dir)

def down_l8_emus(emus_dir):
    url = 'http://www2.geog.ucl.ac.uk/~ucfafyi/emus/'
    req = requests.get(url)
    for line in req.text.split():
        if '.pkl' in line:
            fname   = line.split('"')[1].split('<')[0]
            if 'L8' in fname:
                downloader(url, fname, emus_dir)

def down_cams(cams_dir, cams_file):
    url = 'http://www2.geog.ucl.ac.uk/~ucfafyi/cams/'
    downloader(url, cams_file, cams_dir)

def down_dem(eles_dir, example_file):
    lats, lons = get_tile_lat_lon(example_file)
    url = 'http://www2.geog.ucl.ac.uk/~ucfafyi/eles/'
    min_lon, max_lon = np.floor(min(lons)), np.ceil(max(lons))
    min_lat, max_lat = np.floor(min(lats)), np.ceil(max(lats))
    rebuilt_vrt = False
    for la in np.arange(min_lat, max_lat + 1):
        for lo in np.arange(min_lon, max_lon + 1):
            if la>=0:
                lat_str = 'N%02d'%(int(abs(la)))
            else:
                lat_str = 'S%02d'%(int(abs(la)))
            if lo>=0:
                lon_str = 'E%03d'%(int(abs(lo)))
            else:
                lon_str = 'W%03d'%(int(abs(lo)))
            fname = 'ASTGTM2_%s%s_dem.tif'%(lat_str, lon_str)
            if len(glob(os.path.join(eles_dir, fname)))==0:
                downloader(url, fname, eles_dir)
                rebuilt_vrt = True
    if rebuilt_vrt:
        gdal.BuildVRT(eles_dir + '/global_dem.vrt', glob(eles_dir +'/*.tif'), outputBounds = (-180,-90,180,90)).FlushCache()
    
def down_s2_modis(modis_dir, s2_dir):
    date  = datetime.strptime('-'.join(s2_dir.split('/')[-5:-2]), '%Y-%m-%d')
    tiles = get_hv(s2_dir+'/B04.jp2')
    days   = [(date - timedelta(days = i)).strftime('%Y%j') for i in np.arange(16, 0, -1)] + \
             [(date + timedelta(days = i)).strftime('%Y%j') for i in np.arange(0, 17,  1)]
    fls = zip(np.repeat(tiles, len(days)), np.tile(days, len(tiles)))
    f = lambda fl: helper(fl, modis_dir)
    parmap(f, fls, nprocs=5)

def down_l8_modis(modis_dir, l8_file):
    date = datetime.strptime(l8_file.split('/')[-1].split('_')[3], '%Y%m%d')
    tiles = get_hv(l8_file)
    days   = [(date - timedelta(days = i)).strftime('%Y%j') for i in np.arange(16, 0, -1)] + \
             [(date + timedelta(days = i)).strftime('%Y%j') for i in np.arange(0, 17,  1)]
    fls = zip(np.repeat(tiles, len(days)), np.tile(days, len(tiles)))
    f = lambda fl: helper(fl, modis_dir)
    parmap(f, fls, nprocs=5)

def helper(fl, modis_dir):
    f_temp = modis_dir + '/MCD43A1.A%s.%s.006*.hdf'
    tile, day = fl
    if len(glob(f_temp%(day, tile))) == 0:
        year, doy = int(day[:4]), int(day[4:])
        get_modisfiles( 'MOTA', 'MCD43A1.006', year, tile, None,
                         doy_start= doy, doy_end = doy + 1, out_dir = modis_dir, verbose=1)

