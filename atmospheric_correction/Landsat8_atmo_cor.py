#!/usr/bin/python
import os
from glob import glob
import argparse
from l8_aerosol import solve_aerosol
from l8_correction import atmospheric_correction
parser = argparse.ArgumentParser(description='Landsat 8 Atmopsheric correction Excutable')
parser.add_argument('-f', "--l8_file",        help='A L8 file',                                         required=True)
parser.add_argument("-m", "--MCD43_file_dir", help="Directory where you store MCD43A1.006 data",        default='./MCD43/')
parser.add_argument("-e", "--emulator_dir",   help="Directory where you store emulators.",              default='./emus/')
parser.add_argument("-d", "--dem",            help="A global dem file, and a vrt file is recommonded.", default='./eles/global_dem.vrt')
parser.add_argument("-c", "--cams",           help="Directory where you store cams data.",              default='./cams/')
parser.add_argument("--version",              action="version",                                         version='%(prog)s - Version 2.0')
args = parser.parse_args()
if len(glob(args.emulator_dir + '/*OLI*L8*.pkl')) < 3:
   print('No emus, start downloading...')
   url = 'http://www2.geog.ucl.ac.uk/~ucfafyi/emus/'
   import requests
   req = requests.get(url)
   for line in req.text.split():
       if '.pkl' in line:
           fname   = line.split('"')[1].split('<')[0]
           if 'L8' in line:
               new_url = url + fname
               new_req = requests.get(new_url, stream=True)
               print('downloading %s' % fname)
               with open(os.path.join(args.emulator_dir, fname), 'wb') as fp:
                   for chunk in new_req.iter_content(chunk_size=1024):
                       if chunk:
                           fp.write(chunk)
l8_toa_dir = '/'.join(args.l8_file.split('/')[:-1])
pr, date  = args.l8_file.split('/')[-1].split('_')[2:4]
path, row = int(pr[:3]), int(pr[3:])
year, month, day = int(date[:4]), int(date[4:6]), int(date[6:8])
aero = solve_aerosol(year, month, day, l8_tile = (int(path), int(row)), emus_dir = args.emulator_dir, mcd43_dir   = args.MCD43_file_dir, l8_toa_dir = l8_toa_dir, global_dem=args.dem, cams_dir=args.cams)
aero.solving_l8_aerosol()
atmo_cor = atmospheric_correction(year, month, day, (int(path), int(row)))
atmo_cor.atmospheric_correction()
