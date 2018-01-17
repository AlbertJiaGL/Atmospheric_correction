#!/usr/bin/python
import os
from glob import glob
import argparse
from l8_aerosol import solve_aerosol
from l8_correction import atmospheric_correction
parser = argparse.ArgumentParser(description='Landsat 8 Atmopsheric correction Excutable')
parser.add_argument('-p','--path',            help='Landsat path number',                               required=True)
parser.add_argument('-r','--row',             help='Landsat row number',                                required=True)
parser.add_argument('-D','--date',            help='Sensing date in the format of: YYYYMMDD',           required=True)
parser.add_argument('-f', "--l8_toa_dir",     help='Directory where you store L8 toa',                  required=True)
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
year, month, day = int(args.date[:4]), int(args.date[4:6]), int(args.date[6:8])
aero = solve_aerosol(year, month, day, l8_tile = (int(args.path), int(args.row)), emus_dir = args.emulator_dir, mcd43_dir   = args.MCD43_file_dir, l8_toa_dir = args.l8_toa_dir, global_dem=args.dem, cams_dir=args.cams)
aero.solving_l8_aerosol()
atmo_cor = atmospheric_correction(year, month, day, (int(args.path), int(args.row)))
atmo_cor.atmospheric_correction()
