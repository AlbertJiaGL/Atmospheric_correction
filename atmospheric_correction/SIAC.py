#!/usr/bin/env python
import os
import sys
import argparse
from the_aerosol import solve_aerosol
from the_correction import atmospheric_correction
root = os.getcwd()
parser = argparse.ArgumentParser(description='SIAT Atmopsheric correction Excutable')
parser.add_argument('-s2', "--S2_file_dir",       help='Sentinel 2 file path',                              default=None)
parser.add_argument('-l8', "--L8_file_dir",       help='Landsat 8 file path',                               default=None)
parser.add_argument("-m", "--MCD43_file_dir",    help="Directory where you store MCD43A1.006 data",        default=root + '/MCD43/')
parser.add_argument("-e", "--emulator_dir",      help="Directory where you store emulators.",              default=root + '/emus/')
parser.add_argument("-d", "--dem",               help="A global dem file, and a vrt file is recommonded.", default=root + '/eles/global_dem.vrt')
parser.add_argument("-c", "--cams",              help="Directory where you store cams data.",              default=root + '/cams/')
parser.add_argument("-ss", "--sensor_satellite", help="Data from which Satellite is used: S2A or S2B",     default=None)
parser.add_argument("--version",                 action="version",                                         version='%(prog)s - Version 1.0')
args = parser.parse_args()

if args.S2_file_dir is not None:
    args.sensor_satellite = 'MSI', 'S2A'
    print args.sensor_satellite 

