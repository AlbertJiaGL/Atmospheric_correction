#/usr/bin/env python 
import os
import sys
sys.path.insert(0, 'util')
import gdal
import json
import datetime
import logging
import numpy as np
#from ddv import ddv
from glob import glob
from scipy import signal, ndimage
try:
    import cPickle as pkl
except:
    import pickle as pkl
from osgeo import osr
from scipy.ndimage import binary_dilation, binary_erosion
from smoothn import smoothn
from grab_s2_toa import read_s2
from multi_process import parmap
from reproject import reproject_data
from scipy.interpolate import griddata
from grab_brdf import MCD43_SurRef, array_to_raster
#from grab_uncertainty import grab_uncertainty
from atmo_solver_test import solving_atmo_paras
#from emulation_engine import AtmosphericEmulationEngine
from psf_optimize import psf_optimize
import warnings
warnings.filterwarnings("ignore")

class solve_aerosol(object):
    '''
    Prepareing modis data to be able to pass into 
    atmo_cor for the retrieval of atmospheric parameters.
    '''
    def __init__(self,
                 year, 
                 month, 
                 day,
                 emus_dir    = '/home/ucfajlg/Data/python/S2S3Synergy/optical_emulators',
                 mcd43_dir   = '/data/selene/ucfajlg/Ujia/MCD43/',
                 s2_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/s_data/',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 wv_emus_dir = '/home/ucfafyi/DATA/Multiply/emus/wv_MSI_retrieval_S2A.pkl',
                 cams_dir    = '/home/ucfafyi/DATA/Multiply/cams/',
                 mod08_dir   = '/home/ucfafyi/DATA/Multiply/mod08/',
                 satellite   = 'S2A',
                 s2_tile     = '29SQB',
                 acquisition = '0',
                 s2_psf      = None,
                 qa_thresh   = 255,
                 aero_res    = 610, # resolution for aerosol retrival in meters should be larger than 500
                 reconstruct_s2_angle = True):

        self.year        = year 
        self.month       = month
        self.day         = day
        self.date        = datetime.datetime(self.year, self.month, self.day)
        self.doy         = self.date.timetuple().tm_yday
        self.mcd43_dir   = mcd43_dir
        self.emus_dir    = emus_dir
        self.qa_thresh   = qa_thresh
        self.s2_toa_dir  = s2_toa_dir
        self.global_dem  = global_dem
        self.wv_emus_dir = wv_emus_dir
        self.cams_dir    = cams_dir
        self.mod08_dir   = mod08_dir 
        self.satellite   = satellite
        self.s2_tile     = s2_tile
        self.acquisition = acquisition
        self.s2_psf      = s2_psf 
        self.s2_u_bands  = 'B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A', 'B09' #bands used for the atmo-cor
        self.band_indexs = [1, 2, 3, 7, 11, 12]
        self.boa_bands   = [469, 555, 645, 869, 1640, 2130]
        self.full_res    = (10980, 10980)
        self.aero_res    = aero_res
        self.mcd43_tmp   = '%s/MCD43A1.A%d%03d.%s.006.*.hdf'
        self.reconstruct_s2_angle  = reconstruct_s2_angle
        self.s2_spectral_transform = [[ 1.06946607,  1.03048916,  1.04039226,  1.00163932,  1.00010918, 0.95607606,  0.99951677],
                                      [ 0.0035921 , -0.00142761, -0.00383504, -0.00558762, -0.00570695, 0.00861192,  0.00188871]]       
    def _load_xa_xb_xc_emus(self,):
        xap_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xap_%s.pkl'%(self.s2_sensor, self.satellite))[0]
        xbp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xbp_%s.pkl'%(self.s2_sensor, self.satellite))[0]
        xcp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xcp_%s.pkl'%(self.s2_sensor, self.satellite))[0]
        if sys.version_info >= (3,0):
            f = lambda em: pkl.load(open(em, 'rb'), encoding = 'latin1')
        else:
            f = lambda em: pkl.load(open(em, 'rb'))
        self.emus = parmap(f, [xap_emu, xbp_emu, xcp_emu])

    def repeat_extend(self,data, shape=(10980, 10980)):
        da_shape    = data.shape
        re_x, re_y  = int(1.*shape[0]/da_shape[0]), int(1.*shape[1]/da_shape[1])
        new_data    = np.zeros(shape)
        new_data[:] = -9999
        new_data[:re_x*da_shape[0], :re_y*da_shape[1]] = np.repeat(np.repeat(data, re_x,axis=0), re_y, axis=1)
        return new_data
        
    def gaussian(self, xstd, ystd, angle, norm = True):
        win = 2*int(round(max(1.96*xstd, 1.96*ystd)))
        winx = int(round(win*(2**0.5)))
        winy = int(round(win*(2**0.5)))
        xgaus = signal.gaussian(winx, xstd)
        ygaus = signal.gaussian(winy, ystd)
        gaus  = np.outer(xgaus, ygaus)
        r_gaus = ndimage.interpolation.rotate(gaus, angle, reshape=True)
        center = np.array(r_gaus.shape)/2
        cgaus = r_gaus[int(center[0]-win/2): int(center[0]+win/2), int(center[1]-win/2):int(center[1]+win/2)]
        if norm:
            return cgaus/cgaus.sum()
        else:
            return cgaus 
    
    def _get_tcwv(self, img, vza, vaa, sza, saa, ele):
        b8a       = np.repeat(np.repeat(img['B8A']*0.0001, 2, axis=0), 2, axis=1)[self.Hx, self.Hy]
        b9        = np.repeat(np.repeat(img['B09']*0.0001, 6, axis=0), 6, axis=1)[self.Hx, self.Hy]
        try:
            wv_emus   = pkl.load(open(self.wv_emus_dir, 'rb'))
        except:
            wv_emus   = pkl.load(open(self.wv_emus_dir, 'rb'), encoding = 'latin1')
        up, bot = wv_emus.inputs.max(axis=0), wv_emus.inputs.min(axis=0)
        vza, vaa  = vza['B09'][self.Hx, self.Hy], vaa['B09'][self.Hx, self.Hy]
        sza, saa  = sza[self.Hx, self.Hy], saa[self.Hx, self.Hy]
        elevation = ele[self.Hx, self.Hy]
        raa       = abs(saa-vaa)
        inputs    = np.array([b9, b8a, sza, vza, raa, elevation]).T
        tcwv_mask = ( b8a < 0.1 ) | self.bad_pix[self.Hx, self.Hy] | (b8a > up[1]) | (b9 > up[0]) | (b9 < bot[0]) | \
                                                                   (sza > up[2]) | (sza < bot[2]) | (vza > up[3]) | \
                                                                   (vza < bot[3]) | (raa > up[4]) | (raa < bot[4]) | \
                                                                   (elevation > up[5]) | (elevation < bot[5])
        if tcwv_mask.all():
            self.logger.warning('Inputs values for WV emulator are out of tainning ranges and ECMWF tcwv is used.')
            #pass
        else:
            tcwv      = np.zeros(self.full_res)
            tcwv[:]   = np.nan
            tcwv_unc  = tcwv.copy()
            s2_tcwv, s2_tcwv_unc, _ = wv_emus.predict(inputs, do_unc = True)
            if tcwv_mask.sum() >= 1:
                s2_tcwv[tcwv_mask]  = np.interp(np.flatnonzero( tcwv_mask), \
                            np.flatnonzero(~tcwv_mask), s2_tcwv[~tcwv_mask]) # simple interpolation
            tcwv    [self.Hx, self.Hy] = s2_tcwv
            tcwv_unc[self.Hx, self.Hy] = s2_tcwv_unc
            if not np.isnan(tcwv).all():
                self.tcwv                  = np.nanmean(tcwv    .reshape(self.num_blocks, self.block_size, \
                                                                         self.num_blocks, self.block_size), axis = (3,1))
                self.tcwv_unc              = np.nanmax (tcwv_unc.reshape(self.num_blocks, self.block_size, \
                                                                         self.num_blocks, self.block_size), axis = (3,1)) + 0.05
            else:
                self.logger.warning('Failed to get TCWV from sen2cor look up table and ECMWF tcwv is used.')
    def _get_psf(self, selected_img):
        self.logger.info('No PSF parameters specified, start solving.')
        high_img    = np.repeat(np.repeat(selected_img['B11'], 2, axis=0), 2, axis=1)*0.0001
        high_indexs = self.Hx, self.Hy
        low_img     = np.ma.array(self.s2_boa[4])
        qa, cloud   = self.s2_boa_qa[4], self.bad_pix
        xstd, ystd  = 29.75, 39
        psf         = psf_optimize(high_img, high_indexs, low_img, qa, cloud, 0.1, xstd = xstd, ystd = ystd)
        xs, ys      = psf.fire_shift_optimize()
        ang         = 0
        self.logger.info('Solved PSF parameters are: %.02f, %.02f, %d, %d, %d, and the correlation is: %.03f.' \
                             %(xstd, ystd, 0, xs, ys, 1-psf.costs.min()))
        return xstd, ystd, ang, xs, ys

    def _mcd43_cloud(self,flist, lx, ly, example_file, boa, b12):
           
        g            = gdal.BuildVRT('', list(flist))
        if g is None:
            print('Please download files: ', [i.split('"')[1].split('/')[-1] for i in list(flist)])
            raise IOError
        temp_data    = np.zeros((g.RasterYSize, g.RasterXSize))
        temp_data[:] = np.nan
        temp_data[lx, ly] = boa[5,:]
        self.boa_b12 = reproject_data(array_to_raster(temp_data, g), example_file, outputType = gdal.GDT_Float32).data
        toa_b12      = np.repeat(np.repeat(b12/10000., 2, axis=0), 2, axis=1)
        mask         = (abs(self.boa_b12 - toa_b12)>0.1) | (self.boa_b12 < 0.01) | (self.boa_b12 > 1.)
        emask        = binary_erosion(mask, structure=np.ones((3,3)).astype(bool), iterations=15)
        dmask        = binary_dilation(emask, structure=np.ones((3,3)).astype(bool), iterations=100)
        self.cloud   = self.cloud | dmask | (toa_b12 < 0.0001)

    def _mod08_aot(self,):
        try:
            temp = 'HDF4_EOS:EOS_GRID:"%s":mod08:Aerosol_Optical_Depth_Land_Ocean_Mean'
            g = gdal.Open(temp%glob('%s/MOD08_D3.A2016%03d.006.*.hdf'%(self.mod08_dir, self.doy))[0])
            dat = reproject_data(g, self.s2_file_dir+'/B01.jp2', outputType= gdal.GDT_Float32).data * g.GetRasterBand(1).GetScale() + g.GetRasterBand(1).GetOffset()
            dat[dat<=0]  = np.nan
            dat[dat>1.5] = np.nan
            mod08_aot = np.nanmean(dat)
        except:
            mod08_aot = np.nan
        return mod08_aot

    def _get_ddv_aot(self, selected_img, tcwv, tco3, ele_data):
        b2, b4,   = selected_img['B02']/10000., selected_img['B04']/10000.
        b8, b12   = selected_img['B08']/10000., np.repeat(np.repeat(selected_img['B12']/10000., 2, axis=0), 2, axis=1)
        ndvi_mask = (((b8 - b4)/(b8 + b4)) > 0.4) & (b12 > 0.01) & (b12 < 0.25) & (~self.bad_pix)
        if ndvi_mask.sum() < 1000:
            self.logger.info('No enough DDV found in this sence for aot restieval, and ECWMF AOT is used.')
        else:
            Hx, Hy = np.where(ndvi_mask)
            if ndvi_mask.sum() > 1000:
                random_choice     = np.random.choice(len(Hx), 1000, replace=False)
                random_choice.sort()
                Hx, Hy            = Hx[random_choice], Hy[random_choice]
                ndvi_mask[:]      = False
                ndvi_mask[Hx, Hy] = True
            Hx, Hy    = np.where(ndvi_mask)
            nHx, nHy  = (10 * Hx/self.aero_res).astype(int), (10 * Hy/self.aero_res).astype(int)
            blue_vza  = np.cos(np.deg2rad(self.vza[0, nHx, nHy]))
            blue_sza  = np.cos(np.deg2rad(self.sza[nHx, nHy]))
            red_vza   = np.cos(np.deg2rad(self.vza[2, nHx, nHy]))
            red_sza   = np.cos(np.deg2rad(self.sza[nHx, nHy]))
            blue_raa  = np.cos(np.deg2rad(self.vaa[0, nHx, nHy] - self.saa[nHx, nHy]))
            red_raa   = np.cos(np.deg2rad(self.vaa[2, nHx, nHy] - self.saa[nHx, nHy]))
            red, blue = b4 [Hx, Hy], b2[Hx, Hy]
            swif      = b12[Hx, Hy]
            red_emus  = np.array(self.emus)[:, 3]
            blue_emus = np.array(self.emus)[:, 1]

            zero_aod    = np.zeros_like(red)
            red_inputs  = np.array([red_sza,  red_vza,  red_raa,  zero_aod, tcwv[nHx, nHy], tco3[nHx, nHy], ele_data[Hx, Hy]])
            blue_inputs = np.array([blue_sza, blue_vza, blue_raa, zero_aod, tcwv[nHx, nHy], tco3[nHx, nHy], ele_data[Hx, Hy]])

            p           = np.r_[np.arange(0., 0.01, 0.0001), np.arange(0.01, 1., 0.02), np.arange(1., 1.5, 0.05),  np.arange(1.5, 2., 0.1)]
            f           = lambda aot: self._ddv_cost(aot, blue, red, swif, blue_inputs, red_inputs,  blue_emus, red_emus)
            costs       = parmap(f, p)
            min_ind     = np.argmin(costs)
            self.logger.info('DDV solved aod is %.03f.'% p[min_ind])
            mod08_aot   = self._mod08_aot()
            #print(mod08_aot, self.aot.mean(), p[min_ind])
            self.aot[:] = np.nanmean([mod08_aot, self.aot.mean(), p[min_ind]])

    def _ddv_cost(self, aot, blue, red, swif, blue_inputs, red_inputs,  blue_emus, red_emus):
        blue_inputs[3, :] = aot
        red_inputs [3, :] = aot
        blue_xap_emu, blue_xbp_emu, blue_xcp_emu = blue_emus
        red_xap_emu,  red_xbp_emu,  red_xcp_emu  = red_emus
        blue_xap, blue_xbp, blue_xcp             = blue_xap_emu.predict(blue_inputs.T)[0], \
                                                   blue_xbp_emu.predict(blue_inputs.T)[0], \
                                                   blue_xcp_emu.predict(blue_inputs.T)[0]
        red_xap,  red_xbp,  red_xcp              = red_xap_emu.predict(red_inputs.T)  [0], \
                                                   red_xbp_emu.predict(red_inputs.T)  [0], \
                                                   red_xcp_emu.predict(red_inputs.T)  [0]
        y        = blue_xap * blue - blue_xbp
        blue_sur = y / (1 + blue_xcp * y)
        y        = red_xap * red - red_xbp
        red_sur  = y / (1 + red_xcp * y)
        blue_dif = 0 #(blue_sur - 0.25 * swif   )**2
        red_dif  = 0 #(red_sur  - 0.5  * swif   )**2
        rb_dif   = (blue_sur - 0.6  * red_sur)**2
        cost     = 0.5 * (blue_dif + red_dif + rb_dif)
        return cost.sum()

    def mask_bad_pix(self, selected_img):
        ndvi                 = (selected_img['B08']-selected_img['B04'])/(1. * selected_img['B04'] + selected_img['B08'])
        water_mask           = ((ndvi < 0.01) & (selected_img['B08'] < 1100)) | ((ndvi < 0.1) & (selected_img['B08'] < 500)) | \
                                                  np.repeat(np.repeat((selected_img['B12'] < 1), 2, axis=0), 2, axis=1)
        self.ker_size        = int(round(max(1.96 * 29.75, 1.96 * 39)))
        self.water_mask      = binary_erosion (water_mask, structure = np.ones((3,3)).astype(bool), iterations=5).astype(bool) 
        self.bad_pix         = binary_dilation(self.cloud | self.water_mask, \
                               structure=np.ones((3,3)).astype(bool), iterations=int(self.ker_size/2)).astype(bool)

    def _s2_aerosol(self,):
        
        self.logger.propagate = False
        self.logger.info('Start to retrieve atmospheric parameters.')
        s2   = read_s2(self.s2_toa_dir, self.s2_tile, self.year, self.month, self.day, acquisition = self.acquisition, bands = self.s2_u_bands)
        
        self.logger.info('Reading in TOA reflectance.')
        selected_img      = s2.get_s2_toa() 
        self.s2_file_dir  = s2.s2_file_dir
        self.example_file = s2.s2_file_dir+'/B04.jp2'
        
        self.logger.info('Getting cloud mask.')
        self.cloud = s2.get_s2_cloud()
        
        self.logger.info('Loading emulators.')
        self._load_xa_xb_xc_emus()
        
        self.logger.info( 'Getting the angles and simulated surface reflectance.')
        s2.get_s2_angles()
        sa_files = [s2.angles['saa'], s2.angles['sza']] 
        if os.path.exists(self.s2_file_dir + '/angles/VAA_VZA_B12.tif'):
            va_files = [self.s2_file_dir + '/angles/VAA_VZA_B%02d.tif'%i for i in [2,3,4,8,11,12]]
        else:
            va_files = [np.array([s2.angles['vaa'][band], s2.angles['vza'][band]]) for band in self.s2_u_bands[:-2]]
        if len(glob(self.s2_file_dir + '/MCD43.npz')) == 0:
            boa, unc, hx, hy, lx, ly, flist = MCD43_SurRef(self.mcd43_dir, self.example_file, \
                                                           self.year, self.doy, [sa_files, va_files],
                                                           sun_view_ang_scale=[1, 0.01], bands = [3,4,1,2,6,7], tolz=0.001, reproject=False)
            np.savez(self.s2_file_dir + '/MCD43.npz', boa=boa, unc=unc, hx=hx, hy=hy, lx=lx, ly=ly, flist=flist)
        else:
            f = np.load(self.s2_file_dir + '/MCD43.npz', encoding='latin1')
            boa, unc, hx, hy, lx, ly, flist = f['boa'], f['unc'], f['hx'], f['hy'], f['lx'], f['ly'], f['flist']
            scale = 0.01 / np.nanmin(unc)
            unc   = scale * np.array(unc)
        self.Hx, self.Hy = hx, hy
        
        self.logger.info('Update cloud mask.') 
        self.mask_bad_pix(selected_img)
        
        self.logger.info('Applying spectral transform.')
        self.s2_boa_qa = np.ma.array(unc)
        self.s2_boa    = np.ma.array(boa)*np.array(self.s2_spectral_transform)[0,:-1][...,None] + \
                                          np.array(self.s2_spectral_transform)[1,:-1][...,None]
        shape          = (self.num_blocks, int(s2.angles['sza'].shape[0] / self.num_blocks), \
                          self.num_blocks, int(s2.angles['sza'].shape[1] / self.num_blocks))
        self.sza = s2.angles['sza'].reshape(shape).mean(axis = (3, 1))
        self.saa = s2.angles['saa'].reshape(shape).mean(axis = (3, 1))
        self.vza = []
        self.vaa = []
        for band in self.s2_u_bands[:-2]:
            self.vza.append(s2.angles['vza'][band].reshape(shape).mean(axis = (3, 1)))
            self.vaa.append(s2.angles['vaa'][band].reshape(shape).mean(axis = (3, 1)))
        self.vza = np.array(self.vza) 
        self.vaa = np.array(self.vaa)
        self.raa = self.saa[None, ...] - self.vaa
        
        self.logger.info('Getting elevation.')
        ele_data       = reproject_data(self.global_dem, self.example_file, outputType= gdal.GDT_Float32).data
        mask           = ~np.isfinite(ele_data)
        ele_data       = np.ma.array(ele_data, mask = mask)/1000.
        self.elevation = ele_data.reshape((self.num_blocks, int(ele_data.shape[0] / self.num_blocks), \
                                           self.num_blocks, int(ele_data.shape[1] / self.num_blocks))).mean(axis=(3,1))

        self.logger.info('Getting pripors from ECMWF forcasts.')
        sen_time_str    = json.load(open(s2.s2_file_dir+'/tileInfo.json', 'r'))['timestamp']
        self.sen_time   = datetime.datetime.strptime(sen_time_str, u'%Y-%m-%dT%H:%M:%S.%fZ') 
        aot, tcwv, tco3 = np.array(self._read_cams(self.example_file)).reshape((3, self.num_blocks, \
                                   self.block_size, self.num_blocks, self.block_size)).mean(axis=(4, 2))
        self.aot        = aot.copy()
        self.aot[:]     = np.nanmean(aot)
        self.tco3       = tco3 * 46.698
        self.tcwv       = tcwv / 10. 
        self.logger.info('Mean values from ECMWF forcasts are: %.03f, %.03f, %.03f.'%(self.aot.mean(), self.tcwv.mean(), self.tco3.mean()))
        #self._get_ddv_aot(selected_img, tcwv, tco3, ele_data)
        self.aot_unc    = np.ones(self.aot.shape)  * 0.4
        self.tcwv_unc   = np.ones(self.tcwv.shape) * 0.1
        self.tco3_unc   = np.ones(self.tco3.shape) * 0.1
        
        self.logger.info('Trying to get the tcwv from the emulation of sen2cor look up table.')
        try:
            self._get_tcwv(selected_img, s2.angles['vza'], s2.angles['vaa'], s2.angles['sza'], s2.angles['saa'], ele_data)
        except:
            self.logger.warning('Getting tcwv from the emulation of sen2cor look up table failed, ECMWF TCWV is used.')
        
        self.logger.info('Applying PSF model.')
        if self.s2_psf is None:
            xstd, ystd, ang, xs, ys = self._get_psf(selected_img)
        else:
            xstd, ystd, ang, xs, ys = self.s2_psf
        # apply psf shifts without going out of the image extend  
        shifted_mask = np.logical_and.reduce(((self.Hx+int(xs)>=0),
                                              (self.Hx+int(xs)<self.full_res[0]), 
                                              (self.Hy+int(ys)>=0),
                                              (self.Hy+int(ys)<self.full_res[0])))
        
        self.Hx, self.Hy = self.Hx[shifted_mask]+int(xs), self.Hy[shifted_mask]+int(ys)
        #self.Lx, self.Ly = self.Lx[shifted_mask], self.Ly[shifted_mask]
        self.s2_boa      = self.s2_boa   [:, shifted_mask]
        self.s2_boa_qa   = self.s2_boa_qa[:, shifted_mask]

        self.logger.info('Getting the convolved TOA reflectance.')
        imgs = []
        for i, band in enumerate(self.s2_u_bands[:-2]):
            if selected_img[band].shape != self.full_res:
                imgs.append(self.repeat_extend(selected_img[band], shape = self.full_res))
            else:
                imgs.append(selected_img[band])
        self.bad_pixs = self.bad_pix[self.Hx, self.Hy]
        del selected_img; del s2.imgs; del s2.angles['vza']; del s2.angles['vaa']
        del s2.angles['sza']; del s2.angles['saa']; del s2.sza; del s2.saa; del s2
        ker = self.gaussian(xstd, ystd, ang) 
        f   = lambda img: signal.fftconvolve(img, ker, mode='same')[self.Hx, self.Hy]*0.0001 
        half = parmap(f,imgs[:3])
        self.s2_toa = np.array(half + parmap(f,imgs[3:]))
        border_mask = (self.Hx > 10830) | (self.Hx < 150) | (self.Hy > 10830) | (self.Hy < 150)
        points      = np.array([self.Hx[~border_mask], self.Hy[~border_mask]]).T
        self.s2_toa = np.array([griddata(points, self.s2_toa[i,~border_mask], \
                              (self.Hx, self.Hy), method='nearest') for i in range(self.s2_toa.shape[0])])
        del imgs
        # get the valid value masks
        qua_mask = np.all(self.s2_boa_qa <= self.qa_thresh, axis = 0)

        boa_mask = np.all(~self.s2_boa.mask,    axis = 0) &\
                   np.all(self.s2_boa >= 0.001, axis = 0) &\
                   np.all(self.s2_boa < 1,      axis = 0)

        toa_mask = np.all(self.s2_toa >= 0.0001,axis = 0) #&\
                   #np.all(self.s2_toa <  1.5,     axis = 0)
        self.s2_mask    = boa_mask & qua_mask & toa_mask
        self.Hx         = self.Hx          [self.s2_mask]
        self.Hy         = self.Hy          [self.s2_mask]
        self.s2_toa     = self.s2_toa   [:, self.s2_mask]
        self.s2_boa     = self.s2_boa   [:, self.s2_mask]
        self.s2_boa_unc = self.s2_boa_qa[:, self.s2_mask]
        
        self.logger.info('Solving...')
        tempm = np.zeros(self.full_res)
        tempm[self.Hx, self.Hy] = 1
        tempm = tempm.reshape(self.num_blocks, self.block_size, \
                              self.num_blocks, self.block_size).astype(int).sum(axis=(3,1))
        #mask = ~self.water_mask
        #self.mask = mask.reshape(self.num_blocks, self.block_size, \
        #                         self.num_blocks, self.block_size).astype(int).sum(axis=(3,1))
        self.mask  = tempm > 0. #& ((self.mask/((1.*self.block_size)**2)) > 0.) 
        #self.mask = binary_erosion(self.mask, structure=np.ones((3,3)).astype(bool))
        #self.mask[:] = True
        #self.mask[:2,  :] = False
        #self.mask[:, -2:] = False
        #self.mask[-2:, :] = False
        #self.mask[:,  :2] = False
        if self.mask.sum() ==0:
            self.logger.info('No valid value is found for retrieval of atmospheric parameters and priors are stored.')
            return np.array([[self.aot, self.tcwv, self.tco3], [self.aot_unc, self.tcwv_unc, self.tco3_unc]])
        else:
            self.aero = solving_atmo_paras(self.s2_boa, 
                                           self.s2_toa,
                                           self.sza, 
                                           self.vza,
                                           self.saa, 
                                           self.vaa,
                                           self.aot, 
                                           self.tcwv,
                                           self.tco3, 
                                           self.elevation,
                                           self.aot_unc,
                                           self.tcwv_unc,
                                           self.tco3_unc,
                                           self.s2_boa_unc,
                                           self.Hx, self.Hy,
                                           self.mask,
                                           self.full_res,
                                           self.aero_res,
                                           self.emus,
                                           self.band_indexs,
                                           self.boa_bands,
                                           gamma = 10.)
            solved = self.aero._optimization()
            return solved

    def _read_cams(self, example_file, parameters = ['aod550', 'tcwv', 'gtco3']):
        netcdf_file = datetime.datetime(self.sen_time.year, self.sen_time.month, \
                                        self.sen_time.day).strftime("%Y-%m-%d.nc")
        template    = 'NETCDF:"%s":%s'
        ind         = np.abs((self.sen_time.hour  + self.sen_time.minute/60. + \
                              self.sen_time.second/3600.) - np.arange(0,25,3)).argmin()
        sr         = osr.SpatialReference()
        sr.ImportFromEPSG(4326)
        proj       = sr.ExportToWkt()
        results = []
        for para in parameters:
            fname   = template%(self.cams_dir + '/' + netcdf_file, para)
            g       = gdal.Open(fname)
            g.SetProjection(proj)
            sub     = g.GetRasterBand(int(ind+1))
            offset  = sub.GetOffset()
            scale   = sub.GetScale()
            bad_pix = int(sub.GetNoDataValue())
            rep_g   = reproject_data(g, example_file, outputType= gdal.GDT_Float32).g
            data    = rep_g.GetRasterBand(int(ind+1)).ReadAsArray()
            data    = data*scale + offset
            mask    = (data == (bad_pix*scale + offset)) | np.isnan(data)
            if mask.sum()>=1:
                data[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), data[~mask])
            results.append(data)
        return results

    def solving_s2_aerosol(self,):
        
        self.s2_sensor  = 'MSI'
        self.logger = logging.getLogger('Sentinel 2 Atmospheric Correction')
        
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
        self.logger.propagate = False

        self.logger.info('Doing Sentinel %s tile: %s on %d-%02d-%02d.'%(self.satellite, self.s2_tile, self.year, self.month, self.day))
        self.block_size = int(self.aero_res/10)
        self.num_blocks = int(np.ceil(self.full_res[0]/self.block_size)) 
        ret             = self._s2_aerosol()
        self.solved     = ret[0].reshape(3, self.num_blocks, self.num_blocks)
        self.unc        = ret[1].reshape(3, self.num_blocks, self.num_blocks)
        self.unc[0, :2,  :] = self.unc[0].max()
        self.unc[1, :2,  :] = self.unc[1].max()
        self.unc[2, :2,  :] = self.unc[2].max()
        self.unc[0, :, -2:] = self.unc[0].max()
        self.unc[1, :, -2:] = self.unc[1].max()
        self.unc[2, :, -2:] = self.unc[2].max()
        self.unc[0, -2:, :] = self.unc[0].max()
        self.unc[1, -2:, :] = self.unc[1].max()
        self.unc[2, -2:, :] = self.unc[2].max()
        self.unc[0, :,  :2] = self.unc[0].max()
        self.unc[1, :,  :2] = self.unc[1].max()
        self.unc[2, :,  :2] = self.unc[2].max()
        self.logger.info('Finished retrieval and saving them into local files.')
        self._example_g = gdal.Open(self.s2_file_dir+'/B04.jp2')
        para_names      = 'aot', 'tcwv', 'tco3', 'aot_unc', 'tcwv_unc', 'tco3_unc'
        high_aod        = (self.solved[0] > 0.5) & \
                          (self.solved[0] < 1.7)                                                                                      
        self.solved[0][high_aod] = self.solved[0][high_aod] + \
                                  (self.solved[0][high_aod] - 0.5) * 0.08
        arrays          = list(self.solved ) + list(self.unc)
        name_arrays     = zip(para_names, arrays)
        ret = parmap(self._save_posterior, name_arrays)
        self.post_aot,     self.post_tcwv,     self.post_tco3, \
        self.post_aot_unc, self.post_tcwv_unc, self.post_tco3_unc = ret 

    def _save_posterior(self, name_array):
        name, array = name_array
        if (self.mask.sum() > 0) & ('unc' not in name):
            self.mask[:2,  :] = False
            self.mask[:, -2:] = False
            self.mask[:,  :2] = False
            self.mask[-2:, :] = False 
            array = griddata(np.array(np.where(self.mask)).T, array[self.mask], \
                                     (np.repeat(range(self.num_blocks), self.num_blocks).reshape(self.num_blocks, self.num_blocks), \
                                      np.tile  (range(self.num_blocks), self.num_blocks).reshape(self.num_blocks, self.num_blocks)), method='nearest')
            #array[~self.mask] = np.nanmean(array[self.mask])
        xmin, ymax  = self._example_g.GetGeoTransform()[0], \
                      self._example_g.GetGeoTransform()[3]
        projection  = self._example_g.GetProjection()
        xres, yres = self.block_size*10, self.block_size*10
        geotransform = (xmin, xres, 0, ymax, 0, -yres)
        nx, ny = self.num_blocks, self.num_blocks
        outputFileName = self.s2_file_dir + '/%s.tif'%name
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Float32, options=["TILED=YES", "COMPRESS=DEFLATE"])
        dst_ds.SetGeoTransform(geotransform)   
        dst_ds.SetProjection(projection) 
        dst_ds.GetRasterBand(1).WriteArray(array)
        dst_ds.FlushCache()                     
        dst_ds = None
        return array

if __name__ == "__main__":
    aero = solve_aerosol( 2017, 9, 12, mcd43_dir = '/data/nemesis/MCD43/', s2_toa_dir = '/data/nemesis/S2_data/',\
                                      emus_dir = '/home/ucfafyi/DATA/Multiply/emus/', s2_tile='50SMG', s2_psf=None)
    aero.solving_s2_aerosol()
