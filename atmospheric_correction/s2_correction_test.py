#/usr/bin/env python
import os
import sys
sys.path.insert(0,'util')
import gdal
import numpy as np
from numpy import clip, uint8
from glob import glob
import logging
#from Py6S import *
try:
    import cPickle as pkl
except:
    import pickle as pkl
from multi_process import parmap
from grab_s2_toa import read_s2
#from aerosol_solver import solve_aerosol
from reproject import reproject_data
import warnings
warnings.filterwarnings("ignore")

class atmospheric_correction(object):
    '''
    A class doing the atmospheric coprrection with the input of TOA reflectance
    angles, elevation and emulators of 6S from TOA to surface reflectance.
    '''
    def __init__(self,
                 year, 
                 month, 
                 day,
                 s2_tile,
                 acquisition = '0',
                 s2_toa_dir  = '/home/ucfafyi/DATA/S2_MODIS/s_data/',
                 global_dem  = '/home/ucfafyi/DATA/Multiply/eles/global_dem.vrt',
                 emus_dir    = '/home/ucfafyi/DATA/Multiply/emus/',
                 satellite   = 'S2A',
                 reconstruct_s2_angle = True
                 ):              
        
        self.year        = year
        self.month       = month
        self.day         = day
        self.s2_tile     = s2_tile
        self.acquisition = acquisition
        self.s2_toa_dir  = s2_toa_dir
        self.global_dem  = global_dem
        self.emus_dir    = emus_dir
        self.satellite   = satellite
        self.sur_refs     = {}
        self.reconstruct_s2_angle = reconstruct_s2_angle
        self.logger = logging.getLogger('Sentinel 2 Atmospheric Correction')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:       
            ch = logging.StreamHandler()
            ch.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
    def _load_xa_xb_xc_emus(self,):
        xap_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xap_%s.pkl'%(self.s2_sensor, self.satellite))[0]
        xbp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xbp_%s.pkl'%(self.s2_sensor, self.satellite))[0]
        xcp_emu = glob(self.emus_dir + '/isotropic_%s_emulators_*_xcp_%s.pkl'%(self.s2_sensor, self.satellite))[0]
        if sys.version_info >= (3,0):
            f = lambda em: pkl.load(open(em, 'rb'), encoding = 'latin1')
        else:
            f = lambda em: pkl.load(open(em, 'rb'))
        self.xap_emus, self.xbp_emus, self.xcp_emus = parmap(f, [xap_emu, xbp_emu, xcp_emu])    

    def atmospheric_correction(self,):

        self.logger.propagate = False
        self.s2_sensor = 'MSI'
        self.logger.info('Loading emulators.')
        self._load_xa_xb_xc_emus()
        self.s2   = read_s2(self.s2_toa_dir, self.s2_tile, \
                            self.year, self.month, self.day, bands=None, acquisition = self.acquisition)
        self.logger.info('Reading in the reflectance.')
        all_refs = self.s2.get_s2_toa()
        self.logger.info('Reading in the angles')
        self.s2.get_s2_angles(self.reconstruct_s2_angle)
        self.sza,self.saa = self.s2.angles['sza'], self.s2.angles['saa']
        
        self.logger.info('Doing 10 meter bands')
        self.ref = np.array([all_refs[band].astype(float)/10000. for band \
                                      in ['B02', 'B03', 'B04', 'B08']])
        self.vza = np.array([self.s2.angles['vza'][band] for band
                                      in ['B02', 'B03', 'B04', 'B08']])
        self.vaa = np.array([self.s2.angles['vaa'][band] for band
                                      in ['B02', 'B03', 'B04', 'B08']])
        self.logger.info('Getting control variables for 10 meters bands.')

        self.aot, self.tcwv, self.tco3, self.aot_unc, self.tcwv_unc, self.tco3_unc, self.ele = self.get_control_variables('B04')
        self._block_size = 3660
        self._num_blocks = int(10980/self._block_size)
        self._mean_size  = 60
        self.band_indexs = [1, 2, 3, 7]        
        self.logger.info('Fire correction and splited into %d blocks.'%self._num_blocks**2)
        self.fire_correction(self.ref, self.sza, self.vza,\
                             self.saa, self.vaa, self.aot,\
                             self.tcwv, self.tco3, self.aot_unc, \
                             self.tcwv_unc, self.tco3_unc, self.ele,\
                             self.band_indexs)     
        
        self.mask         = (self.ref[[2,1,0], ...] >= 0)
        self.scale        = 0.25 
        self.toa_rgb = clip(self.ref[[2,1,0], ...].transpose(1,2,0)*255/self.scale, 0., 255.).astype(uint8)
        self.boa_rgb = clip(self.boa         [[2,1,0], ...].transpose(1,2,0)*255/self.scale, 0., 255.).astype(uint8)
        self._save_rgb(self.toa_rgb, 'TOA_RGB.tif', self.s2.s2_file_dir+'/B04.jp2')
        self._save_rgb(self.boa_rgb, 'BOA_RGB.tif', self.s2.s2_file_dir+'/B04.jp2')
        
        gdal.Translate(self.s2.s2_file_dir+'/TOA_overview.jpg', self.s2.s2_file_dir+'/TOA_RGB.tif', \
                       format = 'JPEG', widthPct=50, heightPct=50, resampleAlg='cubic' ).FlushCache()
        gdal.Translate(self.s2.s2_file_dir+'/BOA_overview.jpg', self.s2.s2_file_dir+'/BOA_RGB.tif', \
                       format = 'JPEG', widthPct=50, heightPct=50, resampleAlg='cubic' ).FlushCache()

        del self.ref; del self.vza; del self.vaa; del self.aot; \
        del self.tcwv; del self.tco3; del self.ele
        
        #self.sur_refs.update(dict(zip(['B02', 'B03', 'B04', 'B08'], self.boa)))
        self._save_img(self.boa, ['B02', 'B03', 'B04', 'B08']); del self.boa 
          
        self.logger.info('Doing 20 meter bands')
        self.ref = np.array([all_refs[band].astype(float)/10000. for band \
                                      in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']])

        self.vza = np.array([self.s2.angles['vza'][band] for band \
                                      in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']]).reshape(6, 5490, 2, 5490, 2).mean(axis=(4,2))
        self.vaa = np.array([self.s2.angles['vaa'][band] for band \
                                      in ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']]).reshape(6, 5490, 2, 5490, 2).mean(axis=(4,2))

        self.sza = self.sza.reshape(5490, 2, 5490, 2).mean(axis=(3, 1))
        self.saa = self.saa.reshape(5490, 2, 5490, 2).mean(axis=(3, 1))


        self.logger.info('Getting control variables for 20 meters bands.')
        self.aot, self.tcwv, self.tco3,\
                           self.ele = self.get_control_variables('B05')
        self._block_size = 1830
        self._num_blocks = int(5490/self._block_size)
        self._mean_size  = 30
        self.band_indexs = [4, 5, 6, 8, 11, 12]
        self.logger.info('Fire correction and splited into %d blocks.'%self._num_blocks**2)
        self.fire_correction(self.ref, self.sza, self.vza,\
                             self.saa, self.vaa, self.aot,\
                             self.tcwv, self.tco3, self.ele,\
                             self.band_indexs)

        del self.ref; del self.vza; del self.vaa;  del self.sza
        del self.saa; del self.aot; del self.tcwv; del self.tco3; del self.ele

        #self.sur_refs.update(dict(zip(['B05', 'B06', 'B07', 'B8A', 'B11', 'B12'], self.boa)))
        self._save_img(self.boa, ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']); del self.boa


        self.logger.info('Doing 60 meter bands')
        self.ref = np.array([all_refs[band].astype(float)/10000. for band \
                                      in ['B01', 'B09', 'B10']])
        self.vza = np.array([self.s2.angles['vza'][band] for band \
                                      in ['B01', 'B09', 'B10']]).reshape(3, 1830, 6, 1830, 6).mean(axis=(4, 2))
        self.vaa = np.array([self.s2.angles['vaa'][band] for band \
                                      in ['B01', 'B09', 'B10']]).reshape(3, 1830, 6, 1830, 6).mean(axis=(4, 2))
        self.sza = self.sza.reshape(1830, 6, 1830, 6).mean(axis=(3,1))
        self.saa = self.saa.reshape(1830, 6, 1830, 6).mean(axis=(3,1))

        self.logger.info('Getting control variables for 60 meters bands.')
        self.aot, self.tcwv, self.tco3,\
                           self.ele = self.get_control_variables('B09')
        self._block_size = 610
        self._num_blocks = int(1830/self._block_size)
        self._mean_size  = 10
        self.band_indexs = [0, 9, 10]
        self.logger.info('Fire correction and splited into %d blocks.'%self._num_blocks**2)
        self.fire_correction(self.ref, self.sza, self.vza,\
                             self.saa, self.vaa, self.aot,\
                             self.tcwv, self.tco3, self.ele,\
                             self.band_indexs)

        del self.ref; del self.vza; del self.vaa;  del self.sza
        del self.saa; del self.aot; del self.tcwv; del self.tco3; del self.ele

        #self.sur_refs.update(dict(zip(['B01', 'B09', 'B10'], self.boa)))
        self._save_img(self.boa, ['B01', 'B09', 'B10']); del self.boa
        del all_refs; del self.s2.selected_img; del self.s2.angles 
        self.logger.info('Done!')

    def _save_rgb(self, rgb_array, name, source_image):
        g            = gdal.Open(source_image)
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        nx, ny       = rgb_array.shape[:2]
        outputFileName = self.s2.s2_file_dir+'/%s'%name
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 3, gdal.GDT_Byte, options=["TILED=YES", "COMPRESS=JPEG"])
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        dst_ds.GetRasterBand(1).WriteArray(rgb_array[:,:,0])
        dst_ds.GetRasterBand(1).SetScale(self.scale)
        dst_ds.GetRasterBand(2).WriteArray(rgb_array[:,:,1])
        dst_ds.GetRasterBand(2).SetScale(self.scale)
        dst_ds.GetRasterBand(3).WriteArray(rgb_array[:,:,2])
        dst_ds.GetRasterBand(3).SetScale(self.scale)
        dst_ds.FlushCache()
        dst_ds = None

    def rgb_scale_offset(arr):
        arrmin = arr.mean() - 3*arr.std()
        arrmax = arr.mean() + 3*arr.std()
        arrlen = arrmax-arrmin
        scale  = arrlen/255.0
        offset = arrmin
        return offset, scale

    def _save_img(self, refs, bands):
        g            = gdal.Open(self.s2.s2_file_dir+'/%s.jp2'%bands[0])
        projection   = g.GetProjection()
        geotransform = g.GetGeoTransform()
        bands_refs   = zip(bands, refs)
        f            = lambda band_ref: self._save_band(band_ref, projection = projection, geotransform = geotransform)
        parmap(f, bands_refs)
      
    def _save_band(self, band_ref, projection, geotransform):
        band, ref = band_ref
        nx, ny = ref.shape
        outputFileName = self.s2.s2_file_dir+'/%s_sur.tif'%band
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, ny, nx, 1, gdal.GDT_Int16, options=["TILED=YES", "COMPRESS=DEFLATE"])
        dst_ds.SetGeoTransform(geotransform)
        dst_ds.SetProjection(projection)
        sur = ref * 10000
        sur[~(sur>=0)] = -9999
        sur = sur.astype(np.int16)
        dst_ds.GetRasterBand(1).SetNoDataValue(-9999)
        dst_ds.GetRasterBand(1).WriteArray(sur)
        dst_ds.FlushCache()                  
        dst_ds = None

    def get_control_variables(self, target_band):
        fnames   = [self.s2.s2_file_dir+'/%s.tif'%i for i in ['aot', 'tcwv', 'tco3', 'aot_unc', 'tcwv_unc', 'tco3_unc']]
        fnames.append(self.global_dem)
        f        = lambda fname: reproject_data(fname, self.s2.s2_file_dir + '/B04.jp2', outputType= gdal.GDT_Float32, xRes=600, yRes=600).data
        aot, tcwv, tco3, aot_unc, tcwv_unc, tco3_unc, ele = parmap(f, fnames)
        mask = ~np.isfinite(ele)
        ele[mask] = np.interp(np.flatnonzero(mask), \
                              np.flatnonzero(~mask), ele[~mask])
        return aot, tcwv, tco3, aot_unc, tcwv_unc, tco3_unc, ele.astype(float)


    def fire_correction(self, toa, sza, vza, saa, vaa, aot, tcwv, tco3, aot_unc, tcwv_unc, tco3_unc, elevation, band_indexs):
        self._toa         = toa
        self._sza         = sza
        self._vza         = vza
        self._saa         = saa
        self._vaa         = vaa
        self._aot         = aot
        self._tcwv        = tcwv
        self._tco3        = tco3
        self._aot_unc     = aot_unc
        self._tcwv_unc    = tcwv_unc
        self._tco3_unc    = tco3_unc
        self._elevation   = elevation
        self._band_indexs = band_indexs
        rows              = np.repeat(np.arange(self._num_blocks), self._num_blocks)
        columns           = np.tile(np.arange(self._num_blocks), self._num_blocks)
        blocks            = zip(rows, columns)
        #self._s2_block_correction_emus_xa_xb_xc([1, 1])
        ret = parmap(self._s2_block_correction_emus_xa_xb_xc, blocks)
        #ret = parmap(self._s2_block_correction_6s, blocks)
        #ret               = parmap(self._s2_block_correction_emus, blocks) 
        self.boa = np.array([i[2] for i in ret]).reshape(self._num_blocks, self._num_blocks, toa.shape[0], \
                             self._block_size, self._block_size).transpose(2,0,3,1,4).reshape(toa.shape[0], \
                             self._num_blocks*self._block_size, self._num_blocks*self._block_size)
                        
        del self._toa; del self._sza; del self._vza;  del self._saa
        del self._vaa; del self._aot; del self._tcwv; del self._tco3; del self._elevation

    def _s2_block_correction_emus_xa_xb_xc(self, block):
        i, j        = block
        self.logger.info('Block %03d--%03d'%(i+1,j+1))
        slice_x     = slice(i*self._block_size,(i+1)*self._block_size, 1)
        slice_y     = slice(j*self._block_size,(j+1)*self._block_size, 1)

        toa         = self._toa    [:,slice_x,slice_y]
        vza         = self._vza    [:,slice_x,slice_y]*np.pi/180.
        vaa         = self._vaa    [:,slice_x,slice_y]*np.pi/180.
        sza         = self._sza      [slice_x,slice_y]*np.pi/180.
        saa         = self._saa      [slice_x,slice_y]*np.pi/180.
        tcwv        = self._tcwv     [slice_x,slice_y]
        tco3        = self._tco3     [slice_x,slice_y]
        aot         = self._aot      [slice_x,slice_y]
        elevation   = self._elevation[slice_x,slice_y]/1000.
        mask        = self.mask      [slice_x,slice_y]
        mean_mask   = self._block_mean(mask.astype(float), self._mean_size).ravel() > 0
        corfs       = []
        xap, xap_dH = []
        xbp, xbp_dH = []
        xcp, xcp_dH = []
        xps = [xap, xbp, xcp]
        xhs = [xap_dH, xbp_dH, xcp_dH]
        for bi, band in enumerate(self._band_indexs):    
            p        = np.array([self._block_mean(i, self._mean_size).ravel() for i in [np.cos(sza), \
                                 np.cos(vza[bi]), np.cos(saa - vaa[bi]), aot, tcwv, tco3, elevation]])
            mp       = p[:, mask]
            for _, emu in enumerate([self.xap_emus, self.xbp_emus, self.xcp_emus]):
                temp1, temp2 = np.zeros_like(p[0]), np.zeros_like(p[0].shape + (3,))
                H, dH        = emu[band].predict(mp.T, do_unc=False)
                temp1[mask]  = H
                temp2[mask]  = np.array(dH)[:, 3:6]
                temp1        = temp1.reshape(self._block_size//self._mean_size, self._block_size//self._mean_size)
                temp1        = np.repeat(np.repeat(temp1, self._mean_size, axis=0), self._mean_size, axis=1)
                if _ == 0:
                    temp1[temp1==0]  = 1.
                temp2        = temp2.reshape(self._block_size//self._mean_size, self._block_size//self._mean_size)
                temp2        = np.repeat(np.repeat(temp2, self._mean_size, axis=0), self._mean_size, axis=1)
                xps[_].append(temp1)
                xhs[_].append(temp2)

        xap, xbp, xcp          = np.array(xps)
        xap_dH, xbp_dH, xcp_dH = np.array(xhs)

        y            = xap * toa - xbp
        boa          = y / (1 + xcp * y)           

        dH           = -1 * (-toa[...,None] * xap_dH - \
                        2 * toa[...,None] * xap_H[...,None] * xbp_H[...,None] * xcp_dH + \
                        toa[...,None]**2 * xap_H[...,None]**2 * xcp_dH + \
                        xbp_dH + \
                        xbp_H[...,None]**2 * xcp_dH) / \
                        (toa[...,None] * xap_H[...,None] * xcp_H[...,None] - \
                        xbp_H[...,None] * xcp_H[...,None] + 1)**2
        
        aot_dH, wv_dH, o3_dH = [ dH[:,:,i] for i in range(3)]
        return [i, j, boa]

    def _block_mean(self, data, block_size):
        x_size, y_size = data.shape
        x_blocks       = x_size//block_size
        y_blocks       = y_size//block_size
        data           = data.copy().reshape(x_blocks, block_size, y_blocks, block_size)        
        small_data     = np.nanmean(data, axis=(3,1))
        return small_data

if __name__=='__main__':
    atmo_cor = atmospheric_correction(2017, 12, 15, '32UPU', s2_toa_dir='/data/nemesis/S2_data/')
    atmo_cor.atmospheric_correction()
