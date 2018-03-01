#/usr/bin/env python
import gdal
import os
import sys
import copy
sys.path.insert(0, 'util')
sys.path.insert(0, './')
import xml.etree.ElementTree as ET
import numpy as np
import pickle as pkl
from multiprocessing import Pool
from glob import glob
from scipy.interpolate import griddata
from scipy.signal import fftconvolve
#import subprocess
from s2a_angle_bands_mod import s2a_angle
from reproject import reproject_data
from multi_process import parmap
from skimage.morphology import disk, binary_dilation, binary_erosion

def read_s2_band(fname):
        g = gdal.Open(fname)
        if g is None:
            raise IOError
        else:
            return g.ReadAsArray()


class read_s2(object):
    '''
    A class reading S2 toa reflectance, taken the directory, date and bands needed,
    It will read in the cloud mask as well, if no cloud.tiff, then call the classification
    algorithm to get the cloud mask and save it.
    '''
    def __init__(self, 
                 s2_toa_dir,
                 s2_tile, 
                 year, month, day,
                 acquisition = '0',
                 bands   = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A']):
        self.s2_toa_dir  = s2_toa_dir
        self.s2_tile     = s2_tile
        self.year        = year
        self.month       = month
        self.day         = day
        self.bands       = bands # selected bands
        self.s2_bands    = 'B01', 'B02', 'B03','B04','B05' ,'B06', 'B07', \
                           'B08','B8A', 'B09', 'B10', 'B11', 'B12' #all bands
        self.s2_file_dir = os.path.join(self.s2_toa_dir, self.s2_tile[:-3],\
                                        self.s2_tile[-3], self.s2_tile[-2:],\
                                        str(self.year), str(self.month), str(self.day), acquisition)
        self.selected_img = None
        self.done         = False

    def _read_all(self, done = False):
        fname     = [self.s2_file_dir+'/%s.jp2'%i for i in self.s2_bands]
        pool      = Pool(processes=len(fname))
        ret       = pool.map(read_s2_band, fname)    
        self.imgs = dict(zip(self.s2_bands, ret))
        self.done = True

    def get_s2_toa(self,vrt = False):
        self._read_all(self.done)
        if self.bands is None:
            self.bands = self.s2_bands
        selc_imgs = [self.imgs[band] for band in self.bands] 
        return dict(zip(self.bands, selc_imgs))

    def get_s2_cloud(self,):
        if len(glob(self.s2_file_dir+'/cloud.tif'))==0:
            self._read_all(self.done)  
            cloud_bands  = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
            ratio        = [ 6,     1,     1,     2,     1,     2,     6,     6,     2,     2]
            
            refs         = [(self.imgs[band]/10000.).reshape(1830, 6//ratio[i], 1830,  6//ratio[i]) for i, band in enumerate(cloud_bands)]
            refs         = np.array([ref.sum(axis=(3,1)) / (ref>=0.0001).sum(axis=(3,1)) for ref in refs])
            classifier   = pkl.load(open('./data/sen2cloud_detector.pkl', 'rb'))
            mask         = np.all(refs >= 0.0001, axis=0)
            cloud_probs  = classifier.predict_proba(refs[:, mask].T)[:,1]
            cloud        = np.zeros((1830, 1830))
            cloud[mask]  = cloud_probs
            cloud_mask   = cloud > 0.75
            cloud_mask   = binary_erosion (cloud_mask, disk(2))
            self.cloud   = binary_dilation(cloud_mask, disk(3))
            self.cloud   = np.repeat(np.repeat(self.cloud,6, axis = 0), 6, axis = 1)
            g            = gdal.Open(self.s2_file_dir+'/B01.jp2')
            driver       = gdal.GetDriverByName('GTiff')
            g1           = driver.Create(self.s2_file_dir+'/cloud.tif', \
                                         g.RasterXSize, g.RasterYSize, 1, \
                                         gdal.GDT_Byte,  options=["TILED=YES", "COMPRESS=DEFLATE"])
            g1.SetGeoTransform(g.GetGeoTransform())
            g1.SetProjection  (g.GetProjection()  )
            g1.GetRasterBand(1).WriteArray((cloud * 100).astype(int))
            g1=None; g=None
        else:
            cloud = gdal.Open(self.s2_file_dir+\
                             '/cloud.tif').ReadAsArray()
            cloud_mask   = cloud > 74 #rounding issue                           
            cloud_mask   = binary_erosion (cloud_mask, disk(2))     
            self.cloud   = binary_dilation(cloud_mask, disk(3)) 
            self.cloud   = np.repeat(np.repeat(self.cloud,6, axis = 0), 6, axis = 1)
        try:
            mask = self.imgs['B04'] >= 1.
        except:
            mask = gdal.Open(self.s2_file_dir + '/B04.jp2').ReadAsArray() >= 1.
        self.cloud_cover = 1. * self.cloud.sum() / mask.sum()

        return self.cloud
        
    def get_s2_angles(self, reconstruct = False):
        if len(glob(self.s2_file_dir + '/angles/VAA_VZA_*.tif')) == 1:
            pass
        else:
            self._get_s2_angles(reconstruct=reconstruct)

    def _get_s2_angles(self, reconstruct = True):

        tree = ET.parse(self.s2_file_dir+'/metadata.xml')
        root = tree.getroot()
        #Sun_Angles_Grid
        saa =[]
        sza =[]
        msz = []
        msa = []
        #Viewing_Incidence_Angles_Grids
        vza = {}
        vaa = {}
        mvz = {}
        mva = {}
        for child in root:
            for j in child:
                for k in j.findall('Sun_Angles_Grid'):
                    for l in k.findall('Zenith'):
                        for m in l.findall('Values_List'):
                            for x in m.findall('VALUES'):
                                sza.append(x.text.split())

                    for n in k.findall('Azimuth'):
                        for o in n.findall('Values_List'):
                            for p in o.findall('VALUES'):
                                saa.append(p.text.split())
                for ms in j.findall('Mean_Sun_Angle'):
                    self.msz = float(ms.find('ZENITH_ANGLE').text)
                    self.msa = float(ms.find('AZIMUTH_ANGLE').text)

                for k in j.findall('Viewing_Incidence_Angles_Grids'):
                    for l in k.findall('Zenith'):
                        for m in l.findall('Values_List'):
                            vza_sub = []
                            for x in m.findall('VALUES'):
                                vza_sub.append(x.text.split())
                            bi, di, angles = k.attrib['bandId'], \
                                             k.attrib['detectorId'], np.array(vza_sub).astype(float)
                            vza[(int(bi),int(di))] = angles

                    for n in k.findall('Azimuth'):
                        for o in n.findall('Values_List'):
                            vaa_sub = []
                            for p in o.findall('VALUES'):
                                vaa_sub.append(p.text.split())
                            bi, di, angles = k.attrib['bandId'],\
                                             k.attrib['detectorId'], np.array(vaa_sub).astype(float)
                            vaa[(int(bi),int(di))] = angles

                for mvia in j.findall('Mean_Viewing_Incidence_Angle_List'):
                    for i in mvia.findall('Mean_Viewing_Incidence_Angle'):
                        mvz[int(i.attrib['bandId'])] = float(i.find('ZENITH_ANGLE').text)
                        mva[int(i.attrib['bandId'])] = float(i.find('AZIMUTH_ANGLE').text)
        sza  = np.array(sza).astype(float)
        saa  = np.array(saa).astype(float)
        saa[saa>180] = saa[saa>180] - 360
        mask = np.isnan(sza)
        sza  = griddata(np.array(np.where(~mask)).T, sza[~mask], \
                       (np.repeat(range(23), 23).reshape(23,23), \
                        np.tile  (range(23), 23).reshape(23,23)), method='nearest')
        mask = np.isnan(saa) 
        saa  = griddata(np.array(np.where(~mask)).T, saa[~mask], \
                       (np.repeat(range(23), 23).reshape(23,23), \
                        np.tile  (range(23), 23).reshape(23,23)), method='nearest') 
        self.saa, self.sza = np.repeat(np.repeat(np.array(saa), 500, axis = 0), 500, axis = 1)[:10980, :10980], \
                             np.repeat(np.repeat(np.array(sza), 500, axis = 0), 500, axis = 1)[:10980, :10980]

        g                = gdal.Open(self.s2_file_dir + '/B04.jp2')
        geo              = g.GetGeoTransform()
        projection       = g.GetProjection()
        geotransform     = (geo[0], 5000, geo[2], geo[3], geo[4], -5000)
        outputFileName   = self.s2_file_dir + '/angles/SAA_SZA.tif'
        if os.path.exists(outputFileName):
            os.remove(outputFileName)
        dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, 23, 23, 2, gdal.GDT_Int16, options=["TILED=YES", "COMPRESS=DEFLATE"])
        dst_ds.SetGeoTransform(geotransform)   
        dst_ds.SetProjection(projection) 
        dst_ds.GetRasterBand(1).WriteArray((saa * 100).astype(int))
        dst_ds.GetRasterBand(2).WriteArray((sza * 100).astype(int))
        dst_ds.FlushCache()                     
        dst_ds, g = None, None

        dete_id = np.unique([i[1] for i in vaa.keys()])
        band_id = range(13)
        bands_vaa = []
        bands_vza = []
        for i in band_id:
            band_vaa = np.zeros((23,23))
            band_vza = np.zeros((23,23))
            band_vaa[:] = np.nan
            band_vza[:] = np.nan
            for j in dete_id:
                    try:
                        good = ~np.isnan(vaa[(i,j)])
                        band_vaa[good] = vaa[(i,j)][good]
                        good = ~np.isnan(vza[(i,j)])
                        band_vza[good] = vza[(i,j)][good]
                    except:
                        pass 
            bands_vaa.append(band_vaa)
            bands_vza.append(band_vza)
        bands_vaa, bands_vza = np.array(bands_vaa), np.array(bands_vza)
        vaa  = {}; vza  = {}
        mva_ = {}; mvz_ = {}
        for i, band in enumerate(self.s2_bands):
            vaa[band]  = bands_vaa[i] if not np.isnan(bands_vaa[i]).all() else np.nanmean(bands_vaa, axis=0)
            vza[band]  = bands_vza[i] if not np.isnan(bands_vza[i]).all() else np.nanmean(bands_vza, axis=0)
            try:
                mva_[band] = mva[i]
                mvz_[band] = mvz[i]
            except:
                mva_[band] = np.nanmean([mva[z] for z in mva.keys()])
                mvz_[band] = np.nanmean([mvz[z] for z in mvz.keys()])

        bands = self.s2_bands
        self.vza = {}; self.vaa = {}
        self.mvz = {}; self.mva = {}
        for band in bands:
            mask  = np.isnan(vza[band])
            if (~mask).sum() == 0:
                g_vza    = np.zeros((23, 23)) 
                m_vza    = np.nanmean([mvz_[_] for _ in bands])
                g_vaa[:] = m_vza      if not np.isnan(m_vza)      else 0
                g_vza[:] = mvz_[band] if not np.isnan(mvz_[band]) else g_vaa[0, 0]
                self.mvz[band]   = m_vza      if not np.isnan(m_vza)      else 0
                self.mvz[band]   = mvz_[band] if not np.isnan(mvz_[band]) else self.mvz[band] 
            else: 
                g_vza = griddata(np.array(np.where(~mask)).T, vza[band][~mask], \
                                (np.repeat(range(23), 23).reshape(23,23), \
                                 np.tile  (range(23), 23).reshape(23,23)), method='nearest')
            mask  = np.isnan(vaa[band]) 
            if (~mask).sum() == 0:
                g_vaa    = np.zeros((23, 23))
                m_vaa    = np.nanmean(mva_.values())
                g_vaa[:] = m_vaa      if not np.isnan(m_vaa)      else 0
                g_vaa[:] = mva_[band] if not np.isnan(mva_[band]) else g_vaa[0, 0]
                self.mva[band]   = m_vaa      if not np.isnan(m_vaa)      else 0
                self.mva[band]   = mva_[band] if not np.isnan(mva_[band]) else self.mva[band]
            else:        
                g_vaa = griddata(np.array(np.where(~mask)).T, vaa[band][~mask], \
                                (np.repeat(range(23), 23).reshape(23,23), \
                                 np.tile  (range(23), 23).reshape(23,23)), method='nearest') 
                self.mvz[band]   = mvz_[band] 
                self.mva[band]   = mva_[band]

            # seems like scene containing positive and negative vaa
            # is more likely to have the wrong vaa angle and the mean value is used
            if not ((g_vaa <= 0).all() or (g_vaa >= 0).all()):
                reconstruct        = False # no need for reconstruct anymore
                g_vaa[:]           = self.mva[band] 

            g_vaa[g_vaa>180] = g_vaa[g_vaa>180] - 360
            self.vza[band]   = np.repeat(np.repeat(g_vza, 500, axis = 0), 500, axis = 1)[:10980, :10980]
            self.vaa[band]   = np.repeat(np.repeat(g_vaa, 500, axis = 0), 500, axis = 1)[:10980, :10980]

            g                = gdal.Open(self.s2_file_dir + '/B04.jp2')
            geo              = g.GetGeoTransform()
            projection       = g.GetProjection()
            geotransform     = (geo[0], 5000, geo[2], geo[3], geo[4], -5000)
            outputFileName   = self.s2_file_dir + '/angles/VAA_VZA_%s.tif'%band
            if os.path.exists(outputFileName):
                os.remove(outputFileName)
            dst_ds = gdal.GetDriverByName('GTiff').Create(outputFileName, 23, 23, 2, gdal.GDT_Int16, options=["TILED=YES", "COMPRESS=DEFLATE"])
            dst_ds.SetGeoTransform(geotransform)   
            dst_ds.SetProjection(projection) 
            dst_ds.GetRasterBand(1).WriteArray((g_vaa * 100).astype(int))
            dst_ds.GetRasterBand(2).WriteArray((g_vza * 100).astype(int))
            dst_ds.FlushCache()                     
            dst_ds, g = None, None
        self.angles = {'sza':self.sza, 'saa':self.saa, 'msz':self.msz, 'msa':self.msa,\
                           'vza':self.vza, 'vaa': self.vaa, 'mvz':self.mvz, 'mva':self.mva}

        if reconstruct:
            try:
                #if len(glob(self.s2_file_dir + '/angles/VAA_VZA_*.tif')) == 13:
                #    pass
                #else:
		    #print 'Reconstructing Sentinel 2 angles...'
                s2a_angle(self.s2_file_dir+'/metadata.xml')
		    #subprocess.call(['python', './python/s2a_angle_bands_mod.py', \
		    #                  self.s2_file_dir+'/metadata.xml',  '10'])
            
                if self.bands is None:
                    bands = self.s2_bands
                else:
                    bands = self.bands
                self.vaa = {}; self.vza = {}
                fname = [self.s2_file_dir+'/angles/VAA_VZA_%s.tif'%band for band in bands]
                if len(glob(self.s2_file_dir + '/angles/VAA_VZA_*.tif')) == 13:
                    f = lambda fn: reproject_data(fn, self.s2_file_dir+'/B04.jp2', outputType= gdal.GDT_Float32).data
                    ret = parmap(f, fname)
                    for i,angs in enumerate(ret):
		        #angs[0][angs[0]<0] = (36000 + angs[0][angs[0]<0])
                        angs = angs.astype(float)/100.
                        self.vaa[bands[i]] = angs[0]
                        self.vza[bands[i]] = angs[1]
                    self.angles = {'sza':self.sza, 'saa':self.saa, 'msz':self.msz, 'msa':self.msa,\
                                   'vza':self.vza, 'vaa': self.vaa, 'mvz':self.mvz, 'mva':self.mva}
                else:
                    print ('Reconstruct failed and original angles are used.')
            except:
                print('Reconstruct failed and original angles are used.')
if __name__ == '__main__':
    
    s2 = read_s2('/store/S2_data/', '50SMH', \
                  2017, 10, 12, bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'] )
    '''
    s2.selected_img = s2.get_s2_toa() 
    '''
    cm = s2.get_s2_cloud()
    #s2.get_s2_angles()
