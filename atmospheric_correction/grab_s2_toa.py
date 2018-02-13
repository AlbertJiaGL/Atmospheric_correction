#/usr/bin/env python
import gdal
import os
import sys
import copy
sys.path.insert(0, 'util')
sys.path.insert(0, './')
import xml.etree.ElementTree as ET
import numpy as np
from multiprocessing import Pool
from glob import glob
from scipy.interpolate import griddata
from scipy.signal import fftconvolve
#import subprocess
from s2a_angle_bands_mod import s2a_angle
from reproject import reproject_data
from multi_process import parmap
from skimage.morphology import disk, dilation

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
                 bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12', 'B8A']):
        self.s2_toa_dir = s2_toa_dir
        self.s2_tile    = s2_tile
        self.year       = year
        self.month      = month
        self.day        = day
        self.bands      = bands # selected bands
        self.s2_bands   = 'B01', 'B02', 'B03','B04','B05' ,'B06', 'B07', \
                          'B08','B8A', 'B09', 'B10', 'B11', 'B12' #all bands
        self.s2_file_dir = os.path.join(self.s2_toa_dir, self.s2_tile[:-3],\
                                        self.s2_tile[-3], self.s2_tile[-2:],\
                                        str(self.year), str(self.month), str(self.day), acquisition)
        self.selected_img = None
        
    def get_s2_toa(self,vrt = False):
        if vrt:
        # open the created vrt file with 10 meter, 20 meter and 60 meter 
        # grouped togehter and use gdal memory map to open it
            g = gdal.Open(self.s2_file_dir+'/10meter.vrt')
            data= g.GetVirtualMemArray()
            b2,b3,b4,b8 = data
            g1 = gdal.Open(self.s2_file_dir+'/20meter.vrt')
            data1 = g1.GetVirtualMemArray()
            b5, b6, b7, b8a, b11, b12 = data1
            g2 = gdal.Open(self.s2_file_dir+'/60meter.vrt')
            data2 = g2.GetVirtualMemArray()
            b1, b9, b10 = data2
            img = dict(zip(self.s2_bands, [b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b8a]))
            if self.bands is not None:
                imgs = {k: img[k] for k in self.bands}
            else:
                imgs = img
        else:
             if self.bands is None:
                 self.bands = self.s2_bands
             fname = [self.s2_file_dir+'/%s.jp2'%i for i in self.bands]
             pool = Pool(processes=len(fname))
             ret = pool.map(read_s2_band, fname)
             imgs = dict(zip(self.bands, ret))
        self.selected_img = copy.deepcopy(imgs)
        return self.selected_img

    def get_s2_cloud(self,):

        needed_bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        cloud_bands  = ['B01', 'B02', 'B04', 'B05', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
        ratio        = [ 6,     1,     1,     2,     1,     2,     6,     6,     2,     2]
        fname        = [self.s2_file_dir+'/%s.jp2'%i for i in needed_bands]
        pool         = Pool(processes=len(fname))
        ret          = pool.map(read_s2_band, fname)
        imgs         = dict(zip(self.bands, ret))
        refs         = np.array([np.repeat(np.repeat(imgs[band]/10000., ratio[i], axis=0), ratio[i], axis=1) for i, band in enumerate(cloud_bands)]).reshape(10,-1)
        #refs         = refs.reshape(10, 10980//2, 2, 10980//2, 2,).mean(axis=(4,2).reshape(10,-1)
        classifier   = pkl.load(open('./data/sen2cloud_detector.pkl', 'rb'))
        cloud_probs  = classifier.predict(refs.T)[:,1]
        cloud_mask   = cloud_probs > 0.4
        fftconvolve(cloud_prob, disk(4)/np.sum(disk(4)), mode='same')

        return cloud_mask
        
        if glob(self.s2_file_dir+'/cloud.tif')==[]:
            print('Rasterizing cloud mask')
            g     = gdal.Open(self.s2_file_dir+'/B04.jp2')
            geo_t = g.GetGeoTransform()
            x_size, y_size = g.RasterXSize, g.RasterYSize
            xmin, xmax  = min(geo_t[0], geo_t[0] + x_size * geo_t[1]), \
                          max(geo_t[0], geo_t[0] + x_size * geo_t[1])
            ymin, ymax  = min(geo_t[3], geo_t[3] + y_size * geo_t[5]), \
                          max(geo_t[3], geo_t[3] + y_size * geo_t[5])
            xRes, yRes  = abs(geo_t[1]), abs(geo_t[5])
            try:
                if len(open(self.s2_file_dir+ "/qi/MSK_CLOUDS_B00.gml", 'rb').readlines())>5:
                    self.cirrus = gdal.Rasterize("", self.s2_file_dir+ "/qi/MSK_CLOUDS_B00.gml", \
                                                 format="MEM", xRes=xRes, yRes=yRes, where="maskType='CIRRUS'", \
                                                 outputBounds=[xmin, ymin, xmax, ymax], noData=np.nan, burnValues=1).ReadAsArray()
                else:
                    self.cirrus = np.zeros((x_size, y_size)).astype(bool)
            except:
                self.cirrus = np.zeros((x_size, y_size)).astype(bool)
            try:
                if len(open(self.s2_file_dir+ "/qi/MSK_CLOUDS_B00.gml", 'rb').readlines())>5:
                    self.cloud  = gdal.Rasterize("", self.s2_file_dir+ "/qi/MSK_CLOUDS_B00.gml", \
                                             format="MEM", xRes=xRes, yRes=yRes, where="maskType='OPAQUE'", \
                                             outputBounds=[xmin, ymin, xmax, ymax], noData=np.nan, burnValues=2).ReadAsArray()
                else:
                    self.cloud = np.zeros((x_size, y_size)).astype(bool)
            except:
                self.cloud  = np.zeros((x_size, y_size)).astype(bool)
            cloud_mask  = self.cirrus + self.cloud
            driver = gdal.GetDriverByName('GTiff')
            g1 = driver.Create(self.s2_file_dir+'/cloud.tif', \
                               g.RasterXSize, g.RasterYSize, 1, gdal.GDT_Byte,  options=["TILED=YES", "COMPRESS=JPEG"])
            projection   = g.GetProjection()
            geotransform = g.GetGeoTransform()
            g1.SetGeoTransform( geotransform )
            g1.SetProjection( projection )
            gcp_count = g.GetGCPs()
            if gcp_count != 0:
                g1.SetGCPs( gcp_count, g.GetGCPProjection() )
            g1.GetRasterBand(1).WriteArray(cloud_mask)
            g1=None; g=None
        else:
            cloud_mask = gdal.Open(self.s2_file_dir+\
                                   '/cloud.tif').ReadAsArray()
        self.cirrus = (cloud_mask == 1)
        self.cloud  = (cloud_mask >= 2)
        #self.cloud[:] = False
        self.cloud_cover = 1.*(self.cloud==2)/self.cloud.size

    def get_s2_angles(self, reconstruct = True, slic = None):


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

        if self.bands is None:
            bands = self.s2_bands
        else:
            bands = self.bands
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
                m_vaa    = np.nanmean([mva_[_] for _ in bands])
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
            self.vza[band]   = np.repeat(np.repeat(g_vza, 500, axis = 0), 500, axis = 1)[:10980, :10980]
            g_vaa[g_vaa>180] = g_vaa[g_vaa>180] - 360
            self.vaa[band]   = np.repeat(np.repeat(g_vaa, 500, axis = 0), 500, axis = 1)[:10980, :10980]


            # seems like scene containing positive and negative vaa
            # is more likely to have the wrong vaa angle and the mean value is used
            if not ((self.vaa[band] <= 0).all() or (self.vaa[band] >= 0).all()):
                reconstruct = False # no need for reconstruct anymore
                self.vaa[band][:]                    = self.mva[band] 
                self.vaa[band][self.vaa[band] > 180] = self.vaa[band][self.vaa[band] > 180] - 360
        self.angles = {'sza':self.sza, 'saa':self.saa, 'msz':self.msz, 'msa':self.msa,\
                           'vza':self.vza, 'vaa': self.vaa, 'mvz':self.mvz, 'mva':self.mva}

        if reconstruct:
            try:
                if len(glob(self.s2_file_dir + '/angles/VAA_VZA_*.tif')) == 13:
                    pass
                else:
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
                        if slic is None:
                            self.vaa[bands[i]] = angs[0]
                            self.vza[bands[i]] = angs[1]
                        else:
                            x_ind, y_ind = np.array(slic)
                            self.vaa[bands[i]] = angs[0][x_ind, y_ind]
                            self.vza[bands[i]] = angs[1][x_ind, y_ind]
                    self.angles = {'sza':self.sza, 'saa':self.saa, 'msz':self.msz, 'msa':self.msa,\
                                   'vza':self.vza, 'vaa': self.vaa, 'mvz':self.mvz, 'mva':self.mva}
                else:
                    print ('Reconstruct failed and original angles are used.')
            except:
                print('Reconstruct failed and original angles are used.')
if __name__ == '__main__':
    
    s2 = read_s2('/store/S2_data/', '50SLH', \
                  2016, 3, 14, bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12'] )
    '''
    s2.selected_img = s2.get_s2_toa() 
    s2.get_s2_cloud()
    '''
    s2.get_s2_angles()
