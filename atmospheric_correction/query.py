import requests
from datetime import datetime, timedelta
with open('auth', 'rb') as f:
    auth = tuple(f.read().split('\n')[0].split(','))
from osgeo import ogr
from osgeo import osr
import numpy as np

source = osr.SpatialReference()
source.ImportFromEPSG(4326)
target = osr.SpatialReference()
target.ImportFromProj4('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')
transform = osr.CoordinateTransformation(source, target)
tile_area = 12056040000.

base = 'https://scihub.copernicus.eu/dhus/search?start=0&rows=100&q='
t_date = datetime(2016, 11, 1)

def find_all(location, start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'), cloud_cover='[0 TO 100]', product_type='S2MSI1C', search_by_tile=True, val_pix_thresh = 0):
    if isinstance(cloud_cover, (int, float)):
        cloud_cover = '[0 TO %s]'%cloud_cover
    elif isinstance(cloud_cover, str):
        cloud_cover = cloud_cover
    else:
        raise IOError('Cloud cover can only be number or string, e.g. "[0 TO 10.1]"')
    ret = []
    temp1 = None
    if datetime.strptime(end, '%Y-%m-%d') < t_date:
        search_by_tile = False
    elif datetime.strptime(start, '%Y-%m-%d') < t_date:
        print 'Searching data after %s'%(t_date.strftime('%Y-%m-%d'))
        temp1 = '( footprint:"Intersects(%s)" ) AND ( beginPosition:[%sT00:00:00.000Z TO %sT23:59:59.999Z] '+\
               'AND endPosition:[%sT00:00:00.000Z TO %sT23:59:59.999Z] ) AND (platformname:Sentinel-2 AND producttype:%s '+\
               'AND cloudcoverpercentage:%s)&orderby=beginposition desc& &format=json'
    else:
        pass   
    if isinstance(search_by_tile, str):
        temp = '( footprint:"Intersects(%s)" ) AND ( beginPosition:[%sT00:00:00.000Z TO %sT23:59:59.999Z] '+\
               'AND endPosition:[%sT00:00:00.000Z TO %sT23:59:59.999Z] ) AND (platformname:Sentinel-2 AND producttype:%s '+\
               'AND cloudcoverpercentage:%s AND filename:S2?_MSIL1C_????????T??????_N*_R*_T'+search_by_tile +'_*)&orderby=beginposition desc& &format=json'
    elif search_by_tile:
        temp = '( footprint:"Intersects(%s)" ) AND ( beginPosition:[%sT00:00:00.000Z TO %sT23:59:59.999Z] '+\
               'AND endPosition:[%sT00:00:00.000Z TO %sT23:59:59.999Z] ) AND (platformname:Sentinel-2 AND producttype:%s '+\
               'AND cloudcoverpercentage:%s AND filename:S2?_MSIL1C_????????T??????_N*_R*_T*_*)&orderby=beginposition desc& &format=json'
    else:
        temp = '( footprint:"Intersects(%s)" ) AND ( beginPosition:[%sT00:00:00.000Z TO %sT23:59:59.999Z] '+\
               'AND endPosition:[%sT00:00:00.000Z TO %sT23:59:59.999Z] ) AND (platformname:Sentinel-2 AND producttype:%s '+\
               'AND cloudcoverpercentage:%s)&orderby=beginposition desc& &format=json'
    url = base + temp%(location, start, end, start, end, product_type, cloud_cover)
    #print url
    r = requests.get(url, auth = auth)
    feed = r.json()['feed']
    total = int(feed['opensearch:totalResults'])
    pages = int(np.ceil(total/100.))
    print 'Total', total, 'files in', pages, 'pages'
    print 'searching page', 1, 'of', pages
    for i in range(len(feed['entry'])):
        title = feed['entry'][i]['title']
        cloud = feed['entry'][i]['double']['content']
        durl  = feed['entry'][i]['link'][0]['href']#.replace('$value', '\$value')  
        qurl  = feed['entry'][i]['link'][2]['href']#.replace('$value', '\$value')  
        date  = feed['entry'][i]['date'][1]['content']
        for j in feed['entry'][i]['str']:
            if 'POLYGON ((' in j['content']:
                    foot = j['content']
                    if search_by_tile:
                        geom = ogr.CreateGeometryFromWkt(foot)
                        geom.Transform(transform)
                        val_pix = (geom.GetArea()/tile_area)*100.
                        if val_pix > val_pix_thresh:
                            ret.append([title, date, foot, cloud, durl, qurl, val_pix])
                    else:
                        val_pix = None
                        ret.append([title, date, foot, cloud, durl, qurl, val_pix])
    if total >= 100:
        for page in range(1, pages):
            print 'searching page', page+1, 'of', pages
            url = base.replace('start=0&rows=100', 'start=%d&rows=100'%(page*100)) + temp%(location, start, end, start, end, product_type, cloud_cover)
            r = requests.get(url, auth = auth)
            feed = r.json()['feed']                  
            date  = feed['entry'][i]['date'][1]['content']
            for i in range(len(feed['entry'])):
                title = feed['entry'][i]['title']
                cloud = feed['entry'][i]['double']['content']
                durl  = feed['entry'][i]['link'][0]['href']#.replace('$value', 'i$value')
                qurl  = feed['entry'][i]['link'][2]['href']#.replace('$value', '\$value')   
                for ds in feed['entry'][i]['date']:
                    if ds['name'] == u'beginposition':
                        date  = ds['content']
                for j in feed['entry'][i]['str']:
                    if 'POLYGON ((' in j['content']:
                            foot = j['content']
                            if search_by_tile:
                                geom = ogr.CreateGeometryFromWkt(foot)
                                geom.Transform(transform)
                                val_pix = (geom.GetArea()/tile_area)*100.
                                if val_pix > val_pix_thresh:                    
                                    ret.append([title, foot, cloud, durl, qurl, val_pix])
                            else:
                                val_pix = None
                                ret.append([title, date, foot, cloud, durl, qurl, val_pix])
    if temp1 is not None:
        print 'Searching data before %s'%(t_date.strftime('%Y-%m-%d'))
        url = base + temp1%(location, start, t_date.strftime('%Y-%m-%d'), start, t_date.strftime('%Y-%m-%d'), product_type, cloud_cover) 
        #print url            
        r = requests.get(url, auth = auth)
        feed = r.json()['feed']           
        total = int(feed['opensearch:totalResults'])
        pages = int(np.ceil(total/100.))  
        print 'Total', total, 'files in', pages, 'pages'                                
        print 'searching page', 1, 'of', pages                                          
        for i in range(len(feed['entry'])):
            title = feed['entry'][i]['title']                                           
            cloud = feed['entry'][i]['double']['content']                               
            durl  = feed['entry'][i]['link'][0]['href']#.replace('$value', '\$value')   
            qurl  = feed['entry'][i]['link'][2]['href']#.replace('$value', '\$value')   
            for ds in feed['entry'][i]['date']:
                    if ds['name'] == u'beginposition':
                        date  = ds['content'] 
            for j in feed['entry'][i]['str']:                                           
                if 'POLYGON ((' in j['content']:                                        
                        foot = j['content']                                             
            ret.append([title, date, foot, cloud, durl, qurl, None])                                
        if total >= 100:     
            for page in range(1, pages):  
                print 'searching page', page+1, 'of', pages                             
                url = base.replace('start=0&rows=100', 'start=%d&rows=100'%(page*100)) + temp1%(location,start, \
                      t_date.strftime('%Y-%m-%d'), start, t_date.strftime('%Y-%m-%d'), product_type, cloud_cover)
                r = requests.get(url, auth = auth)
                feed = r.json()['feed']                                                 
                for i in range(len(feed['entry'])):                                     
                    title = feed['entry'][i]['title']                                   
                    cloud = feed['entry'][i]['double']['content']                       
                    durl  = feed['entry'][i]['link'][0]['href']#.replace('$value', 'i$value')
                    qurl  = feed['entry'][i]['link'][2]['href']#.replace('$value', '\$value')   
                    for ds in feed['entry'][i]['date']:
                        if ds['name'] == u'beginposition':
                            date  = ds['content'] 
                    for j in feed['entry'][i]['str']:
                        if 'POLYGON ((' in j['content']:
                            foot = j['content']
                    ret.append([title, date, foot, cloud, durl, qurl, None])
        
        
    return ret

def query_sen2(location, start='2015-01-01', end=datetime.now().strftime('%Y-%m-%d'), cloud_cover='[0 TO 100]', product_type='S2MSI1C', search_by_tile=True, band = None, val_pix_thresh = 0 ):
    ret = find_all(location, start, end, cloud_cover, product_type, search_by_tile, val_pix_thresh)
    for re in ret:
        if band is not None:
            r = requests.get("%s/Nodes('%s.SAFE')/Nodes('manifest.safe')/$value"%(re[4].split('/$value')[0], re[0]), auth = auth)
            man = r.content                                                                                                      
            for i in man.split('</'):                                                                                            
                if 'locatorType="URL" href=' in i:                                                                               
                    fname = i.split('href=')[1].split('"')[1]                                                                    
                    if isinstance(search_by_tile, str):                                                                                    
                        if (search_by_tile in fname) & (band in fname):                                                                                   
                            print fname 
                    else:
                        if band in fname:                                                                                   
                            print fname 
        else:
            sen_time = datetime.strptime(re[1].split('T')[0], '%Y-%m-%d')
            if (sen_time  < t_date) & isinstance(search_by_tile, str): 
                #print 'filtering data before %s'%(t_date.strftime('%Y-%m-%d'))
                r = requests.get("%s/Nodes('%s.SAFE')/Nodes('manifest.safe')/$value"%(re[4].split('/$value')[0], re[0]), auth = auth)
                man = r.content
                for i in man.split('</'):
                    if 'locatorType="URL" href=' in i:
                        fname = i.split('href=')[1].split('"')[1]
                        if search_by_tile in fname:
                            print fname
        
                #r = requests.get("%s/Nodes('%s.SAFE')/Nodes('GRANULE')/Nodes?$format=json"%(re[4].split('/$value')[0], re[0]), auth = auth) 
                #rj = r.json()
                #for i in rj['d']['results']:
                #    if tile_name is not None:
                #        if tile_name in i['Id']:
                #            if band is not None:
                #                #print i['Id'], i['Nodes']['__deferred']['uri']
                #                print i['Nodes']['__deferred']['uri'] + "('IMG_DATA')/Nodes('" + i['Id'].split('_N')[0] + "_%s.jp2')/$value" %band
def downdown(url_fname, auth=auth):
    url, fname = url_fname
    r = requests.get(url, stream=True, auth=auth)
    with open(fname, 'wb') as f:
        for chunk in r.iter_content(chunk_size=10240): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


if __name__ == '__main__':
     a = query_sen2('POLYGON((4.795757185813976 41.21247680330302,19.787525048248387 41.21247680330302,19.787525048248387 51.690725710472634,4.795757185813976 51.690725710472634,4.795757185813976 41.21247680330302))', search_by_tile='33TVJ', end='2017-12-10', start='2016-01-01', val_pix_thresh=60, cloud_cover=20.1, band = 'B02')
