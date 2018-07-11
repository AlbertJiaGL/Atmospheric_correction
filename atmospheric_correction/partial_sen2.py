import os
import time
import errno
import shutil
import struct
import requests
import numpy as np
from multiprocessing import Pool
from collections import namedtuple
from zipfile import ZipExtFile
try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO
with open('auth', 'rb') as f:
    auth = tuple(f.read().split('\n')[0].split(','))
url = "https://scihub.copernicus.eu/dhus/odata/v1/Products('db63f27e-7dd3-4479-adaf-6a9abd49cddd')/$value"
def get_CDr(url):
    '''
    Get the footer of zip file and taking the maximum 16k comments length
    into account to reduce the requsts times.
    '''
    headers = {"Range": "bytes=-65558"}
    r = requests.get(url, headers=headers, auth=auth)
    total_size = int(r.headers['Content-Range'].split('/')[-1])
    foot = r.content
    end32 = foot.rfind('PK\x05\x06')
    if end32 != -1:
        sig, disk_num, disk_dir, dircount, dircount2, dirsize, diroffset, comment_len = struct.unpack("<4s4H2LH", foot[end32: end32 + 22])
    else:
        raise IOError('Bad zip file.')

    end64 = foot.rfind('PK\x06\x07')
    if end64 !=-1:
        size_end_64 = 20
        sig, diskno, reloff, disks = struct.unpack("<4sLQL", foot[end64: end64 + size_end_64])

    foot64 = foot.rfind('PK\x06\x06')
    if foot64 != -1:
        size_foot_64 = 56
        footer64 = foot[foot64: foot64 + size_foot_64]
        sig, sz, create_version, read_version, disk_num, disk_dir, \
        dircount, dircount2, dirsize, diroffset = struct.unpack("<4sQ2H2L4Q", footer64)
    headers = {"Range": "bytes=%d-%d"%(diroffset, total_size)}    
    r = requests.get(url, headers=headers, auth=auth)
    CD = r.content
    return CD, diroffset, dircount2, total_size 

def decode_CD(url):
    structCentralDir = "<4s4B4HL2L5H2L"
    CD, back, number_entry, total_size = get_CDr(url)
    #CD = CDr.content
    #total_size = int(CDr.headers['Content-Range'].split('/')[-1])
    #total_size = int(CDr.headers['Content-Length'])
    names = ['cdsig', 'create_version', 'create_system', 'extract_version', 'reserved', 'flag_bits', \
             'compress_type', 't', 'd', 'CRC', 'compress_size', 'file_size', 'n', 'm', 'k', 'volume', \
             'internal_attr', 'external_attr', 'header_offset', 'filename', 'date_time', 'total_size', 'back']
    off = 0
    flist = []
    for i in range(number_entry):
        cd = struct.unpack(structCentralDir, CD[off: off + 46])
        fname = CD[off+46: off+46+cd[12]]  
        date_time = ( (cd[8]>>9)+1980, (cd[8]>>5)&0xF, cd[8]&0x1F, cd[7]>>11, (cd[7]>>5)&0x3F, (cd[7]&0x1F) * 2 ) 
        cd = list(cd) + [fname, date_time, total_size, back]
        #cd[9] = None
        CDT = namedtuple("CDT", names, verbose=False, rename=False)
        cdt = CDT._make(cd) 
        if cdt.m >= 4:
            extra = CD[off + cdt.n +46: off +cdt.n +46+ cdt.m]
            while len(extra) >=4:
                tp, ln = struct.unpack('<HH', extra[:4])
                if tp == 1:
                    if ln >= 24:
                        counts = struct.unpack('<QQQ', extra[4:28])
                    elif ln == 16:
                        counts = struct.unpack('<QQ', extra[4:20])
                    elif ln == 8:
                        counts = struct.unpack('<Q', extra[4:12])
                    elif ln == 0:
                        counts = ()
                    else:
                        raise RuntimeError, "Corrupt extra field %s"%(ln,)
                extra = extra[ln+4:]
            idx = 0
            if cdt.file_size in (0xffffffffffffffffL, 0xffffffffL):
                #cdt.file_size = counts[idx]
                cdt = cdt._replace(file_size=counts[idx]) 
                idx += 1  

            if cdt.compress_size == 0xFFFFFFFFL:
                #cdt.compress_size = counts[idx]
                cdt = cdt._replace(compress_size=counts[idx]) 
                idx += 1

            if cdt.header_offset == 0xffffffffL:
                #old = cdt.header_offset
                cdt = cdt._replace(header_offset=counts[idx])
                #cdt.header_offset = counts[idx]
                idx+=1

        off  = off + cdt.n + cdt.m + cdt.k + 46
        flist.append(cdt)
    flist = sorted(flist, key=lambda cdt: cdt.header_offset)
    offsets = np.array([i.header_offset for i in flist])
    new_list = []
    names = []
    for i in range(len(offsets)):
        cdt = flist[i]
        ncdt = namedtuple('filed_%d'%i, cdt._fields + ('end',))
        names.append('filed_%d'%i)
        vals = [va for va in cdt]
        if i < len(offsets)-1: 
            vals.append(offsets[i+1])
        else:
            vals.append(back-1)
        nncdt = ncdt._make(vals)
        new_list.append(nncdt)
    fl = namedtuple('Full', names)
    nfl = fl._make(new_list)
    return nfl

def get_r(p):                                                         
    url, r, auth, proxy = p
    headers = {"Range": "bytes=%d-%d"%(r[0], r[1])}
    #print r[0], r[1]-1
    length = r[1] - r[0] + 1
    proxies={"http": proxy, "https": proxy}
    for i in range(100):
        r = requests.get(url, headers=headers, auth=tuple(auth))#, proxies=proxies)
        if len(r.content) == length:
            break
        else:
            print headers
            print auth
            print len(r.content), length
            time.sleep(1)
    return r.content

def get_menber(url, rang, fname, zipinfo_dict, auth, proxy):
    zipinfo = namedtuple('GenericDict', zipinfo_dict.keys())(**zipinfo_dict)
    headers = {"Range": "bytes=%d-%d"%(rang[0], rang[1]-1)}
    proxies={"http": proxy, "https": proxy}
    r = requests.get(url, headers=headers, auth=auth)#, proxies=proxies)
    bad_file = []
    if r.ok:
        structFileHeader = "<4s2B4HL2L2H"
        stringFileHeader = "PK\003\004"
        sizeFileHeader = 30
        fname_length, extra_length = struct.unpack(structFileHeader, r.content[:sizeFileHeader])[10:12]
        off = sizeFileHeader + fname_length + extra_length
        target = open(fname, 'wb')
        f = StringIO(r.content[off:])
        if len(r.content[off:]) != zipinfo.compress_size:
            bad_file.append(fname)
        source = ZipExtFile(f, 'r', zipinfo, None, False)
        shutil.copyfileobj(source, target)
        target.close()
        size = os.stat(fname).st_size
        if size != zipinfo.file_size:
            bad_file.append(fname)
            os.remove(fname)
    else:
        bad_file.append(fname)
    if len(bad_file)>0:
        print('Bad files: ')
        print bad_file
        return bad_file
    
def get_content(url, rang, auths, proxy):
    ranges = np.linspace(rang[0], rang[1], len(auths)*2, dtype=int)
    nr = [[ranges[i], ranges[i+1]-1] for i in range(len(ranges)) if ranges[i] != ranges[-1]]
    re_proxy = proxy[:len(nr)] * 2
    re_auths =  auths * 2
    ps = [[url, nr[i], re_auths[i], re_proxy[i]]for i in range(len(nr))]
    p = Pool(len(nr)) 
    ret = p.map(get_r, ps)   
    ret = ''.join(ret)[:-1]
    pool.close()
    pool.join()
    return ret

def decode_and_save(ret,fname, zipinfo_dict):
    zipinfo = namedtuple('GenericDict', zipinfo_dict.keys())(**zipinfo_dict)
    structFileHeader = "<4s2B4HL2L2H"
    stringFileHeader = "PK\003\004"                   
    sizeFileHeader = 30 
    fname_length, extra_length = struct.unpack(structFileHeader, ret[:sizeFileHeader])[10:12]
    off = sizeFileHeader + fname_length + extra_length

    if not os.path.exists(os.path.dirname(fname)):
        try:
            os.makedirs(os.path.dirname(fname))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    target = open(fname, 'wb')

    f = StringIO(ret[off:])
    if len(ret[off:]) != zipinfo.compress_size:
        print len(ret[off:]),zipinfo.compress_size
        print fname, 'length is not right'                   
    source = ZipExtFile(f, 'r', zipinfo, None, False)
    shutil.copyfileobj(source, target)
    target.close()      
    size = os.stat(fname).st_size
    if size != zipinfo.file_size:
        print 'wrong file size', fname
        #bad_file.append(fname)                        
        #os.remove(fname)
        #return fname

def get_content_once(url, rang, auths, proxy):
    ranges = np.linspace(rang[0], rang[1], len(auths)*2, dtype=int)
    nr = [[ranges[i], ranges[i+1]-1] for i in range(len(ranges)) if ranges[i] != ranges[-1]]
    re_proxy = proxy[:len(nr)] * 2
    re_auths =  auths * 2
    ps = [[url, nr[i], re_auths[i], re_proxy[i]]for i in range(len(nr))]
    p = Pool(len(nr)) 
    ret = p.map(get_r, ps)   
    ret = ''.join(ret)[:-1]  
    pool.close()
    pool.join()                                                                                                         
    return ret

def _test_menber():
    flist = decode_CD(url)
    selected = [i for i in range(len(flist)) if ('50SKJ' in flist[i].filename) and ('.jp2' in flist[i].filename)]
    def helper(inp):                                                                                             
        url, rang, fname, zipinfo, auth, proxy = inp                                                             
        return get_menber(url, rang, fname, zipinfo, tuple(auth), proxy)
    auths = np.loadtxt('auths', dtype=str).tolist()                     
    #good = get_good()                                                  
    good = range(100)                                                   
    auths_repeat = auths * 10000                                        
    good_repeat = good  * 10000                                         
    inps = []                                                           
    for j, i in enumerate(selected):                                    
         inps.append([url, [flist[i].header_offset, flist[i].end], \
                      flist[i].filename.split('/')[-1], vars(flist[i]), auths_repeat[j], good_repeat[j]])        
    p = Pool(min([len(auths), len(selected)]))                         
    for inp in inps:                                                   
        get_menber(*tuple(inp))                                        
    ret = p.map(helper, inps) 
    pool.close()
    pool.join()                                                                                                         

def _test_auths():
    flist = decode_CD(url)
    selected = [i for i in range(len(flist)) if ('50SKJ' in flist[i].filename) and ('.jp2' in flist[i].filename)]
    def helper(inp):                                                                                             
        url, rang, fname, zipinfo, auth, proxy = inp                                                             
        return get_menber(url, rang, fname, zipinfo, tuple(auth), proxy)
    auths = np.loadtxt('auths', dtype=str).tolist()                     
    #good = get_good()                                                  
    good = range(100)                                                   
    auths_repeat = auths * 10000                                        
    good_repeat = good  * 10000                                         
    inps = []                                                           
    for j, i in enumerate(selected):                                    
         inps.append([url, [flist[i].header_offset, flist[i].end], \
                      flist[i].filename.split('/')[-1], vars(flist[i]), auths_repeat[j], good_repeat[j]])        
    #p = Pool(min([len(auths), len(selected)]))                         
    #for inp in inps:                                                   
    #    get_menber(*tuple(inp))                                        
    #ret = p.map(helper, inps) 
    rets = []                                                           
    fnames = []                                                         
    zipinfo_dicts = []                                                  
    from datetime import datetime                                       
    startTime = datetime.now()                                                                  
    for j, i in enumerate(selected):                                    
        try:                                                            
            ret = get_content(url, [flist[i].header_offset, flist[i].end], auths, good)
            rets.append(ret)                                            
            fnames.append(flist[i].filename)             
            zipinfo_dicts.append(vars(flist[i]))                        
        except:                                                         
            print flist[i].filename.split('/')[-1], 'failed'            
    inps = [[rets[i], fnames[i], zipinfo_dicts[i]] for i in range(len(rets))]
    def helper(inp):                                                    
        ret,fname, zipinfo_dict = inp                                   
        decode_and_save(ret,fname, zipinfo_dict)                        
                                                                        
        #ret = multi_get_member(url, [flist[i].header_offset, flist[i].end], \
        #                 flist[i].filename.split('/')[-1], vars(flist[i]), auths, good)
        #print ret                                                      
    p = Pool(len(rets))                                                 
    p.map(helper, inps)
    pool.close()
    pool.join()                                                                                                         
    print datetime.now() - startTime

def save_helper(inp):
    ret,fname, zipinfo_dict = inp
    decode_and_save(ret,fname, zipinfo_dict)

def downloader(url):
    flist = decode_CD(url)
    selected = [i for i in range(len(flist)) if (flist[i].filename[-1] !='/')]
    #from datetime import datetime                                       
    #startTime = datetime.now() 
    auths = np.loadtxt('auths', dtype=str).tolist()
    good = range(100)
    rang = [flist[selected[0]].header_offset, flist[selected[-1]].end]
    ret = get_content_once(url, rang, auths, good) 
    rets = []                                      
    fnames = []                                   
    zipinfo_dicts = []
    rets = []        
    off = flist[selected[0]].header_offset
    for i in selected:
        buf = ret[flist[i].header_offset -off: flist[i].end - off]
        rets.append(buf)
        fnames.append(flist[i].filename)
        zipinfo_dicts.append(vars(flist[i]))                                                                                          
     
    inps = [[rets[i], fnames[i], zipinfo_dicts[i]] for i in range(len(rets))]
    p = Pool(len(rets))                            
    p.map(save_helper, inps)                            
    pool.close()
    pool.join()                                                                                                         
    #print datetime.now() - startTime

if __name__ == '__main__':
    downloader(url)
