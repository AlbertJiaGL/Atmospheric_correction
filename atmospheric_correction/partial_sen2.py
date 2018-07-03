import os
import shutil
import struct
import requests
import numpy as np
from collections import namedtuple
from zipfile import ZipExtFile
try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO
with open('auth', 'rb') as f:
    auth = tuple(f.read().split('\n')[0].split(','))
#url = "https://scihub.copernicus.eu/dhus/odata/v1/Products('58f47726-a280-4a7e-9663-22253a67da44')/$value"
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
        sig, disk_num, disk_dir, dircount, dircount2, dirsize, diroffseti, comment_len = struct.unpack("<4s4H2LH", foot[end32: end32 + 22])
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
            vals.append(total_size)
        nncdt = ncdt._make(vals)
        new_list.append(nncdt)
    fl = namedtuple('Full', names)
    nfl = fl._make(new_list)
    return nfl

def get_menber(url, rang, fname, zipinfo_dict, auth):
    zipinfo = namedtuple('GenericDict', zipinfo_dict.keys())(**zipinfo_dict)
    headers = {"Range": "bytes=%d-%d"%(rang[0], rang[1]-1)}
    requests.adapters.DEFAULT_RETRIES = 5
    r = requests.get(url, headers=headers, auth=auth)
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
    
    
if __name__ == '__main__':
    flist = decode_CD(url)
    selected = [i for i in range(len(flist)) if ('50SKJ' in flist[i].filename) and ('.jp2' in flist[i].filename)]
    from multiprocessing import Pool
    from functools import partial
    import sys
    sys.path.insert(0, 'util')
    from multi_process import parmap
    def helper(inp):
        url, rang, fname, zipinfo, auth = inp
        return get_menber(url, rang, fname, zipinfo, auth)
    auths = np.loadtxt('auths', dtype=str)
    auths_repeat = auths.tolist() * 10000
    inps = []
    for j, i in enumerate(selected): 
         inps.append([url, [flist[i].header_offset, flist[i].end], \
                      flist[i].filename.split('/')[-1], vars(flist[i]), tuple(auths_repeat[j])])
    p = Pool(min([len(auths), len(selected)]))
    ret = p.map(helper, inps)


