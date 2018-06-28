import os
import shutil
import requests
import struct
from collections import namedtuple
from zipfile import ZipExtFile
try:
    from cStringIO import StringIO
except:
    from StringIO import StringIO
auth = ('MarcYin', 'Marc1234')
url = "https://scihub.copernicus.eu/dhus/odata/v1/Products('58f47726-a280-4a7e-9663-22253a67da44')/$value"

def get_footer(url):
    headers = {"Range": "bytes=-1000"}
    foot = requests.get(url, headers=headers, auth=auth).content
    footer = None
    for i in range(1000):
        if foot[i:i+4] =='PK\x05\x06':           
            footer = foot[i:]
    if footer is None:
        raise IOError('Bad zip file.')
    else:
        return footer
#footer = get_footer(url)
#number_entry = struct.unpack('<h', footer[10:12])[0]
#size_CD = struct.unpack('<L', footer[12:16])

def get_CDr(url):
    footer = get_footer(url)
    size_CD = struct.unpack('<L', footer[12:16])[0]
    number_entry = struct.unpack('<h', footer[10:12])[0]
    back = -1 * (size_CD + len(footer))
    headers = {"Range": "bytes=%d"%(back)}
    r = requests.get(url, headers=headers, auth=auth)
    return r, back, number_entry

def decode_CD(url):
    structCentralDir = "<4s4B4HL2L5H2L"
    CDr, back, number_entry = get_CDr(url)
    CD = CDr.content
    total_size = int(CDr.headers['Content-Range'].split('/')[-1])
    #total_size = int(CDr.headers['Content-Length'])
    names = ['cdsig', 'create_version', 'create_system', 'extract_version', 'reserved', 'flag_bits', \
             'compress_type', 't', 'd', 'CRC', 'compress_size', 'file_size', 'n', 'm', 'k', 'volume', \
             'internal_attr', 'external_attr', 'header_offset', 'filename', 'date_time', 'total_size', 'back']
    off = 0
    flist = []
    for i in range(number_entry):
        cd = struct.unpack(structCentralDir, CD[off: off + 46])
        CDT = namedtuple("CDT", names, verbose=False, rename=False)
        fname = CD[off+46: off+46+cd[12]]  
        date_time = ( (cd[8]>>9)+1980, (cd[8]>>5)&0xF, cd[8]&0x1F, cd[7]>>11, (cd[7]>>5)&0x3F, (cd[7]&0x1F) * 2 ) 
        cd = list(cd) + [fname, date_time, total_size, back]
        cd[9] = None
        cdt = CDT._make(cd) 
        off  = off + cdt.n + cdt.m + cdt.k + 46
        flist.append(cdt)
    content_range = []
    for i in range(len(flist)-1):
        content_range.append([flist[i].header_offset, flist[i+1].header_offset])
    content_range.append([flist[-1].header_offset, total_size+back])
        
    return flist, content_range

def get_menber(url, rang, fname, zipinfo):
    headers = {"Range": "bytes=%d-%d"%(rang[0], rang[1]-1)}
    r = requests.get(url, headers=headers, auth=auth)
    structFileHeader = "<4s2B4HL2L2H"
    stringFileHeader = "PK\003\004"
    sizeFileHeader = struct.calcsize(structFileHeader)
    fname_length, extra_length = struct.unpack(structFileHeader, r.content[:sizeFileHeader])[10:12]
    off = sizeFileHeader + fname_length+ extra_length
    target = open(fname, 'wb')
    #with open('/tmp/a_', 'wb') as f: 
    #    f.write(r.content[off:])
    #with open('/tmp/a_', 'rb') as f:
    f = StringIO(r.content[off:])
    source = ZipExtFile(f, 'r', zipinfo, None, False)
    shutil.copyfileobj(source, target)
    target.close()
    #os.remove('/tmp/a_')
flist = decode_CD(url)
#selcted = [[flist[0][i], flist[1][i]] for i in range(len(flist[0])) if ('.jp2' in flist[0][i].filename) and ('_B' in flist[0][i].filename)]
#selected = [[flist[0][i], flist[1][i]] for i in range(len(flist[0])) if ('.gml' in flist[0][i].filename)] #and ('_B' in flist[0][i].filename)]
#for i in selected: 
#     print i[0].filename.split('/')[-1]
#     get_menber(url, i[1], i[0].filename.split('/')[-1], i[0])


