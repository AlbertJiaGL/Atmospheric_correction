import requests
from multiprocessing import Pool
def test(proxy):
    url = 'https://httpbin.org/ip'
    try:
        response = requests.get(url,proxies={"http": proxy, "https": proxy}, timeout=0.2)
        return proxy
    except:
        pass


def get_proxies():
    url = 'https://free-proxy-list.net/'
    response = requests.get(url)                                                         
    proxies = []                                         
    sr = response.content.split('>')                      
    for i in range(len(sr)):                                                                                                     
        try:                                                                                                                     
            ip = sr[i].split('</td')[0]                                                                              
            port = sr[i+2].split('</td')[0]  
            port = int(port)
            proxy = ip + ':' + str(port)
            proxies.append(proxy)
        except:
            pass
    return proxies
def get_good():
    proxies = get_proxies()
    p = Pool(len(proxies))
    ret = p.map(test, proxies)
    good = [i for i in ret if i is not None]
    return good
