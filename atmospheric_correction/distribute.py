from dask.distributed import Client
from distributed.deploy.ssh import SSHCluster
def dmap(func, jobs):
    client = Client('tcp://nemesis:8786')
    L=client.map(func, jobs)
    return client.gather(L)


    with open('/home/ucfafyi/hosts.txt', 'rb') as f:                                                                                                         
        hosts = f.read().split()
    c = SSHCluster(scheduler_addr=hosts[0], scheduler_port = 8786, worker_addrs=hosts[1:], nthreads=0, nprocs=1, \
                   ssh_username =None, ssh_port=22, ssh_private_key=None, nohost=False, logdir='/tmp/')
    client = Client('tcp://nemesis:8786')
    #pmap = partial(dmap, client = client)
    

