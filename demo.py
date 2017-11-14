# coding=utf-8
import numpy as np
import reikna.cluda as cluda
from reikna.fft import FFT, FFTShift
import pyopencl.array as clarray
from pyopencl import clmath
from reikna.core import Computation, Transformation, Parameter, Annotation, Type
from reikna.algorithms import PureParallel
from matplotlib import cm
import time as t
import matplotlib.pyplot as plt

import statistic_functions4 as sf
#import mylog as Log
np.set_printoptions(threshold=np.inf)

batch = 100
N = 1024
api = cluda.any_api()
thr = api.Thread.create()  

data = np.load('8psk_data.npy')
data = np.reshape(data, (batch*4, N)) # 一共 batch*4 = 400次

t1 = t.clock()
data0 = data[0:batch, :].astype(np.complex128)
data_g = thr.to_device(data0)
print(t.clock()-t1)
#compile 
fft = FFT(data_g, (0,1))
fftc = fft.compile(thr)
data_f = thr.array(data0.shape, dtype=np.complex128)
shift = FFTShift(data_f, (0,1))
shiftc = shift.compile(thr)
data_shift = thr.array(data0.shape, dtype=np.complex128)
sum = sf.stat(thr)
logg10 = sf.logg10(thr)
def myfft(data):
    '''
    input: 
    data: cluda-Array (100, 1024)
    -----------------------------------------------
    output:
    TS_gpu: cluda-Array (1000, 1024)
    '''
    #FFT
    t_fft = t.clock()
    data_f = thr.array(data.shape, dtype=np.complex128)
    STAT_gpu = thr.array(data.shape, dtype=np.complex128)
    fftc(data_f, data)
    shiftc(STAT_gpu, data_f)
    #log
    t_log = t.clock()
    STAT_gpu = abs(STAT_gpu) 
    logg10(STAT_gpu, global_size = (N, batch))
    #统计，插值
    t_st = t.clock()
    TS_gpu = cluda.ocl.Array(thr, shape=(1000, N), dtype=np.int)
    sum(TS_gpu, STAT_gpu, global_size = (N,batch))

    print('fft: %f, log: %f, stat: %f'%(t_log-t_fft, t_st-t_log, t.clock()-t_st))
    print('total: %f'%(t.clock()-t_fft))
    return TS_gpu

i=0
j=0
fig=plt.figure()
#fig, ax = plt.subplots()
summ = 0
while i<100:
    t1 = t.clock()
    data0 = data[j:(j+1)*batch, :].astype(np.complex128)
    data_g = thr.to_device(data0)
    out = myfft(data_g)
    out = out.get()
    t2 = t.clock()
    #nipy_spectral
    plt.clf()
    #plt.imshow(out, cmap = cm.hot)
    plt.imshow(out, cmap = 'nipy_spectral')
    plt.ylim(0,1000)    
    
    plt.pause(0.00000001)
    print('No. %d, transmission+compute: %f, plot: %f'%(i, t2-t1, t.clock()-t2))
    summ = summ + t2-t1
    j = j + 1
    i = i + 1
    if j == 4:
        j=0

print('avg compute: %f'%(summ/100))
        
    
    

