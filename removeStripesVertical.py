import cv2
import numpy as np
from pywt import dwt2, wavedec2, idwt2
from scipy.fftpack import fft,ifft,fftshift,ifftshift
from utils import *
from detection import *

def removeStripes(img, decNum, wname, sigma):
    #print(img.shape)
    cH, cV, cD = [],[],[]
    # wavelet decomposition
    for i in range(decNum):
        img, (ch, cv, cd) = dwt2(img, wname)
        #print(cd.shape)
        cH.append(ch)
        cV.append(cv)
        cD.append(cd)
    #print(len(cD[1]))
    # FFT transform of horizontal frequency bands
    for i in range(decNum):
        fcV = fftshift(fft(cV[i]))
        my, mx = fcV.shape
        # damping of vertical stripe information
        damp=1-np.exp(np.negative(np.linspace(-(my//2),(my//2),my)**2/(2*sigma**2)))
        #damp = np.reshape(damp,(len(damp),1))
        mask = np.zeros(fcV.shape)
        #print(np.linspace(-(my//2),(my//2),my))
        for ii in range(mx):
            mask[:,ii] = damp
        #print(np.reshape(np.repeat(damp,mx),(my,mx)))
        fcV = fcV*mask
        #inverse FFT
        cV[i] = ifft(ifftshift(fcV))

    #wavelet reconstruction
    nimg = img
    for i in range(decNum-1,-1, -1):
        y, x = cH[i].shape
        nimg=nimg[:y,:x]
        nimg=idwt2((nimg,(cH[i],cV[i],cD[i])),wname)

    return abs(nimg)

f = readImg(r'path to your image')
img_rgb = readImg(r'path to your image', False)
showImg(img_rgb)
# print(f)
wavename = 'coif1'

detectionCall(img_rgb,f)
# print(img)
# showImg(img)

