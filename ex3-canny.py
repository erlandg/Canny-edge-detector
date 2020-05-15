import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
from PIL import Image
from math import floor

# Canny edge detection

im = np.array(Image.open('Fig1016(a)(building_original).tif')).astype('float')/255
P = im.shape[0];   Q = im.shape[1]

G = lambda n,s: np.array([[np.exp(-((x-(n-1)/2)**2 + (y-(n-1)/2)**2)/(2*s)) for y in range(n)] for x in range(n)])
s = 5;   n = 9
Gmat = G(n,s)
fs = ss.convolve2d(im, Gmat, 'same')

roby = np.array([[-1,-1,-1],[0,0,0],[1,1,1]]);   robx = roby.T
gx = ss.convolve(fs, robx, mode='same')
gy = ss.convolve(fs, roby, mode='same')
alpha = np.arctan(gy/gx)
Mmat = np.sqrt(gx**2 + gy**2)

d1 = 2*np.pi * 22.5/360; d2 = d1+np.pi/4; d3 = d2+np.pi/4; d4 = d3+np.pi/4
d = [-d4, -d3, -d2, -d1, d1, d2, d3, d4]

gN = np.zeros([P, Q])
Th = 2.5
Tl = 1.3
for x in range(1,P-1):
    for y in range(1,Q-1):
        K = Mmat[x,y]
        dNind = np.argmin(np.absolute(alpha[x,y] - d))
        if dNind > 3: dNind -= 4

        if (dNind == 0): a = Mmat[x-1,y+1]; b = Mmat[x+1,y-1]
        elif (dNind == 1): a = Mmat[x-1,y]; b=Mmat[x+1,y]
        elif (dNind == 2): a = Mmat[x-1,y-1]; b=Mmat[x+1,y+1]
        elif (dNind == 3): a = Mmat[x,y-1]; b=Mmat[x,y+1]

        if (fs[x,y] >= a) and (fs[x,y] >= b): gN[x,y] = K

Thim = gN.copy(); Tlim = gN.copy()
Thim[gN<Th] = 0;   Tlim[gN<Tl] = 0

gF = np.zeros([P, Q])
for x in range(1,P-1):
    for y in range(1,Q-1):
        a = (np.absolute(Thim[x+1,y]) + np.absolute(Thim[x-1,y]) +
             np.absolute(Thim[x,y+1]) + np.absolute(Thim[x,y+1]) +
             np.absolute(Thim[x+1,y+1]) + np.absolute(Thim[x+1,y-1]) +
             np.absolute(Thim[x-1,y+1]) + np.absolute(Thim[x-1,y-1]))
        if Thim[x,y] != 0: gF[x,y] = Thim[x,y]
        elif a != 0: gF[x,y] = Tlim[x,y]

plt.subplot(131)
plt.imshow(Tlim)
plt.title(r'Low threshold, T_L = %g' % Tl)
plt.subplot(132)
plt.imshow(Thim)
plt.title(r'High threshold, T_H = %g' % Th)
plt.subplot(133)
plt.imshow(gF)
plt.title(r'Canny filter, s = %g, n = %d' % (s, n))
plt.show()
