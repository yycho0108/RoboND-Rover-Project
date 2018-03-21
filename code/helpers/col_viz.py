#!/usr/bin/env python2

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb

def get_color(T,R,Z):
    if not isinstance(T, np.ndarray):
        T = np.full_like(R, T)
    if not isinstance(R, np.ndarray):
        R = np.full_like(Z, R)
    if not isinstance(Z, np.ndarray):
        Z = np.full_like(T, Z)
    HSV = np.transpose((T.ravel() / (2*np.pi), R.ravel()/256., Z.ravel()/256.))
    C = hsv_to_rgb(HSV)
    C = np.reshape(C, list(T.shape) + [3])
    return C

def pi3(h,s,v, ax=None, label=''):
    if ax is None:
        ax = plt.gca(projection='3d')

    hl,hh = h
    th0, th1 = np.deg2rad(hl*2), np.deg2rad(hh*2)

    sl,sh = s
    vl,vh = v

    r = np.linspace(sl, sh) # R = saturation
    t = np.linspace(th0, th1) # T = HUE
    z = np.linspace(vl, vh) # Z = VALUE

    R, T = np.meshgrid(r,t)
    X = R * np.cos(T)
    Y = R * np.sin(T)
    Z = np.ones_like(X) * vl
    
    ax.plot_surface(X,Y,Z, facecolors=get_color(T,R,vl))
    Z = np.ones_like(X) * vh
    ax.plot_surface(X,Y,Z, facecolors=get_color(T,R,vh))

    T,Z = np.meshgrid(t,z)
    X = sl * np.cos(T)
    Y = sl * np.sin(T)
    ax.plot_surface(X,Y,Z, facecolors=get_color(T,sl,Z))
    X = sh * np.cos(T)
    Y = sh * np.sin(T)
    ax.plot_surface(X,Y,Z, facecolors=get_color(T,sh,Z))

    R,Z = np.meshgrid(r,z)
    X = R * np.cos(th0)
    Y = R * np.sin(th0)
    ax.plot_surface(X,Y,Z, facecolors=get_color(th0,R,Z))
    X = R * np.cos(th1)
    Y = R * np.sin(th1)
    ax.plot_surface(X,Y,Z, facecolors=get_color(th1,R,Z))

    r = (sl+sh)/2.0
    t = (th0+th1)/2.0

    x = r*np.cos(t)
    y = r*np.sin(t)
    z = (vl+vh)/2.0

    ax.text(x,y,z, label, size=10, zorder=999,
            color='cyan'
            )


    ##rg, tg, zg = np.meshgrid(r, t, z)
    #rg, tg = np.meshgrid(r, t)
    ##print rg.shape

    #x = rg * np.cos(tg)
    #y = rg * np.sin(tg)
    ##z = zg
    #z = np.repeat(z[np.newaxis,:], 50, axis=0)

    #ax.plot_surface(x,y,z)
    ##ax.scatter(x.ravel(), y.ravel(), z.ravel())

    ##u = np.linspace(th0, th1)
    ##v = np.linspace(sl, sh)
    ##z = np.linspace(vl, vh) # heights


    ##x = np.outer(np.cos(u), v)
    ##y = np.outer(np.sin(u), v)
    ##z = np.outer(np.ones(np.size(u)), 

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

pi3((0,50), (0,50), (180,256), ax=ax, label='nav')
pi3((0,35), (0,256), (1,90), ax=ax, label='obs')
pi3((20,30), (230,256), (100,230), ax=ax, label='roc')

t = np.linspace(-np.pi, np.pi)
z = np.linspace(0, 256)
t, z = np.meshgrid(t,z)
x = 256 * np.cos(t)
y = 256 * np.sin(t)
ax.plot_surface(x,y,z, alpha=0.1)

ax.set_xlabel(r'$s \cdot cos(h)$')
ax.set_ylabel(r'$s \cdot sin(h)$')
ax.set_zlabel(r'$v$')
ax.set_title('Colorspace Visualization')

plt.show()
    

#a = np.logical_and
#h, s, v = np.indices((180,256,256))
#
#nav = a(0<=h,h<=50) & a(0<=s,s<=50) & a(180<=v,v<=256)
#obs = a(0<=h,h<=35) & a(0<=s,s<=256) & a(1<=v,v<=90)
#roc = a(20<=h,h<=30) & a(230<=s,s<=256) & a(100<=v,v<=230)
#
#voxels = (nav | obs | roc)
#
#fig = plt.figure()
#ax = fig.gca(projection='3d')
##ax = fig.add_subplot(111, projection='3d')
#
#ax.plot_surface(nav
#plt.show()
