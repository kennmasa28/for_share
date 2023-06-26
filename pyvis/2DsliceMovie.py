#coding:utf-8
import athena_read
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize , LogNorm
import plotlib

######### plot information #####################
ng = 0
fpass = './test04/data/'
ppass = './test04/absj/'
ppass2 = './test04/beta/'
fsize = 20
title = r'$|J|$'
title2 = r'$\beta$'
max1 = 100
min1 = 0.1
max2 = 50
min2 = 0.05
start = 0
end = 200

def makefigure(step):
    ############# prepare #########################
    plotlib.prep_tex()
    plt.rcParams["font.size"] = fsize
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(1, 1, 1)
    #ax2 = fig.add_subplot(1, 2, 2)
    data = athena_read.athdf(fpass + 'flare03.out2.{:0=5}.athdf'.format(step),num_ghost=ng)
    #datau = athena_read.athdf(fpass + 'flare03.out3.{:0=5}.athdf'.format(step),num_ghost=ng)
    ########## load (for xy slice) ###############
    #xmin, xmax, ymin, ymax, zmin, zmax = 940, 1622, 240, 1440, 0, 1
    xmin, xmax, ymin, ymax, zmin, zmax = 0, 900, 0, 1200, 0, 1
    #xmin, xmax, ymin, ymax, zmin, zmax = 0, 600, 0, 800, 0, 1

    x = data['x1f'][xmin:xmax]
    y = data['x2f'][ymin:ymax]
    z = data['x3f'][ng:len(data['x3f'])-1-ng]
    p = data['press'][zmin:zmax, ymin:ymax, xmin:xmax]
    rho = data['rho'][zmin:zmax, ymin:ymax, xmin:xmax]
    #vx = data['vel1'][zmin:zmax, ymin:ymax, xmin:xmax]
    #vy = data['vel2'][zmin:zmax, ymin:ymax, xmin:xmax]
    vz = data['vel3'][zmin:zmax, ymin:ymax, xmin:xmax]
    Bx = data['Bcc1'][zmin:zmax, ymin:ymax, xmin:xmax]
    By = data['Bcc2'][zmin:zmax, ymin:ymax, xmin:xmax]
    Bz = data['Bcc3'][zmin:zmax, ymin:ymax, xmin:xmax]
    xf = data['x1f'].copy()
    yf = data['x2f'].copy()
    zf = data['x3f'].copy()
    dxf = xf[1:] - xf[0:-1]
    dyf = yf[1:] - yf[0:-1]
    dzf = zf[1:] - zf[0:-1]
    # jx = (Bz[:-1,1:,:-1]-Bz[:-1,:-1,:-1])/dyf[0] -  (By[1:,:-1,:-1]-By[:-1,:-1,:-1])/dzf[0]
    # jy = (Bx[1:,:-1,:-1]-Bx[:-1,:-1,:-1])/dzf[0] -  (Bz[:-1,:-1,1:]-Bz[:-1,:-1,:-1])/dxf[0]
    # jz = (By[:-1,:-1,1:]-By[:-1,:-1,:-1])/dxf[0] -  (Bx[:-1,1:,:-1]-Bx[:-1,:-1,:-1])/dyf[0]
    jx = (Bz[:,1:,:-1]-Bz[:,:-1,:-1])/dyf[0] 
    jy =  -  (Bz[:,:-1,1:]-Bz[:,:-1,:-1])/dxf[0]
    jz = (By[:,:-1,1:]-By[:,:-1,:-1])/dxf[0] -  (Bx[:,1:,:-1]-Bx[:,:-1,:-1])/dyf[0]
    j = np.sqrt(jx**2 + jy**2 + jz**2) + 1e-8 #1e-8はログで表示してもエラーが出ないようにするため
    #ju = datau['user_out_var9'][zmin:zmax, ymin:ymax, xmin:xmax]

    pmag = (Bx**2 + By**2 + Bz**2)/2
    #B = np.sqrt(Bx**2 + By**2 + Bz**2)
    beta = p/pmag
    #va = B/np.sqrt(rho)
    t = data['Time']

    #plotjz = jz[:,:,450-270].transpose()


    ########### plot #################
    #cf1 = ax1.pcolormesh(x[:],y[:],ju[0,:,:] ,norm=LogNorm(vmin=min1, vmax=max1), cmap='plasma', shading='nearest')
    cf1 = ax1.pcolormesh(x[:-1],y[:-1],j[0,:,:] ,norm=LogNorm(vmin=min1, vmax=max1), cmap='plasma', shading='nearest')
    #cf1 = ax1.pcolormesh(x[:],y[:],np.log10(rho[0,:,:]) ,norm=Normalize(vmin=min1, vmax=max1), cmap='gist_heat', shading='nearest')
    #cf1 = ax1.pcolormesh(x,y,Bz[0,:,:] ,norm=Normalize(vmin=min1, vmax=max1), cmap='seismic', shading='nearest')
    #cf1 = ax1.pcolormesh(x[:-1],y[:-1],jz[0,:,:] ,norm=Normalize(vmin=min2, vmax=max2), cmap='seismic', shading='nearest')
    #cf1 = ax1.pcolormesh(x,y,beta[0,:,:] ,norm=LogNorm(vmin=min1, vmax=max1), cmap='hot_r', shading='nearest')
    #cf1 = ax1.pcolormesh(z[:-1],y[:-1],plotjz[:,:] ,norm=Normalize(vmin=min1, vmax=max1), cmap='seismic', shading='nearest')
    fig.colorbar(cf1, ax=ax1, orientation="vertical", shrink=1)
    cf1.set_clim(min1,max1)
    ax1.set_title(title,fontsize=fsize)
    step = str(np.round(t*17.6,decimals=1))
    yt=y[-1]+0.8
    tlabel = 'time='+ step + 's'
    ax1.text(2.0,yt,plotlib.AddFontCommand(tlabel),fontsize=24)
    ax1.set_xlabel(plotlib.AddFontCommand(r'$x$'))
    ax1.set_ylabel(plotlib.AddFontCommand(r'$y$'))
    # xi = np.linspace(x.min(), x.max(), x.size)
    # yi = np.linspace(y.min(), y.max(), y.size)
    # ax1.streamplot(xi, yi, Bx[0,:,:], By[0,:,:] ,color='w')



    # cf2 = ax2.pcolormesh(x[:-1],y[:-1],jz[0,:,:] ,norm=Normalize(vmin=min2, vmax=max2), cmap='seismic',shading='nearest')
    # pp2 = fig.colorbar(cf2, ax=ax2, orientation="vertical")
    # cf2.set_clim(min2,max2)
    # ax2.set_xlabel(plotlib.AddFontCommand(r'$x$'))
    # ax2.set_ylabel(plotlib.AddFontCommand(r'$y$'))
    # ax2.set_title(r'$J_{z}$ on xy plane',fontsize=16)

    
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax1.set_aspect('equal')
    #ax2.set_aspect('equal')
    #plt.subplots_adjust(left=0.2, right=0.8, bottom=0.40, top=0.60)
    plt.tight_layout()
    filename='im{:0=4}.png'.format(i-start)
    fig.savefig(ppass + filename)
    plt.close()


    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(1, 1, 1)
    cf1 = ax1.pcolormesh(x,y,beta[0,:,:] ,norm=LogNorm(vmin=min2, vmax=max2), cmap='hot_r', shading='nearest')
    fig.colorbar(cf1, ax=ax1, orientation="vertical", shrink=1)
    cf1.set_clim(min2,max2)
    ax1.set_title(title2,fontsize=fsize)
    step = str(np.round(t*17.6,decimals=1))
    yt=y[-1]+0.8
    tlabel = 'time='+ step + 's'
    ax1.text(2.0,yt,plotlib.AddFontCommand(tlabel),fontsize=24)
    ax1.set_xlabel(plotlib.AddFontCommand(r'$x$'))
    ax1.set_ylabel(plotlib.AddFontCommand(r'$y$'))
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    ax1.set_aspect('equal')
    plt.tight_layout()
    filename='im{:0=4}.png'.format(i-start)
    fig.savefig(ppass2 + filename)

    print(i)


#----make pictures
for i in range(start,end):
    makefigure(i)