#coding:utf-8
import athena_read
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize , LogNorm
import plotlib

######### plot information #####################
ng = 0
fpass = './test04/data/flare03.out2.00068.athdf'
fpassu = './test04/data/flare03.out3.00068.athdf'
fsize = 20
title = r'$|c_{s}|$'
max1 = 3.0
min1 = 0.0
# max2 = 5.0
# min2 = 0.0

############# prepare #########################
plotlib.prep_tex()
plt.rcParams["font.size"] = fsize
fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(1, 1, 1)
# ax2 = fig.add_subplot(1, 2, 2)
data = athena_read.athdf(fpass,num_ghost=ng)
datau = athena_read.athdf(fpassu,num_ghost=ng)

########## load (for xy slice) ###############
#xmin, xmax, ymin, ymax, zmin, zmax = 0, 180, 0, 240, 0, 1
#xmin, xmax, ymin, ymax, zmin, zmax = 0, 900, 0, 1200, 0, 1
#xmin, xmax, ymin, ymax, zmin, zmax = 150, 750, 100, 800, 0, 20
xmin, xmax, ymin, ymax, zmin, zmax = 350, 550, 450, 650, 0, 1
#xmin, xmax, ymin, ymax, zmin, zmax = 940, 1622, 240, 1599, 0, 1
#xmin, xmax, ymin, ymax, zmin, zmax = 0, 2400, 0, 1599, 0, 1
#xmin, xmax, ymin, ymax, zmin, zmax = 0,1280,0,800,0,1
#xmin, xmax, ymin, ymax, zmin, zmax = 0,640,0,400,0,1
#xmin, xmax, ymin, ymax, zmin, zmax = 0,512,0,256,0,1
x = data['x1f'][xmin:xmax]
y = data['x2f'][ymin:ymax]
z = data['x3f'][ng:len(data['x3f'])-1-ng]
p = data['press'][zmin:zmax, ymin:ymax, xmin:xmax]
rho = data['rho'][zmin:zmax, ymin:ymax, xmin:xmax]
vx = data['vel1'][zmin:zmax, ymin:ymax, xmin:xmax]
vy = data['vel2'][zmin:zmax, ymin:ymax, xmin:xmax]
vz = data['vel3'][zmin:zmax, ymin:ymax, xmin:xmax]
Bx = data['Bcc1'][zmin:zmax, ymin:ymax, xmin:xmax]
By = data['Bcc2'][zmin:zmax, ymin:ymax, xmin:xmax]
Bz = data['Bcc3'][zmin:zmax, ymin:ymax, xmin:xmax]
rho = datau['user_out_var0'][zmin:zmax,ymin:ymax, xmin:xmax]
beta = datau['user_out_var8'][zmin:zmax,ymin:ymax, xmin:xmax]
J = datau['user_out_var9'][zmin:zmax,ymin:ymax, xmin:xmax]
B = np.sqrt(Bx**2 + By**2 + Bz**2)
t = data['Time']
cs = np.sqrt(p/rho)


########### plot #################
#cf1 = ax1.pcolormesh(x[:],y[:],J[0,:,:] ,norm=LogNorm(vmin=min1, vmax=max1), cmap='plasma', shading='nearest')
#cf1 = ax1.pcolormesh(x,y,eta[0,:,:] ,norm=Normalize(vmin=min1, vmax=max1), cmap='seismic', shading='nearest')
#cf1 = ax1.pcolormesh(x,y,vy[0,:,:] ,norm=Normalize(vmin=min1, vmax=max1), cmap='seismic', shading='nearest')
#cf1 = ax1.pcolormesh(x[:],y[:],p[0,:,:] ,norm=Normalize(vmin=min1, vmax=max1), cmap='gist_heat', shading='nearest')
#cf1 = ax1.pcolormesh(x,y,Bx ,norm=Normalize(vmin=min1, vmax=max1), cmap='seismic', shading='nearest')
#cf1 = ax1.pcolormesh(x,y,beta ,norm=LogNorm(vmin=min1, vmax=max1), cmap='hot_r', shading='nearest')
cf1 = ax1.pcolormesh(x,y,cs[0,:,:] ,norm=Normalize(vmin=min1, vmax=max1), cmap='plasma', shading='nearest')
ax1.plot(np.linspace(-0.45, -0.25, 100), 12.2*np.ones(100), '-', color='g')
#cf1 = ax1.pcolormesh(x[:-1],y[:-1],jz[0,:,:] ,norm=Normalize(vmin=min1, vmax=max1), cmap='seismic', shading='nearest')
pp1 = fig.colorbar(cf1, ax=ax1, orientation="vertical", shrink=1.0)
cf1.set_clim(min1,max1)
ax1.set_title(title,fontsize=fsize)
xi = np.linspace(x.min(), x.max(), x.size)
yi = np.linspace(y.min(), y.max(), y.size)
ax1.streamplot(xi, yi, Bx[0,:,:], By[0,:,:] ,density=2, color='k', arrowstyle='-', linewidth=1.0)


# cf2 = ax2.pcolormesh(x,y,Bz ,norm=Normalize(vmin=min2, vmax=max2), cmap='plasma',shading='nearest')
# pp2 = fig.colorbar(cf2, ax=ax2, orientation="vertical")
# cf2.set_clim(min2,max2)
# ax2.set_xlabel("x",fontsize=14)
# ax2.set_ylabel("y",fontsize=14)
# ax2.set_title('temperature',fontsize=16)

step = str(np.round(t,decimals=1))
ax1.text(2,y[-1]+0.8,'time='+ step,fontsize=18)
ax1.set_xlabel(plotlib.AddFontCommand(r'$z\ [3,000\ \rm{km}]$'))
ax1.set_ylabel(plotlib.AddFontCommand(r'$y\ [3,000\ \rm{km}]$'))
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
ax1.set_aspect('equal')
plt.subplots_adjust(left=0.2, right=0.8, bottom=0.40, top=0.60)
plt.tight_layout()
plt.show()
#print(t*17.6)