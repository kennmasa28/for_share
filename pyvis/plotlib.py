import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axisartist.grid_finder import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors

def change_axis_width(ax,width,direct='in',pad=10,length=7):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(width)
    ax.xaxis.set_tick_params(direction=direct,which='major',width=width,length=length)
    ax.yaxis.set_tick_params(direction=direct,which='major',width=width,length=length)

    ax.xaxis.set_tick_params(direction=direct,which='minor',width=width,length=0.5*length)
    ax.yaxis.set_tick_params(direction=direct,which='minor',width=width,length=0.5*length)

    ax.tick_params(axis='both', which='major', pad=pad)
    
# Note: if you use tex command, then you may get error message
def save_eps(filename):
    print('Saving eps file: '+filename)
    plt.savefig(filename)
    #fig.savefig(filename)
    #plt.rcParams['text.usetex'] = False
    plt.rcParams.update(plt.rcParamsDefault)

def reset_rcParams():
    plt.rcParams.update(plt.rcParamsDefault)

# Use this command BEFORE you use tex command like \int
def prep_tex():
    plt.rcParams['pdf.use14corefonts'] = True
    plt.rcParams['text.usetex'] = True
    plt.rcParams['ps.useafm'] = True
    plt.rcParams['text.latex.preamble']=[r'\usepackage{amsmath}',r'\boldmath']
    plt.rcParams['axes.unicode_minus'] = False
    #plt.rcParams['text.latex.preamble']=[r'\usepackage{sfmath} \boldmath']

def AddFontCommand(label):
    return r'\textrm{\textbf{' + label + '}}' ## r is necessary
    #return r'\textrm{' + label + '}' ## r is necessary

## Usage: for example,
## > var,x,y,time = athena_read.Get2Ddata('OrszagTang.out2.00100.athdf','rho')
## > plt.ion()
## > fig, ax = plt.subplots()
## > im = plot_cart(x,y,var,fig,ax)
## Ideally, x, y should be the face-centered coordinates
## (x1f(nx+1) and x2f(ny+1)) as pcolormesh uses 
## https://matplotlib.org/devdocs/gallery/images_contours_and_fields/pcolormesh_grids.html
def plot_cart(x,y,var,fig,ax,**kwargs):
    vmin     = kwargs.get('vmin',var.min())
    vmax     = kwargs.get('vmax',var.max())
    colormap = kwargs.get('cmap','viridis')
    log      = kwargs.get('log',0)
    time     = kwargs.get('time',1.)
    xlabel   = kwargs.get('xlabel','x')
    ylabel   = kwargs.get('ylabel','y')
    aspect   = kwargs.get('aspect','auto')
    if(log == 0):
        cnorm = colors.Normalize(vmin=vmin,vmax=vmax)
    else:
        cnorm = colors.LogNorm(vmin=vmin,vmax=vmax)

    nx = var.shape[1]; ny = var.shape[0]
    x1 = x.copy(); y1 = y.copy()
    # if (x1.size == nx):
    #     x1=np.append(x1,x1[-1]+0.5*(x1[-1]-x1[-2]))
    #     print('x is appended')
    # if (y1.size == ny):
    #     y1=np.append(y1,y1[-1]+0.5*(y1[-1]-y1[-2]))
    #     print('y is appended')

    xlim     = kwargs.get('xlim',[x1.min(),x1.max()])
    ylim     = kwargs.get('ylim',[y1.min(),y1.max()])

    #X, Y = np.meshgrid(y, x)
    X, Y = np.meshgrid(x1,y1)
    #X, Y = np.meshgrid(x,y)
    fig.subplots_adjust(bottom=0.2,top=0.9,left=0.2,right=0.80)
    #im = ax.pcolormesh(x, y, var, 
    im = ax.pcolormesh(X, Y, var, 
            cmap=colormap,norm=cnorm,
            shading='nearest')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    #ax.set_aspect('equal')
    ax.set_aspect(aspect=aspect)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if(len(ax.texts)>0): ax.texts[0].remove() 
    if 'time' in kwargs:
        time = kwargs['time']
        text0 = 'Time = '+'{:.2f}'.format(time)
        ax.text(0.0, 1.03, text0, transform = ax.transAxes)
    
    ## remove cax from fig.axes
    ## (I assume that cax is the last member in fig.axes)
    if (len(fig.axes)>1):
        fig.axes[-1].remove() 
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.5)
    cb  = fig.colorbar(im, cax=cax, orientation='vertical')
    #fig.colorbar(im,orientation='vertical')

    return im
