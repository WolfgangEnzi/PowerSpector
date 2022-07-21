"""

PowerSpector v1.0 - Wolfgang Enzi 2022

"""

import numpy as np
import matplotlib.pylab as plt
from scipy.interpolate import griddata
from matplotlib.widgets import Button

""" Helpfull functions for power spectra """

# Function that generates the space that the 2D Gaussian Random Field will live on
def get_grids(xL,yL,xN,yN):

        x = np.linspace(-xL/2.0,xL/2.0,xN)
        y = np.linspace(-xL/2.0,xL/2.0,xN)

        xk = np.fft.fftfreq(xN)* np.pi*2.0 /xL
        yk = np.fft.fftfreq(yN)* np.pi*2.0 /yL

        xy =  np.array([np.ones( (yN))[:,np.newaxis]*x[np.newaxis,:],np.ones((xN))[np.newaxis,:]*y[:,np.newaxis]])

        kxky =  np.array([np.ones( (yN))[:,np.newaxis]*xk[np.newaxis,:],np.ones((xN))[np.newaxis,:]*yk[:,np.newaxis]])

        return xy , kxky

# Function that generates the 2D Gaussian Random Field that is displayed
def get_data(k,k_plot,Pk,seed=24234):
    np.random.seed(seed)
    white = np.random.normal(0,1,k.shape)
    fft_white = np.fft.fft2(white)
    Pk_ = griddata(k_plot,Pk,k,fill_value=0.0)
    fft_colored =  fft_white.real * np.sqrt(Pk_) +  1j * fft_white.imag * np.sqrt(Pk_)
    colored = (np.fft.ifft2(fft_colored)).real
    return colored

""" Define the main parameters """

#  Parameters for the box setup
xL = 1.0
yL = 1.0
xN = 100
yN = 100

# first random seed is chosen random
seed = np.random.poisson(100)

xy, kxky = get_grids(xL,yL,xN,yN)
k = np.sqrt( np.sum(kxky**2,axis=0) )

ipos = k>0
kN = 12

k_plot = np.exp( np.linspace(np.log(np.min(k[ipos])),np.log(np.max(k))*0.7,kN) )
dk = k_plot[1]- k_plot[0]

# the default power spectrum
ndefault = 4
Pk = (k_plot)**-ndefault

field_data = get_data(k,k_plot,Pk,seed)

""" Plotting of the two subplots """

fig = plt.figure(figsize=(10,5))

ax2 = fig.add_subplot(122)

im2 = ax2.imshow(field_data,cmap="RdYlBu")
cbar = plt.colorbar(im2,fraction=0.046, pad=0.04)
plt.xlabel("x")
plt.ylabel("y")
ax = fig.add_subplot(121)

plt.yscale("log")
plt.xscale("log")
im, = ax.plot(k_plot,Pk,drawstyle="steps-mid")

plt.xlabel("k")
plt.ylabel("P(k)")

""" Clicking in the Histogram """

# define function that changes the power in the histogram plot
def onclick(event):
    if event.inaxes == ax:
        global ix, iy, im, Pk, seed
        ix, iy = event.xdata, event.ydata

        idx = np.argmin( (ix - k_plot)**2 )
        idx = int( np.clip(idx,0,kN-1) )
        Pk[idx] = iy
        im.set_ydata(  Pk )
        
        ###
        
        field_data = get_data(k,k_plot,Pk,seed)
        im2.set_array(field_data)
        im2.set_clim(np.min(field_data),np.max(field_data))
        
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        
cid = fig.canvas.mpl_connect('button_press_event', onclick)

""" Reset Button """

Resetax= plt.axes([0.1, 0.9, 0.1, 0.04])
Resetbutton = Button(Resetax, 'Reset', hovercolor='0.975')

# define what happens when you click on reset
def reset(event):
    global Pk, im, im2,field_data, seed
    Pk = (k_plot)**-ndefault
    im.set_ydata(  Pk )

    ###

    field_data = get_data(k,k_plot,Pk,seed)
    im2.set_array(field_data)
    im2.set_clim(np.min(field_data),np.max(field_data))

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

Resetbutton.on_clicked(reset)

""" Change Seed Button """

Seedax= plt.axes([0.25, 0.9, 0.15, 0.04])
Seedbutton = Button(Seedax, 'Change Seed', hovercolor='0.975')

# define what happens when you click on Change Seed
def seed_update(event):
    global Pk, im, im2,field_data, seed
    seed += 1
    
    ###

    field_data = get_data(k,k_plot,Pk,seed)
    im2.set_array(field_data)
    im2.set_clim(np.min(field_data),np.max(field_data))

    fig.canvas.draw_idle()
    fig.canvas.flush_events()

Seedbutton.on_clicked(seed_update)

""" Show figure on screen """

plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()




