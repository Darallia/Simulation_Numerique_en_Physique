# Simulation_Numerique_en_Physique

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn')
from copy import copy as copy

def schrodinger(K, a, Ti = 0, T = 200, N = 1024, L = 40, dt = 0.01):
    temps = np.arange(0, T, dt)
    x = np.linspace(-L/2 + L/N, L/2, N+1)
    m = np.linspace(-N/2, N/2, N+1)
    p = np.zeros((len(x), len(temps)), dtype="complex")
    p[:,0] = 0.5 + 0.01*np.cos(2*np.pi*x/L)

    for i, t in enumerate(temps[:-1]):
        g = np.exp(-1j*K*dt*np.abs(p[:, i])**2)*p[:, i]
        g_m = (1/N)*np.fft.fftshift(np.fft.fft(g))
        p_m = np.exp(dt*(-1j/2)*(2*m*np.pi/L)**2)*g_m
        p[:, i+1] = N*np.fft.ifft(np.fft.ifftshift(p_m))

    if a == 0:
        u, v = np.meshgrid(x, temps)
        c = np.linspace(np.min(np.min(np.abs(p))), np.max(np.max(np.abs(p))), 101)
        plt.contourf(u, v, np.abs(np.transpose(p)), c, cmap = 'jet')

        plt.xlabel('x')
        plt.ylabel('t')
        plt.title("amplitude de psi(x, t) pour K = {}".format(K))
        plt.colorbar()
        plt.show()
        
    if a == 1:
        fig, ax =plt.subplots()
        fig.suptitle("Graphes de l'amplitude de psi(x,t), à des intervalles de temps, avec K = {}".format(K))
        ax.plot(x, np.abs(p[:, int(0/dt)]), c ='k', label = "Amplitude au temps 0")
        ax.plot(x, np.abs(p[:, int(25/dt)]), c ='m', label = "Amplitude au temps 25")
        ax.plot(x, np.abs(p[:, int(50/dt)]), c ='y', label = "Amplitude au temps 50")
        ax.plot(x, np.abs(p[:, int(75/dt)]), c ='g', label = "Amplitude au temps 75")
        ax.plot(x, np.abs(p[:, int(100/dt)]), c ='b', label = "Amplitude au temps 100")
        plt.legend(['t = 0', 't = 25', 't = 50', 't = 75', 't = 100'], loc ='upper left')
        plt.xlabel('x')
        plt.ylabel('amplitude de psi(x,t)')
        
    if a ==2:
        plt.title("Graphe de l'amplitude de psi(x,{}) avec K = -1".format(Ti))
        plt.plot(x, np.abs(p[:, int(Ti/dt)]), c ='k', label = "Amplitude au temps {}".format(Ti))
        plt.xlabel('x')
        plt.ylabel('aplitude de psi(x, {})'.format(Ti))
      
def peregrine(K, T, N, L, dt):
    temps = np.arange(-T, T+dt, dt)
    x = np.linspace(-L/2 + L/N, L/2, N+1)
    m = np.linspace(-N/2, N/2, N+1)
    p = np.zeros((len(x), len(temps)), dtype = "complex")
    p2 = copy(p)
    p[:, 0] = np.exp(1j*temps[0])*(1-(4*(1+2j*temps[0]))/(1+4*x**2+4*temps[0]**2))
    p2[:, 0] = copy(p[:, 0])
    
    for i, t in enumerate(temps[:-1]):
        g = np.exp(-1j*K*dt*np.abs(p[:, i])**2)*p[:, i]
        g_m = (1/N)*np.fft.fftshift(np.fft.fft(g))
        p_m = np.exp(dt*(-1j/2)*(2*m*np.pi/L)**2)*g_m
        p[:, i+1] = N*np.fft.ifft(np.fft.ifftshift(p_m))
        p2[:, i+1] = np.exp(1j*t)*(1-(4*(1+2j*t))/(1+4*x**2+4*t**2))
        
   
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (15, 10))
    fig.suptitle('Comparaison entre Peregrine exacte et simulée')
    plt.gcf().subplots_adjust(left = 0.1, bottom = 0.1, right = 0.9, top = 0.9, wspace = 0.2, hspace = 0.2)


    u, v = np.meshgrid(x, temps)
    c1 = np.linspace(np.min(np.min(np.abs(p))),np.max(np.max(np.abs(p))),101)
    c2 = np.linspace(np.min(np.min(np.abs(p2))),np.max(np.max(np.abs(p2))),101)
    a1 = ax1.contourf(u, v, np.abs(np.transpose(p)), c1, cmap = 'jet')
    a2 = ax2.contourf(u, v, np.abs(np.transpose(p2)), c2, cmap = 'jet')
    ax1.set(xlabel = "x", ylabel = "t")
    ax2.set(xlabel = "x", ylabel = "t")
    ax1.set_title("simulation avec Peregrine en CI")
    ax2.set_title("Peregrine")

    fig.colorbar(a2, ax=ax2)
    plt.show()
