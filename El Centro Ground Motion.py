# El Centro Ground Motion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import cumulative_trapezoid as cumtrapz

import warnings
warnings.filterwarnings(action ="ignore", category = RuntimeWarning)

filePath1 = r"C:\Users\Nana Kofi\Desktop\Python Struct. Dynamics\RSN31_PARKF_C08050.xlsx"
df = pd.read_excel(filePath1)


g = 9.81 # in m / s**2
df["Ground Velocity"] = cumtrapz(g*df["Ground Acceleration"], df["dt"], initial = 0)
df["Ground Displacement"] = cumtrapz(g*df["Ground Velocity"], df["dt"], initial = 0)

fig1,axes1 = plt.subplots (figsize = (30,20), nrows = 3, ncols = 1)

axes1[0].set_xlabel('time', fontsize = 20)
axes1[0].set_ylim(-8,8) # (x,y) affects how high or low the curves go
axes1[0].plot(df["dt"], 15*df["Ground Displacement"], label = r'u$_g$(t)', color = 'm', linestyle = '-', linewidth = 2)
axes1[0].set_title('Ground Displacement Against Time', fontsize =20)
axes1[0].legend()

axes1[1].set_xlabel('time', fontsize = 20)
axes1[1].set_ylim([-8,8]) # (x,y) affects how high or low the curves go
axes1[1].plot(df["dt"], 65*df["Ground Velocity"], label = r'$\dot{u}_g(t)$', color = 'g', linestyle = '-', linewidth = 2)
axes1[1].set_title('Ground Veloctiy Against Time', fontsize =20)
axes1[1].legend()

axes1[2].set_xlabel('time', fontsize =20)
axes1[2].set_ylim([-8,8]) # (x,y)  affects how high or low the curves go
axes1[2].plot( df["dt"], 30*df["Ground Acceleration"], label = r'$\dot{\dot{u}}_g(t)$', color = 'b', linestyle = '-', linewidth = 2)
axes1[2].set_title('Ground Acceleration Against Time', fontsize =20)
axes1[2].legend()

def annotate_peak(ax, x, y, factor, label):
    peak_index = np.argmax(y)
    peak_value = y[peak_index]
    peak_time = x[peak_index]
    ax.scatter(peak_time, peak_value, color = 'r', s = 300, edgecolor = 'k', zorder = 8)
    ax.annotate(f'{peak_value:.2f}', xy = (peak_time, peak_value), xytext = (peak_time, peak_value), fontsize = 18, color = 'k')
    
#applying annotation for each subplot
annotate_peak(axes1[0], df["dt"], 16.2*df['Ground Displacement'], 100,  r'u$_g$(t)')
annotate_peak(axes1[1], df["dt"], 68*df['Ground Velocity'], 10, r'$\dot{u}_g(t)$')
annotate_peak(axes1[2], df["dt"], 30*df['Ground Acceleration'], 1, r'$\dot{\dot{u}}_g(t)$')

plt.show()

def central_diff(m,c,k,u0,v0):
    m =2
    g=9.81
    k=400
    c=20
    dt=0.01
    
    U, U[0] = [0]*df.shape[0], u0
    V, V[0] = [0]*df.shape[0], v0
    A = [0]*df.shape[0]
    
    df['Force'] = (-m*g*df["Ground Acceleration"])
    P = df["Force"]
    P_hat = [0]*df.shape[0]
    
    U[-1] = U[0] - (dt*V[0]) + 0.5*A[0]*dt**2
                    
    k_hat = (m/dt**2) + (c/2*dt)
    a = (m/dt**2) - (c/2*dt)
    b = k - ((2*m)/(dt **2))
    
    for i in range (0, df.shape[0]-1):
        P_hat[i] = P[i] - (a*U[i-1]) - (b*U[i])
        U[i+1] = P_hat[i] / k_hat
        V[i] = (U[i+1])- (U[i-1]) / (2*dt)
        A[i] = (U[i+1]) - (2*U[i]) + (U[i-1]) / (dt**2)
        
    return U, V, A

def newmark_method(m,c,k,u0, v0, beta, gamma):
    g = 9.81
    dt = 0.01
    
    df["Force"] = (-m*g*df["Ground Acceleration"])
    P = df["Force"]
    
    n = df.shape[0]
    
    A = [0] *df.shape[0]
    V = [0] *df.shape[0]
    U = [0] *df.shape[0]
    P_hat = [0] *df.shape[0]
    
    A[0] = ( (P[0] - (c*V[0]) - k*U[0]) ) / m
    
    a1 = (m/(beta*(dt**2))) + ((gamma*c)/(beta*dt))
    a2 = (m/(beta*dt)) + ( (( gamma/beta) -1 ) * c)
    a3 = ( (( 1/(2*beta)) -1)*m) + (((gamma/(2*beta)) -1)*c*dt)    
    k_hat = k +a1
    
    for i in range (n -1):
        P_hat[i+1] = P[i+1] + a1*U[i] + a2*V[i] + a3*A[i]
        U[i+1] = P_hat[i+1]/k_hat
        V[i+1] = (gamma/(beta*dt)) * (U[i+1] - U[i]) + (1- (gamma/beta)) * V[i] + dt * (1-gamma/(2*beta)) * A[i]
        A[i+1] = ((U[i+1] - U[i]) / (beta * dt**2)) - (V[i] / (beta*dt)) - ((1/ (2*beta)) -1) * A[i]
        
    return U, V, A

df["CDM_Displacement"], df ["NM_Displacement"] = central_diff(2, 10, 1000, 0, 0)[0], newmark_method(2, 10, 1000, 0, 0, 0.25, 0.50)[0]
df["CDM_Velocity"], df ["NM_Velocity"] = central_diff(2, 10, 1000, 0, 0)[1], newmark_method(2, 10, 1000, 0, 0, 0.25, 0.50)[1]
df["CDM_Acceleration"], df ["NM_Acceleration"] = central_diff(2, 10, 1000, 0, 0)[2], newmark_method(2, 10, 1000, 0, 0, 0.25, 0.50)[2]

fig1, axes1 = plt.subplots ( figsize = (30, 20), nrows =3, ncols =1)

axes1[0].set_xlabel("time", fontsize = 20)
axes1[0].set_ylim(-8,8) # (x,y) affects how high or low the curves go
axes1[0].plot(df["dt"], 150*df["CDM_Displacement"], label = 'Displacemnent (t) ', color = 'm', linestyle = '-', linewidth = 2)
axes1[0].plot(df["dt"], 150*df["NM_Displacement"], label = 'Displacemnent (t) ', color = 'g', linestyle = '-', linewidth = 2)
axes1[0].set_title('System Displacement Against Time', fontsize =20)
axes1[0].legend()

axes1[1].set_xlabel("time", fontsize = 20)
axes1[1].set_ylim(-8,8) # (x,y) affects how high or low the curves go
axes1[1].plot(df["dt"], 3*df["CDM_Velocity"], label = 'Velocity (t) ', color = 'b', linestyle = '-', linewidth = 2)
axes1[1].plot(df["dt"], 15*df["NM_Velocity"], label = 'Velocity (t) ', color = 'r', linestyle = '-', linewidth = 2)
axes1[1].set_title('System Velocity Against Time', fontsize =20)
axes1[1].legend()

axes1[2].set_xlabel("time", fontsize = 20)
axes1[2].set_ylim(-8,8) # (x,y) affects how high or low the curves go
axes1[2].plot(df["dt"], 0.015*df["CDM_Acceleration"], label = 'Acceleration (t) ', color = 'g', linestyle = '-', linewidth = 2)
axes1[2].plot(df["dt"], 1*df["NM_Acceleration"], label = 'Acceleration (t) ', color = 'r', linestyle = '-', linewidth = 2)
axes1[2].set_title('System Acceleration Against Time', fontsize =20)
axes1[2].legend()


annotate_peak(axes1[0], df["dt"], 150*df['CDM_Displacement'], 100,  r'u$_g$(t)')
annotate_peak(axes1[1], df["dt"], 10*df['CDM_Velocity'], 10, r'$\dot{u}_g(t)$')
annotate_peak(axes1[2], df["dt"], 0.015*df['CDM_Acceleration'], 1, r'$\dot{\dot{u}}_g(t)$')

plt.show()


def central_diff(m, c, k):
    filePath1 = r"C:\Users\Nana Kofi\Desktop\Python Struct. Dynamics\RSN31_PARKF_C08050.xlsx"
    df = pd.read_excel(filePath1)
    g = 9.81
    df['Force'] = (-m * g * df["Ground Acceleration"] )
    P = df['Force']
    
    U = [0] * df.shape[0]
    V = [0] * df.shape[0]
    A = [0] * df.shape[0]
    dt = 0.01
    P_hat = [0] * df.shape[0]
    
    U[-1] = U[0] - (dt * V[0])  + (0.5 * A[0] * dt**2)
    k_hat = (m / dt**2) + (c /2*dt)
    a = (m / dt**2) - (c / 2*dt)
    b = k - (2*m / dt**2)
    
    for i in range(0, df.shape[0]-1):
        P_hat[i] = df['Force'][i] - a*U[i-1] - b*U[i]
        U[i+1] = P_hat[i] / k_hat
        V[i] = (U[i+1] - U[i-1]) / (2*dt)
        A[i] = (U[i+1] - 2*U[i] + U[i-1]) / dt**2
        
    return max(U), max(V), max(A)

dRatio = [0.05, 0.10, 0.20, 0.50]
period = np.arange(0, 0.4, 0.001)
k = 1000
U = []
V = []
A = []

pseudoSpectra_V = []
pseudoSpectra_A = []

for epsilon in dRatio:
    for Tn in period:
        m = (k * Tn/4 * np.pi**2)
        c = (2*epsilon * np.sqrt(k*m))
        
        peak_U = central_diff(m, c, k)[0]*10000
        peak_V = central_diff(m, c, k)[1]*10000
        peak_A = central_diff(m, c, k)[2]*10000
        
        pseudoSpectra_A.append(peak_U * (2*np.pi)/Tn)
        pseudoSpectra_V.append(peak_U * ((2*np.pi)/Tn)**2)
        
        U.append(peak_U)
        V.append(peak_V)
        A.append(peak_A)
        
fig3, axes3 = plt.subplots(figsize = (25,10), ncols = 2, nrows = 2)

axes3[0][0].plot(period, A[0:400], label = 'Damping Ratio 5%', color = 'k', linewidth = 3, linestyle = '-')
axes3[0][0].plot(period, A[400:800], label = 'Damping Ratio 10%', color = 'r', linewidth = 3, linestyle = '-')
axes3[0][0].plot(period, A[800:1200], label = 'Damping Ratio 20%', color = 'g', linewidth = 3, linestyle = '-')
axes3[0][0].plot(period, A[1200:1600], label = 'Damping Ratio 50%', color = 'b', linewidth = 3, linestyle = '-')
axes3[0][0].legend
axes3[0][0].set_title('System Acceleration Response Spectra')

axes3[0][1].plot(period, V[0:400], label = 'Damping Ratio 5%', color = 'black', linewidth = 3, linestyle = '-')
axes3[0][1].plot(period, V[400:800], label = 'Damping Ratio 10%', color = 'red', linewidth = 3, linestyle = '-')
axes3[0][1].plot(period, V[800:1200], label = 'Damping Ratio 20%', color = 'g', linewidth = 3, linestyle = '-')
axes3[0][1].plot(period, V[1200:1600], label = 'Damping Ratio 50%', color = 'blue', linewidth = 3, linestyle = '-')
axes3[0][1].legend
axes3[0][1].set_title('System Velocity Response Spectra')

axes3[1][0].plot(period, U[0:400], label = 'Damping Ratio 5%', color = 'black', linewidth = 3, linestyle = '-')
axes3[1][0].plot(period, U[400:800], label = 'Damping Ratio 10%', color = 'red', linewidth = 3, linestyle = '-')
axes3[1][0].plot(period, U[800:1200], label = 'Damping Ratio 20%', color = 'g', linewidth = 3, linestyle = '-')
axes3[1][0].plot(period, U[1200:1600], label = 'Damping Ratio 50%', color = 'blue', linewidth = 3, linestyle = '-')
axes3[1][0].legend
axes3[1][0].set_title('System Displacement Response Spectra')

axes3[1][1].plot(period, pseudoSpectra_A[0:400], label = 'Damping Ratio 5%', color = 'black', linewidth = 3, linestyle = '-')
axes3[1][1].plot(period, pseudoSpectra_A[400:800], label = 'Damping Ratio 10%', color = 'red', linewidth = 3, linestyle = '-')
axes3[1][1].plot(period, pseudoSpectra_A[800:1200], label = 'Damping Ratio 20%', color = 'g', linewidth = 3, linestyle = '-')
axes3[1][1].plot(period, pseudoSpectra_A[1200:1600], label = 'Damping Ratio 50%', color = 'blue', linewidth = 3, linestyle = '-')
axes3[1][1].legend
axes3[1][1].set_title('Pseudo-Acceleration Response Spectrum')


fig4, axes4 = plt.subplots (figsize = (25,18), nrows = 2, ncols = 2)

def init():
    for ax in axes4.flatten():
            ax.clear()
    return axes4.flatten()

def update(frame):
    for ax in axes4.flatten(): # clear previous data
        ax.clear()
        
    # Updating plots
    axes4[0][0].plot(period[:frame], A[:frame], label = 'Damping Ratio 5%', color = 'k', linewidth = 3, linestyle = '-')
    axes4[0][0].plot(period[:frame], A[400:frame+400], label = 'Damping Ratio 10%', color = 'r', linewidth = 3, linestyle = '-')
    axes4[0][0].plot(period[:frame], A[800:frame+800], label = 'Damping Ratio 20%', color = 'g', linewidth = 3, linestyle = '-')
    axes4[0][0].plot(period[:frame], A[1200:frame+1200], label = 'Damping Ratio 50%', color = 'b', linewidth = 3, linestyle = '-')
    axes4[0][0].legend
    axes4[0][0].set_title('System Acceleration Response Spectra')

    axes4[0][1].plot(period[:frame], V[:frame], label = 'Damping Ratio 5%', color = 'k', linewidth = 3, linestyle = '-')
    axes4[0][1].plot(period[:frame], V[400:frame+400], label = 'Damping Ratio 10%', color = 'r', linewidth = 3, linestyle = '-')
    axes4[0][1].plot(period[:frame], V[800:frame+800], label = 'Damping Ratio 20%', color = 'g', linewidth = 3, linestyle = '-')
    axes4[0][1].plot(period[:frame], V[1200:frame+1200], label = 'Damping Ratio 50%', color = 'b', linewidth = 3, linestyle = '-')
    axes4[0][1].legend
    axes4[0][1].set_title('System Velocity Response Spectra')

    axes4[1][0].plot(period[:frame], U[:frame], label = 'Damping Ratio 5%', color = 'k', linewidth = 3, linestyle = '-')
    axes4[1][0].plot(period[:frame], U[400:frame+400], label = 'Damping Ratio 10%', color = 'r', linewidth = 3, linestyle = '-')
    axes4[1][0].plot(period[:frame], U[800:frame+800], label = 'Damping Ratio 20%', color = 'g', linewidth = 3, linestyle = '-')
    axes4[1][0].plot(period[:frame], U[1200:frame+1200], label = 'Damping Ratio 50%', color = 'b', linewidth = 3, linestyle = '-')
    axes4[1][0].legend
    axes4[1][0].set_title('System Displacement Response Spectra')

    axes4[1][1].plot(period[:frame], pseudoSpectra_A[:frame], label = 'Damping Ratio 5%', color = 'k', linewidth = 3, linestyle = '-')
    axes4[1][1].plot(period[:frame], pseudoSpectra_A[400:frame+400], label = 'Damping Ratio 10%', color = 'r', linewidth = 3, linestyle = '-')
    axes4[1][1].plot(period[:frame], pseudoSpectra_A[800:frame+800], label = 'Damping Ratio 20%', color = 'g', linewidth = 3, linestyle = '-')
    axes4[1][1].plot(period[:frame], pseudoSpectra_A[1200:frame+1200], label = 'Damping Ratio 50%', color = 'b', linewidth = 3, linestyle = '-')
    axes4[1][1].legend
    axes4[1][1].set_title('Pseudo-Acceleration Response Spectrum')
    
    return axes4.flatten()
      
# creating animation
animate = animation.FuncAnimation(fig4, update, frames = len(period), init_func = init, blit = False, interval = 100)
animate.save('ResponseSpectra.gif')

plt.tight_layout()
plt.show()




















