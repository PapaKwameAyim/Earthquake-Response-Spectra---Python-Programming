import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz

filePath1 = r"C:\Users\Nana Kofi\Desktop\Python Struct. Dynamics\RSN31_PARKF_C08050.xlsx"
df = pd.read_excel(filePath1)

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

df ["NM_Displacement"]  =  newmark_method(2, 10, 1000, 0, 0, 0.25, 0.50)[0]
df ["NM_Velocity"]      =  newmark_method(2, 10, 1000, 0, 0, 0.25, 0.50)[1]
df ["NM_Acceleration"]  =  newmark_method(2, 10, 1000, 0, 0, 0.25, 0.50)[2]

fig1, axes1 = plt.subplots ( figsize = (30, 20), nrows =3, ncols =1)

axes1[0].set_xlabel("time", fontsize = 20)
axes1[0].set_ylim(-0.5,0.5) # (x,y) affects how high or low the curves go
axes1[0].plot(df["dt"], 100*df["NM_Displacement"], label = 'Displacemnent (t) ', color = 'r', linestyle = '-', linewidth = 2)
axes1[0].set_title('System Displacement Against Time', fontsize =20)
axes1[0].legend()

axes1[1].set_xlabel("time", fontsize = 20)
axes1[1].set_ylim(-0.5,0.5) # (x,y) affects how high or low the curves go
axes1[1].plot(df["dt"], 3*df["NM_Velocity"], label = 'Velocity (t) ', color = 'b', linestyle = '-', linewidth = 2)
axes1[1].set_title('System Velocity Against Time', fontsize =20)
axes1[1].legend()

axes1[2].set_xlabel("time", fontsize = 20)
axes1[2].set_ylim(-5,5) # (x,y) affects how high or low the curves go
axes1[2].plot(df["dt"], 1.2*df["NM_Acceleration"], label = 'Acceleration (t) ', color = 'g', linestyle = '-', linewidth = 2)
axes1[2].set_title('System Acceleration Against Time', fontsize =20)
axes1[2].legend()

plt.figure(figsize = (20, 15) )
plt.xlabel("time", fontsize = 20)
plt.ylim(-5,5) # (x,y) affects how high or low the curves go
plt.plot(df["dt"], 0.5*df["NM_Displacement"], label = 'Displacement (t) ', color = 'r', linestyle = '-', linewidth = 2)
plt.plot(df["dt"], 10*df["NM_Velocity"], label = 'Velocity (t) ', color = 'b', linestyle = '-', linewidth = 2)
plt.plot(df["dt"], 1*df["NM_Acceleration"], label = 'Acceleration (t) ', color = 'g', linestyle = '-', linewidth = 2)
plt.title('System Acceleration Against Time', fontsize =20)
plt.legend()

def annotate_peak(ax, x, y, factor, label):
    peak_index = np.argmax(y)
    peak_value = y[peak_index]
    peak_time = x[peak_index]
    ax.scatter(peak_time, peak_value, color = 'r', s = 300, edgecolor = 'k', zorder = 8)
    ax.annotate(f'{peak_value:.2f}', xy = (peak_time, peak_value), xytext = (peak_time, peak_value), fontsize = 18, color = 'k')
    
annotate_peak(axes1[0], df["dt"], 150*df['NM_Displacement'], 100,  r'u$_g$(t)')
annotate_peak(axes1[1], df["dt"], 10*df['NM_Velocity'], 10, r'$\dot{u}_g(t)$')
annotate_peak(axes1[2], df["dt"], 0.015*df['NM_Acceleration'], 1, r'$\dot{\dot{u}}_g(t)$')

plt.show()