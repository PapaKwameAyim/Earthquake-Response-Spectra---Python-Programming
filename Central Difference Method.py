# Central Difference
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid as cumtrapz

filePath1 = r"C:\Users\Nana Kofi\Desktop\Python Struct. Dynamics\RSN31_PARKF_C08050.xlsx"
df = pd.read_excel(filePath1)

def central_diff(m,c,k,u0,v0):
    m =2 # in tonnes
    g=9.81
    k=400 # N/m
    c=20
    dt=0.01
    
    U, U[0] = [0]*df.shape[0], u0
    V, V[0] = [0]*df.shape[0], v0
    A = [0]*df.shape[0]
    
    df['Force'] = (-m * g * df["Ground Acceleration"])
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

df["CDM_Displacement"]  = central_diff(2, 10, 1000, 0, 0)[0]
df["CDM_Velocity"]      = central_diff(2, 10, 1000, 0, 0)[1] 
df["CDM_Acceleration"]  = central_diff(2, 10, 1000, 0, 0)[2]

fig1, axes1 = plt.subplots ( figsize = (30, 20), nrows =3, ncols =1)

axes1[0].set_xlabel("time", fontsize = 20)
axes1[0].set_ylim(-2,2) # (x,y) affects how high or low the curves go
axes1[0].plot(df["dt"], 30*df["CDM_Displacement"], label = 'Displacemnent (t) ', color = 'm', linestyle = '-', linewidth = 2)
axes1[0].set_title('System Displacement Against Time', fontsize =20)
axes1[0].legend()

axes1[1].set_xlabel("time", fontsize = 20)
axes1[1].set_ylim(-10,10) # (x,y) affects how high or low the curves go
axes1[1].plot(df["dt"], 3*df["CDM_Velocity"], label = 'Velocity (t) ', color = 'b', linestyle = '-', linewidth = 2)
axes1[1].set_title('System Velocity Against Time', fontsize =20)
axes1[1].legend()

axes1[2].set_xlabel("time", fontsize = 20)
axes1[2].set_ylim(-50,50) # (x,y) affects how high or low the curves go
axes1[2].plot(df["dt"], 0.1*df["CDM_Acceleration"], label = 'Acceleration (t) ', color = 'g', linestyle = '-', linewidth = 2)
axes1[2].set_title('System Acceleration Against Time', fontsize =20)
axes1[2].legend()

plt.figure(figsize = (20, 15) )
plt.xlabel("time", fontsize = 20)
plt.ylim(-50,50) # (x,y) affects how high or low the curves go
plt.plot(df["dt"], 0.05*df["CDM_Acceleration"], label = 'Acceleration (t) ', color = 'r', linestyle = '-', linewidth = 2)
plt.plot(df["dt"], 10*df["CDM_Velocity"], label = 'Velocity (t) ', color = 'b', linestyle = '-', linewidth = 2)
plt.plot(df["dt"], 150*df["CDM_Displacement"], label = 'Displacemnent (t) ', color = 'g', linestyle = '-', linewidth = 2)
plt.title('System Acceleration Against Time', fontsize =20)
plt.legend()


def annotate_peak(ax, x, y, factor, label):
    peak_index = np.argmax(y)
    peak_value = y[peak_index]
    peak_time = x[peak_index]
    ax.scatter(peak_time, peak_value, color = 'r', s = 500, edgecolor = 'k', zorder = 8)
    ax.annotate(f'{peak_value:.2f}', xy = (peak_time, peak_value), xytext = (peak_time, peak_value), fontsize = 18, color = 'k')
    
annotate_peak(axes1[0], df["dt"], 150*df['CDM_Displacement'], 100,  r'u$_g$(t)')
annotate_peak(axes1[1], df["dt"], 10*df['CDM_Velocity'], 10, r'$\dot{u}_g(t)$')
annotate_peak(axes1[2], df["dt"], 0.015*df['CDM_Acceleration'], 1, r'$\dot{\dot{u}}_g(t)$')
