import numpy as np
from scipy.fftpack import ifft2
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import time

# Function for ocean wave simulation
# def ocean_wave_3d_func(V, D, mesh_size, patch_size):

# Parameters
V = 6  # 19.5m above ocean surface
D = -60  # Wind direction based on standard coordinates
cover_size = 10  # Cover length /m
P_scale = 8
patch_size = 51.2 * P_scale  # 102m spatial range, determines kx resolution
M_scale = 2  # 324
mesh_size = int(1024 * M_scale) # FFT size (128/256/384/512 for speed)
g = 9.81  # Gravitational constant
length = V * V * 0.9128  # Domain wave length
wind_dir = D * np.pi / 180  # Wind direction in radians
wind_damp = 0.01  # Attenuation
deta_k = 2 * np.pi / patch_size
deta_x = patch_size / mesh_size
A = 0.0081 * patch_size**2  # Amplitude
B = 1 / 8 / np.pi / deta_x

# Generate wave spectrum
nn, mm = np.meshgrid(np.arange(1, mesh_size + 1), np.arange(1, mesh_size + 1))

# Calculate wave vector
kx = (2 * np.pi * (mm - 1 - mesh_size / 2) / patch_size).astype(np.float32)
ky = (2 * np.pi * (nn - 1 - mesh_size / 2) / patch_size).astype(np.float32)
sign_correction = np.mod(mm + nn - 2, 2).astype(bool)

k = np.sqrt(kx**2 + ky**2).astype(np.float32)
# Wind modulation
w_dot_k = (kx / k * np.cos(wind_dir) + ky / k * np.sin(wind_dir)).astype(np.float32)  # Projection of normalized wave number vector on wind direction

# Spectrum at given point
P = (A * np.exp(-0.74 * g**2 / (k**2 * V**4)) * (w_dot_k**2)).astype(np.float32) / 2.36  # Gravitational PM spectrum * cos(theta)
wave_limit = patch_size / 100
P = P * np.exp(-k**2 * wave_limit**2) / k**3
# Filter waves moving in the wrong direction
P[w_dot_k < 0] = P[w_dot_k < 0] * wind_damp
P[np.isnan(P)] = 0

# Calculate initial surface in frequency domain
# RANDN - GAUSSIAN | RAND - NORMAL
H0 = mesh_size / np.sqrt(2) * (np.random.randn(mesh_size, mesh_size) + 1j * np.random.randn(mesh_size, mesh_size)) * np.sqrt(P)
# Get mirrored value of initial surface
Hm = np.rot90(np.conj(H0), 2)  # Equivalent to flipud(fliplr(B))

# Dispersion
W = np.sqrt(g * k)

# Cover parameters
cn = int(np.floor(mesh_size * cover_size / patch_size))  # Cover length /m
vn = int(np.floor(0.5 * patch_size / cover_size))
X, Y = np.meshgrid(np.arange(-cn/2, cn/2), np.arange(-cn/2, cn/2))

ts = 1/50
Num = int(1/ts)
# htt = np.zeros((cn, cn, Num), dtype=np.float32)
second = 5

# Create figure for visualization
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(X*deta_x, Y*deta_x, np.zeros((cn, cn)), cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.set_zlabel('H (m)')
plt.colorbar(surf, ax=ax)
ax.set_aspect('equal')

# Create htt array with the correct size
total_frames = Num * second
htt = np.zeros((cn, cn, total_frames), dtype=np.float32)


# Time loop
for tn in range(1, Num * second + 1):
    time_val = 0 + tn * ts
    
    # Update according to the dispersion relation
    Hkt = H0 * np.exp(1j * W * time_val) + Hm * np.exp(-1j * W * time_val)  # Ensure Hkt has conjugate symmetry
    
    # Generate HeightField at time t using ifft
    Ht = B * np.real(ifft2(Hkt))
    Ht[sign_correction] = -1.0 * Ht[sign_correction]
    
    # Update plot
    if tn % 10 == 0:  # Update plot every 10 frames for speed
        ax.clear()
        surf = ax.plot_surface(X*deta_x, Y*deta_x, Ht[vn*cn:vn*cn+cn, vn*cn:vn*cn+cn], cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_zlabel('H (m)')
        ax.set_title(f'Time: {time_val:.2f}s')
        plt.pause(0.01)
    
    # Store data
    htt[:, :, tn-1] = Ht[vn*cn:vn*cn+cn, vn*cn:vn*cn+cn]
    print(f"Processing frame {tn}/{Num * second}")

# Save data
output_dir = os.path.expanduser('data/ocean_wave_data')
os.makedirs(output_dir, exist_ok=True)

filename = f'TimeVaryWS{V:02d}M{M_scale:03d}P{P_scale:03d}Covs{cover_size:02d}Seco{second:01d}Fps{Num:03d}.npz'
filepath = os.path.join(output_dir, filename)

# Save parameters
params = {
    'windspeed': V,
    'patch_size': patch_size,
    'mesh_size': mesh_size,
    'deta_x': deta_x,
    'fps': Num
}

# Save data and parameters
np.savez(filepath, htt=htt, params=params)
print(f"Data saved to {filepath}")

# plt.show()

# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from mpl_toolkits.mplot3d import Axes3D
# import os
# import time

# # Close all figures and clear variables (similar to MATLAB close all, clear)
# plt.close('all')

# # Define the path and file name
# address = 'data/ocean_wave_data'
# name = 'TimeVaryWS15M002P008Covs10Seco5Fps100.npz'

# # Load the .mat file
# data = np.load(os.path.join(address, name), allow_pickle=True)
# htt = data['htt']  # Ocean height data over time
# para = data['params'].item()  # Parameters

# # Extract parameters
# wind_speed = para['windspeed']
# patch_size = para['patch_size']
# mesh_size = para['mesh_size']
# deta_x = para['deta_x']

# # Constants
# g = 9.81  # Gravitational constant
# length = wind_speed * wind_speed * 0.9128  # Domain wave length

# # Find the number of time steps (equivalent to MATLAB's find function)
# num_time_steps = 0
# for i in range(htt.shape[2]):
#     if htt[0, 0, i] > 0:
#         num_time_steps = i + 1
#     else:
#         break

# if num_time_steps == 0:  # If no positive values found
#     num_time_steps = htt.shape[2]  # Use all time steps

# # Create the grid for plotting
# cn = htt.shape[1]
# x = np.arange(-cn/2, cn/2) * deta_x
# y = np.arange(-cn/2, cn/2) * deta_x
# X, Y = np.meshgrid(x, y)

# # Find global min and max for consistent z-limits
# v_max = np.max(htt)
# v_min = np.min(htt)

# # Create figure and 3D axis
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')

# # Initial plot
# surf = ax.plot_surface(X, Y, htt[:, :, 0], cmap=cm.coolwarm, 
#                        linewidth=0, antialiased=False)

# # Set axis limits and labels
# ax.set_zlim(v_min, v_max)
# ax.set_xlabel('x (m)')
# ax.set_ylabel('y (m)')
# ax.set_zlabel('H (m)')
# plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
# ax.set_title(f"{name[:-4]} Num: 1")

# # Enable interactive mode for animation
# plt.ion()

# # Animation loop
# for tn in range(1, num_time_steps):
#     # Remove the previous surface plot
#     ax.collections[0].remove()
    
#     # Plot the new surface
#     surf = ax.plot_surface(X, Y, htt[:, :, tn], cmap=cm.coolwarm,
#                           linewidth=0, antialiased=False)
    
#     # Update the title
#     ax.set_title(f"{name[:-4]} Num: {tn+1}")
    
#     # Redraw the figure
#     fig.canvas.draw()
#     fig.canvas.flush_events()
    
#     # Add a small delay (adjust as needed for animation speed)
#     time.sleep(1/200)

# # Switch back to non-interactive mode
# plt.ioff()
# plt.show()