import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
import glob
import multiprocessing
import functools
import matplotlib
import matplotlib.colors as colors
import re
#matplotlib.rcParams.update({'font.size': 5})

wp = np.sqrt(4e13 * 100 * 100 * 100 * 1.602176634e-19 * 1.602176634e-19 / (9.109383701528e-31 * 8.854187812813e-12)) #hard coded n0 here to be 4e13
field_unit = 9.109383701528e-31 * 299792458 * wp / 1.602176634e-19
field_unit /= 1e6

matplotlib.rcParams.update({'font.size': 5})

processes = 32

factor = 100

# Obtain lists of the files

e1_files = list(glob.glob('MS/FLD/e1/e1-*.h5'))
e1_files.sort(key=lambda path: int(path[-9:-3]))

# Create the frames_e1 directory
os.system('mkdir -p frames_e1')

# Count the number of files
n_files = len(e1_files)

# Ensure the number of files is the same for each species

def get_vmax(i):
    print(f'pass 1: {i + 1} of {n_files}')
    with h5.File(e1_files[i], 'r') as f:
        n_z, n_x, n_y = f['SIMULATION'].attrs['NX']
        assert n_x == n_y
        e1_slice = np.array(f['e1'][n_x // 2, :, :]) * field_unit
        e1_vmin = np.min(e1_slice)
        e1_vmax = np.max(e1_slice)
    return e1_vmin, e1_vmax

def plot_frame(e1_vmin, e1_vmax, i):
    print(f'pass 2: {i + 1} of {n_files}')
    with h5.File(e1_files[i], 'r') as f:
        n_z, n_x, n_y = f['SIMULATION'].attrs['NX']
        assert n_x == n_y
        z_min, x_min, y_min = f['SIMULATION'].attrs['XMIN']
        z_max, x_max, y_max = f['SIMULATION'].attrs['XMAX']
        z_min -= f.attrs['TIME'][0]
        z_max -= f.attrs['TIME'][0]
        x, dx = np.linspace(x_min, x_max, n_x, retstep=True)
        y, dy = np.linspace(y_min, y_max, n_y, retstep=True)
        z, dz = np.linspace(z_min, z_max, n_z, retstep=True)
        x_mid = np.linspace(x_min - 0.5 * dx, x_max + 0.5 * dx, n_x + 1)
        y_mid = np.linspace(y_min - 0.5 * dy, y_max + 0.5 * dy, n_y + 1)
        z_mid = np.linspace(z_min - 0.5 * dz, z_max + 0.5 * dz, n_z + 1)
        e1_slice = np.array(f['e1'][n_x // 2, :, :]) * field_unit

    fig, (ax1) = plt.subplots(nrows=1, ncols=1)

    assert e1_vmin < 0
    assert e1_vmax > 0
    e1_vmin2 = -max([-e1_vmin, e1_vmax])
    e1_vmax2 = -e1_vmin2

    norm = colors.SymLogNorm(linthresh=e1_vmax2 / factor, vmin=e1_vmin2, vmax=e1_vmax2)

    pcm4 = ax1.pcolormesh(z_mid, y_mid, e1_slice, norm=norm, cmap='coolwarm')
    ax1.set_title('Longitudinal Electric Field', fontsize=12)
    ax1.set_xlim(z.min(), z.max())
    ax1.set_ylim(y_mid.min(), y_mid.max())
    ax1.set_xlabel('$k_p \\zeta (\mu m)$', fontsize=12)
    ax1.set_ylabel('$k_p r (\mu m)$', fontsize=12)
    #fig.colorbar(pcm4, ax=ax1).set_label('$E_z$ (MV/m)')
    cbar=fig.colorbar(pcm4, ax=ax1)
    cbar.set_label('$E_z$ (MV/m)', fontsize=12)
    cbar.ax.tick_params(labelsize=12)
    ax1.tick_params(labelsize=12)
    fig.savefig(f'frames_e1/{i}.png', dpi=400)
    plt.close(fig)



    '''

    fig, (ax2) = plt.subplots(nrows=1, ncols=1)
    h_electron_density_slice[h_electron_density_slice <= h_electron_vmax / factor] = h_electron_vmax / factor
    pcm2 = ax2.pcolormesh(z_mid, y_mid, h_electron_density_slice, norm=colors.LogNorm(h_electron_vmax / factor, h_electron_vmax), cmap='viridis')
    ax2.set_title('Hydrogen Electrons', fontsize=12)
    ax2.set_xlim(z.min(), z.max())
    ax2.set_ylim(y_mid.min(), y_mid.max())
    ax2.set_xlabel('$k_p \\zeta$', fontsize=12)
    ax2.set_ylabel('$k_p r$', fontsize=12)
    cbar=fig.colorbar(pcm2, ax=ax2)
    #fig.colorbar(pcm2, ax=ax2).set_label('$n_e / n_0$', fontsize=14)


    # Set the color bar label with a larger font size
    cbar.set_label('$n_e / n_0$', fontsize=12)

    # Set the tick label size on the color bar
    cbar.ax.tick_params(labelsize=12)  # Set this to your desired value

    ax2.tick_params(labelsize=12)  # Adjusting the size of colorbar ticks
    '''



with multiprocessing.Pool(processes) as pool:
    e1_vmin = 0
    e1_vmax = 0
    for a, b in pool.imap_unordered(get_vmax, range(n_files), processes):
        e1_vmin = min([a, e1_vmin])
        e1_vmax = max([b, e1_vmax])

    partial = functools.partial(plot_frame, e1_vmin, e1_vmax)
    for result in pool.imap_unordered(partial, range(n_files), processes):
        pass
os.system('rm movie_e1.mp4')
os.system('ffmpeg -i frames_e1/%d.png -c:v libx264 -crf 25  -pix_fmt yuv420p movie_e1.mp4')
