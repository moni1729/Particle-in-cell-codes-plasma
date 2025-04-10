import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
import glob
import multiprocessing
import functools
import matplotlib
import matplotlib.colors as colors

factor = 10000

matplotlib.rcParams.update({'font.size': 5})

processes = 32


# Obtain lists of the files
driver_files = list(glob.glob('MS/DENSITY/driver/charge/charge-driver-*.h5'))
driver_files.sort(key=lambda path: int(path[-9:-3]))
h_electron_files = list(glob.glob('MS/DENSITY/H-electrons/charge/charge-H-electrons-*.h5'))
h_electron_files.sort(key=lambda path: int(path[-9:-3]))
he_electron_files = list(glob.glob('MS/DENSITY/He-ADK-electrons/charge/charge-He-ADK-electrons-*.h5'))
he_electron_files.sort(key=lambda path: int(path[-9:-3]))
#amod_files = list(glob.glob('MS/FLD/a_mod/a_mod-*.h5'))
#amod_files.sort(key=lambda path: int(path[-9:-3]))

# Create the frames directory
os.system('mkdir -p frames')

# Count the number of files
n_files = len(driver_files)

# Ensure the number of files is the same for each species
assert len(h_electron_files) == n_files
assert len(he_electron_files) == n_files
#assert len(amod_files) == n_files
assert h_electron_files[-1][-9:-3] == driver_files[-1][-9:-3]
assert he_electron_files[-1][-9:-3] == driver_files[-1][-9:-3]
#assert amod_files[-1][-9:-3] == driver_files[-1][-9:-3]

def get_vmax(i):
    print(f'pass 1: {i + 1} of {n_files}')
    with h5.File(driver_files[i], 'r') as f:
        n_z, n_x, n_y = f['SIMULATION'].attrs['NX']
        assert n_x == n_y
        drive_density_slice = -np.array(f['charge'][n_x // 2, :, :])
        drive_vmax = np.max(drive_density_slice)
    with h5.File(h_electron_files[i], 'r') as f:
        h_electron_density_slice = -np.array(f['charge'][n_x // 2, :, :])
        h_electron_vmax = np.max(h_electron_density_slice)
    with h5.File(he_electron_files[i], 'r') as f:
        he_electron_density_slice = -np.array(f['charge'][n_x // 2, :, :])
        he_electron_vmax = np.max(he_electron_density_slice)
    #with h5.File(amod_files[i], 'r') as f:
        #amod_slice = np.array(f['a_mod'][n_x // 2, :, :])
        #amod_vmin = np.min(amod_slice)
        #amod_vmax = np.max(amod_slice)
    return drive_vmax, h_electron_vmax, he_electron_vmax

def plot_frame(drive_vmax, h_electron_vmax, he_electron_vmax, i):
    print(f'pass 2: {i + 1} of {n_files}')
    with h5.File(driver_files[i], 'r') as f:
        n_z, n_x, n_y = f['SIMULATION'].attrs['NX']
        assert n_x == n_y
        z_min, x_min, y_min = f['SIMULATION'].attrs['XMIN']
        z_max, x_max, y_max = f['SIMULATION'].attrs['XMAX']
        z_min# -= f.attrs['TIME'][0]
        z_max# -= f.attrs['TIME'][0]
        x, dx = np.linspace(x_min, x_max, n_x, retstep=True)
        y, dy = np.linspace(y_min, y_max, n_y, retstep=True)
        z, dz = np.linspace(z_min, z_max, n_z, retstep=True)
        x_mid = np.linspace(x_min - 0.5 * dx, x_max + 0.5 * dx, n_x + 1)
        y_mid = np.linspace(y_min - 0.5 * dy, y_max + 0.5 * dy, n_y + 1)
        z_mid = np.linspace(z_min - 0.5 * dz, z_max + 0.5 * dz, n_z + 1)
        drive_density_slice = -np.array(f['charge'][n_x // 2, :, :])
    with h5.File(h_electron_files[i], 'r') as f:
        h_electron_density_slice = -np.array(f['charge'][n_x // 2, :, :])
    with h5.File(he_electron_files[i], 'r') as f:
        he_electron_density_slice = -np.array(f['charge'][n_x // 2, :, :])
#    with h5.File(amod_files[i], 'r') as f:
#        drive_density_slice = np.array(f['charge'][n_x // 2, :, :])

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    drive_density_slice[drive_density_slice <= drive_vmax / factor] = drive_vmax / factor
    pcm1 = ax1.pcolormesh(z_mid, y_mid, drive_density_slice, norm=colors.LogNorm(drive_vmax / factor, drive_vmax), cmap='viridis')
    ax1.set_title('Drive Beam')
    ax1.set_xlim(z.min(), z.max())
    ax1.set_ylim(y_mid.min(), y_mid.max())
    ax1.set_xlabel('$k_p \\zeta$', fontsize=8)
    ax1.set_ylabel('$k_p r$', fontsize=8)
    cbar=fig.colorbar(pcm1, ax=ax1)
    cbar.set_label('$n_e / n_0$', fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax1.tick_params(labelsize=8)

    h_electron_density_slice[h_electron_density_slice <= h_electron_vmax / factor] = h_electron_vmax / factor
    pcm2 = ax2.pcolormesh(z_mid, y_mid, h_electron_density_slice, norm=colors.LogNorm(h_electron_vmax / factor, h_electron_vmax), cmap='viridis')
    ax2.set_title('Hydrogen Electrons')
    ax2.set_xlim(z.min(), z.max())
    ax2.set_ylim(y_mid.min(), y_mid.max())
    ax2.set_xlabel('$k_p \\zeta$', fontsize=8)
    ax2.set_ylabel('$k_p r$', fontsize=8)
    cbar=fig.colorbar(pcm2, ax=ax2)
    cbar.set_label('$n_e / n_0$', fontsize=8)
    cbar.ax.tick_params(labelsize=8)
    ax2.tick_params(labelsize=8)




    he_electron_vmax += 1
    he_electron_density_slice[he_electron_density_slice <= he_electron_vmax / factor] = he_electron_vmax / factor
    pcm3 = ax3.pcolormesh(z_mid, y_mid, he_electron_density_slice, norm=colors.LogNorm(he_electron_vmax / factor, he_electron_vmax), cmap='viridis')
    ax3.set_title('Helium Electrons')
    ax3.set_xlim(z.min(), z.max())
    ax3.set_ylim(y_mid.min(), y_mid.max())
    ax3.set_xlabel('$k_p \\zeta$', fontsize=8)
    ax3.set_ylabel('$k_p r$', fontsize=8)
    fig.colorbar(pcm3, ax=ax3).set_label('$n_e / n_0$', fontsize=8)
    ax3.tick_params(labelsize=8)
    fig.subplots_adjust(wspace=0.5, hspace=0.5)

#    amod_slice[amod_slice <= amod_vmax / factor] = amod_vmax / factor
 #   pcm4 = ax4.pcolormesh(z_mid, y_mid, amod_slice, norm=colors.LogNorm(amod_vmax / factor, amod_vmax), cmap='viridis')
#    ax4.set_title('Laser')
 #   ax4.set_xlim(z.min(), z.max())
  #  ax4.set_ylim(y_mid.min(), y_mid.max())
 #   ax4.set_xlabel('$k_p \\zeta$')
  #  fig.colorbar(pcm4, ax=ax4).set_label('$a_{\\mathrm{mod}}$')

    fig.savefig(f'frames/{i}.png', dpi=400)
    plt.close(fig)

with multiprocessing.Pool(processes) as pool:
    drive_vmax = 0
    h_electron_vmax = 0
    he_electron_vmax = 0
#    amod_vmin = 0
 #   amod_vmax = 0
    for a, b, c in pool.imap_unordered(get_vmax, range(n_files), processes):
        drive_vmax = max([a, drive_vmax])
        h_electron_vmax = max([b, h_electron_vmax])
        he_electron_vmax = max([c, he_electron_vmax])
 #       amod_vmin = min([d, amod_vmin])
  #      amod_vmax = max([e, amod_vmax])
    partial = functools.partial(plot_frame, drive_vmax, h_electron_vmax, he_electron_vmax)
    for result in pool.imap_unordered(partial, range(n_files), processes):
        pass

os.system('rm movie.mp4')
#os.system('movie frames/%d.png movie.mp4')
os.system('ffmpeg -i frames/%d.png -c:v libx264 -crf 25  -pix_fmt yuv420p movie.mp4')
