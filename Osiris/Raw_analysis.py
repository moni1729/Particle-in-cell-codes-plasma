import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import os
import glob
import multiprocessing
import functools
import matplotlib
import matplotlib.colors as colors

processes = 2

matplotlib.rcParams.update({'font.size': 5})

he_electron_files = list(glob.glob('MS/RAW/driver/RAW-driver-*.h5'))
he_electron_files.sort(key=lambda path: int(path[-9:-3]))
he_electron_files = he_electron_files[::1]
he_electron_files = he_electron_files[:-1]

# Create the frames_H-ions directory
os.system('mkdir -p frames_H-ions')

# Count the number of files
n_files = len(he_electron_files)

def get_vmax(i):
    print(f'pass 1: {i + 1} of {n_files}')
    N=10
    with h5.File(he_electron_files[i], 'r') as f:
        z = np.array(f['x1'])[::N] - f.attrs['TIME'][::N]
        x = np.array(f['x2'])[::N]
        y = np.array(f['x3'])[::N]
        pz = np.array(f['p1'])[::N]
        px = np.array(f['p2'])[::N]
        py = np.array(f['p3'])[::N]
    if len(x.shape) == 0 or x.shape == (1,):
        return None, None, None, None, None, None, None, None, None, None, None, None
    return x.min(), x.max(), y.min(), y.max(), z.min(), z.max(), px.min(), px.max(), py.min(), py.max(), pz.min(), pz.max()

def plot_frame(x_min, x_max, y_min, y_max, z_min, z_max, px_min, px_max, py_min, py_max, pz_min, pz_max, i):
    print(f'pass 2: {i + 1} of {n_files}')
    N=10
    with h5.File(he_electron_files[i], 'r') as f:
        z = np.array(f['x1'])[::N] - f.attrs['TIME'][::N]
        x = np.array(f['x2'])[::N]
        y = np.array(f['x3'])[::N]
        pz = np.array(f['p1'])[::N]
        px = np.array(f['p2'])[::N]
        py = np.array(f['p3'])[::N]

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, dpi=400)

    ax1.scatter(x, px, color='black', marker='s', linewidth=0, s=(2*72./fig.dpi)**2)
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(px_min, px_max)
    ax1.set_ylabel('$p_x / m_e c$', fontsize=8)
    ax1.tick_params(labelsize=8)

    ax2.scatter(y, px, color='black', marker='s', linewidth=0, s=(2*72./fig.dpi)**2)
    ax2.set_xlim(y_min, y_max)
    ax2.set_ylim(px_min, px_max)
    ax2.tick_params(labelsize=8)

    ax3.scatter(z, px, color='black', marker='s', linewidth=0, s=(2*72./fig.dpi)**2)
    ax3.set_xlim(z_min, z_max)
    ax3.set_ylim(px_min, px_max)
    ax3.tick_params(labelsize=8)

    ax4.scatter(x, py, color='black', marker='s', linewidth=0, s=(2*72./fig.dpi)**2)
    ax4.set_xlim(x_min, x_max)
    ax4.set_ylim(py_min, py_max)
    ax4.set_ylabel('$p_y / m_e c$', fontsize=8)
    ax4.tick_params(labelsize=8)

    ax5.scatter(y, py, color='black', marker='s', linewidth=0, s=(2*72./fig.dpi)**2)
    ax5.set_xlim(y_min, y_max)
    ax5.set_ylim(py_min, py_max)
    ax5.tick_params(labelsize=8)

    ax6.scatter(z, py, color='black', marker='s', linewidth=0, s=(2*72./fig.dpi)**2)
    ax6.set_xlim(z_min, z_max)
    ax6.set_ylim(py_min, py_max)
    ax6.tick_params(labelsize=8)

    ax7.scatter(x, pz, color='black', marker='s', linewidth=0, s=(2*72./fig.dpi)**2)
    ax7.set_xlim(x_min, x_max)
    ax7.set_ylim(pz_min, pz_max)
    ax7.set_xlabel('$k_p x$', fontsize=8)
    ax7.set_ylabel('$p_z / m_e c$', fontsize=8)
    ax7.tick_params(labelsize=8)

    ax8.scatter(y, pz, color='black', marker='s', linewidth=0, s=(2*72./fig.dpi)**2)
    ax8.set_xlim(y_min, y_max)
    ax8.set_ylim(pz_min, pz_max)
    ax8.set_xlabel('$k_p y$', fontsize=8)
    ax8.tick_params(labelsize=8)

    ax9.scatter(z, pz, color='black', marker='s', linewidth=0, s=(2*72./fig.dpi)**2)
    ax9.set_xlim(z_min, z_max)
    ax9.set_ylim(pz_min, pz_max)
    ax9.set_xlabel('$k_p \\zeta$', fontsize=8)
    ax9.tick_params(labelsize=8)


    fig.subplots_adjust(wspace=0.5, hspace=0.5)

    fig.savefig(f'frames_H-ions/phase_space_matrix_H-ions_{i}.png')
    plt.close(fig)

with multiprocessing.Pool(processes) as pool:
    x_min = None
    x_max = None
    y_min = None
    y_max = None
    z_min = None
    z_max = None
    px_min = None
    px_max = None
    py_min = None
    py_max = None
    pz_min = None
    pz_max = None
    for a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12 in pool.imap_unordered(get_vmax, range(n_files), processes):
        x_min = x_min if a1 is None else (min([x_min, a1]) if x_min is not None else a1)
        x_max = x_max if a2 is None else (max([x_max, a2]) if x_max is not None else a2)
        y_min = y_min if a3 is None else (min([y_min, a3]) if y_min is not None else a3)
        y_max = y_max if a4 is None else (max([y_max, a4]) if y_max is not None else a4)
        z_min = z_min if a5 is None else (min([z_min, a5]) if z_min is not None else a5)
        z_max = z_max if a6 is None else (max([z_max, a6]) if z_max is not None else a6)
        px_min = px_min if a7 is None else (min([px_min, a7]) if px_min is not None else a7)
        px_max = px_max if a8 is None else (max([px_max, a8]) if px_max is not None else a8)
        py_min = py_min if a9 is None else (min([py_min, a9]) if py_min is not None else a9)
        py_max = py_max if a10 is None else (max([py_max, a10]) if py_max is not None else a10)
        pz_min = pz_min if a11 is None else (min([pz_min, a11]) if pz_min is not None else a11)
        pz_max = pz_max if a12 is None else (max([pz_max, a12]) if pz_max is not None else a12)
    partial = functools.partial(plot_frame, x_min, x_max, y_min, y_max, z_min, z_max, px_min, px_max, py_min, py_max, pz_min, pz_max)
    for result in pool.imap_unordered(partial, range(n_files), processes):
        pass

os.system('rm movie_ep.mp4')
os.system('ffmpeg -i frames_H-ions/phase_space_matrix_H-ions_%d.png -c:v libx264 -crf 25  -pix_fmt yuv420p movie_ep.mp4')

#os.system('ffmpeg -i frames_H-ions/phase_space_matrix_H-ions_%d.png -c:v libvpx-vp9 -pix_fmt yuv420p movie_ep.webm')

#os.system('rm phase_space_matrix_H-ions.mp4')
#os.system('ffmpeg -i frames_e1/%d.png -c:v libvpx-vp9 -pix_fmt yuv420p movie_e1.webm')
#os.system('movie frames_H-ions/phase_space_matrix_H-ions_%d.png -c:v libvpx-vp9 -pix_fmt yuv420p phase_space_matrix_H-ions.webm')
