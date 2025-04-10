import glob
import h5py as h5
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.interpolate
import sys
import synchrad
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.lines as lines
import os
import scipy.integrate

import matplotlib
fsinitial = matplotlib.rcParams['font.size']
matplotlib.rcParams['font.size'] *= 1.5
plt.rcParams["figure.figsize"][0] *= 1.2
plt.rcParams["figure.figsize"][1] *= 1.2


scan_parameters = [None]

# compute skin depth
with open('qpinput.json', 'r') as f:
    prototype = json.load(f)
    kpn1 = 299792458 / np.sqrt(prototype['simulation']['n0'] * 100 * 100 * 100 * 1.602176634e-19 * 1.602176634e-19 / (9.109383701528e-31 * 8.854187812813e-12))

def get_trajectories(name, beam, n_particles='all', seed=None):

    # obtain filenames
    filenames = list(glob.glob(f'simulations/{name}/Beam{beam:04d}/Raw/*'))
    filenames.sort(key=lambda x: int(x[-11:-3]))
    filenames = filenames
    n_files = len(filenames)

    # figure out number of particles
    with h5.File(filenames[0], 'r') as file:
        ids = np.array(file['id'])
        if seed is None:
            np.random.default_rng().shuffle(ids)
        else:
            np.random.default_rng(seed).shuffle(ids)
        if n_particles == 'all':
            n_particles = len(ids)
        else:
            assert n_particles <= len(ids)
    assert len(ids) == len(set(ids))

    # misc
    z = np.empty((len(filenames)), dtype=np.float64)
    particles = {}
    for i in range(n_particles):
        arr = np.empty((n_files, 6))
        arr[:] = np.nan
        particles[ids[i]] = arr

    # iterate through files
    for i, filename in enumerate(filenames):
        with h5.File(filename, 'r') as file:
            print(f'{name}: reading file {i+1} of {len(filenames)}')

            # set z
            z[i] = kpn1 * file.attrs['TIME'][0]

            # read particle data
            x = kpn1 * np.array(file['x1'])
            y = kpn1 * np.array(file['x2'])
            zeta = kpn1 * np.array(file['x3'])
            px = np.array(file['p1'])
            py = np.array(file['p2'])
            pz = np.array(file['p3'])
            ids2 = np.array(file['id'])

            # append particle data
            for j, id in enumerate(ids2):
                if id in particles:
                    view = particles[id][i,:]
                    view[0] = x[j]
                    view[1] = y[j]
                    view[2] = zeta[j]
                    view[3] = px[j]
                    view[4] = py[j]
                    view[5] = pz[j]

    blacklist = []
    for i, (k, v) in enumerate(particles.items()):
        if np.any(np.isnan(v)):
            blacklist.append(k)

    print(len(particles), 'particles')
    print(len(blacklist), 'blacklisted')

    # convert to array
    array = np.empty((len(particles), n_files, 6), dtype=np.float64)
    for i, (k, v) in enumerate(particles.items()):
        array[i, :, :] = v

    return z, array

def save_particles(name, beam, n_particles, seed=None):
    z, particles = get_trajectories(name, beam, n_particles, seed=seed)
    os.system(f'mkdir -p simulations/{name}/trajectories')
    np.save(f'simulations/{name}/trajectories/z', z)
    np.save(f'simulations/{name}/trajectories/beam_{beam}', particles)

def load_particles(name, beam, n_particles):
    z = np.load(f'simulations/{name}/trajectories/z.npy')
    particles = np.load(f'simulations/{name}/trajectories/beam_{beam}.npy')
    assert particles.shape[0] >= n_particles
    return z, particles[:n_particles, :, :]

def convert_trajectories(z, trajectories, points):
    t = z / 299792458
    t2 = np.linspace(t.min(), t.max(), points)
    trajectories2 = np.empty((trajectories.shape[0], points, 9), dtype=np.float64)
    trajectories2[:] = np.nan
    for i in range(trajectories.shape[0]):
        gamma = np.sqrt(1 + trajectories[i, :, 3] ** 2 + trajectories[i, :, 4] ** 2 + trajectories[i, :, 5] ** 2)
        beta_x = trajectories[i, :, 3] / gamma
        beta_y = trajectories[i, :, 4] / gamma
        for j in range(trajectories.shape[1] - 1):
            if np.any(np.logical_not(np.isnan(trajectories[i, j, :]))):
                start_index = j
                break
        else:
            continue
        for j in range(trajectories.shape[1] - 1):
            k = trajectories.shape[1] - j - 1
            if np.any(np.logical_not(np.isnan(trajectories[i, k, :]))):
                end_index = k + 1
                break
        else:
            continue
        assert not np.any(np.isnan(trajectories[i, start_index:end_index, :]))
        assert np.all(np.isnan(trajectories[i, :start_index, :]))
        assert np.all(np.isnan(trajectories[i, end_index:, :]))
        g_old = gamma
        t_new = t[start_index:end_index]
        gamma = gamma[start_index:end_index]
        beta_x = beta_x[start_index:end_index]
        beta_y = beta_y[start_index:end_index]
        mask = (t2 > t[start_index]) & (t2 < t[end_index - 1])
        trajectories2[i, mask, 0] = scipy.interpolate.make_interp_spline(t_new, trajectories[i, start_index:end_index, 0], k=3)(t2[mask])
        trajectories2[i, mask, 1] = scipy.interpolate.make_interp_spline(t_new, trajectories[i, start_index:end_index, 1], k=3)(t2[mask])
        trajectories2[i, mask, 2] = scipy.interpolate.make_interp_spline(t_new, trajectories[i, start_index:end_index, 2], k=3)(t2[mask])
        trajectories2[i, mask, 3] = scipy.interpolate.make_interp_spline(t_new, beta_x, k=3)(t2[mask])
        trajectories2[i, mask, 4] = scipy.interpolate.make_interp_spline(t_new, beta_y, k=3)(t2[mask])
        trajectories2[i, mask, 5] = scipy.interpolate.make_interp_spline(t_new, gamma, k=3)(t2[mask])
        trajectories2[i, mask, 6] = scipy.interpolate.make_interp_spline(t_new, beta_x, k=3).derivative(nu=1)(t2[mask])
        trajectories2[i, mask, 7] = scipy.interpolate.make_interp_spline(t_new, beta_y, k=3).derivative(nu=1)(t2[mask])
        trajectories2[i, mask, 8] = scipy.interpolate.make_interp_spline(t_new, gamma, k=3).derivative(nu=1)(t2[mask])
    return t2, trajectories2

def main():
    # Radition Computation Config
    scan_parameter_fancy_names = [None]#['$\\epsilon_{n, \\mathrm{witness}}' + f' = {int(round(scan_parameter * 1e6))}$ $\\mu$m' for scan_parameter in scan_parameters]
    read_files = False
    compute_radiation = False
    seed = None
    n_particles_to_save = 1000
    n_particles_to_compute_with = 200
    plasma_length = 0.2991
    plasma_actual_length = 0.3
    points_for_radiation_computation = 5000 * plasma_length #10000 * plasma_length
    #energies, energies_midpoint = synchrad.logspace_midpoint(3, 6, 10)
    #energies_is_log = True
    energies, energies_midpoint, energies_step = synchrad.linspace_midpoint(0, 1e4, 200)#2 * 10 ** 4, 30)
    energies_is_log = False
    energies = energies[1:]
    energies_midpoint = energies_midpoint[1:]
    phi_xs, phi_xs_midpoint, phi_xs_step = synchrad.linspace_midpoint(-0.5e-3, 0.5e-3, 200)
    phi_ys, phi_ys_midpoint, phi_ys_step = synchrad.linspace_midpoint(-0.5e-3, 0.5e-3, 200)
    threads = 64
    beam_charges = [{'drive': 1.5e-9, 'witness': 0.5e-9} for _ in scan_parameters]
    double_differential_vmin_vmax = ['auto', 'auto']
    dist_vmin_vmax = ['auto', 'auto']
    cmap = 'magma'
    overal_result_suffix = ''
    results_folder = 'results_single'


    os.system(f'mkdir -p {results_folder}')

    if read_files:
        for i in range(len(scan_parameters)):
            save_particles(i, 1, n_particles_to_save, seed=seed)
            save_particles(i, 2, n_particles_to_save, seed=seed)
            
    data = [[scan_parameter, {'drive': {'fancy_name': 'Drive Beam', 'number': 1}, 'witness': {'fancy_name': 'Witness Beam', 'number': 2}}] for scan_parameter in scan_parameters]
    for i, (scan_parameter, data_dict) in enumerate(data):
        i = 2
        print(f'{i + 1} out of {len(scan_parameters)}: computing radiation for scan parameter {scan_parameter}')
        os.system(f'mkdir -p {results_folder}/{i}')
        if compute_radiation:
            os.system(f'mkdir -p simulations/{i}/radiation')
            for beam_name, beam_dict in data_dict.items():
                z, particles = load_particles(i, beam_dict['number'], n_particles_to_compute_with)
                if z[-1] < plasma_length:
                    raise Exception(f'Plasma length error: cannot truncate simulation length of {z[-1]} to length {plasma_length}.')
                index = np.argmin(np.abs(z - plasma_length))
                z = z[:index]
                particles = particles[:, :index, :]
                t, trajectories = convert_trajectories(z, particles, int(round(points_for_radiation_computation)))
                for j in range(10):
                    plt.plot(z, particles[j, :, 0])
                plt.xlabel('$z$ (m)', fontsize=1.5*fsinitial)
                plt.ylabel('$x$ (m)', fontsize=1.5*fsinitial)
                plt.savefig(f'{results_folder}/{i}/{beam_name}_particles{overal_result_suffix}.png', dpi=400)
                plt.clf()
                for j in range(10):
                    plt.plot(299792458 * t, trajectories[j, :, 0])
                plt.xlabel('$ct$ (m)', fontsize=1.5*fsinitial)
                plt.ylabel('$x$ (m)', fontsize=1.5*fsinitial)
                plt.savefig(f'{results_folder}/{i}/{beam_name}_trajectories{overal_result_suffix}.png', dpi=400)
                plt.clf()
                print('radcomp', flush=True)
                radiation = synchrad.compute_radiation_grid(trajectories, energies, phi_xs, phi_ys, np.diff(t).mean(), threads=threads, NaNs=True)
                print('radcompover', flush=True)
                radiation *= (plasma_actual_length * beam_charges[i][beam_name]) / (trajectories.shape[0] * plasma_length * 1.60217662e-19)
                beam_dict['radiation'] = radiation
            data_dict['all'] = {'radiation': data_dict['drive']['radiation'] + data_dict['witness']['radiation'], 'fancy_name': 'Both Beams'} 
            for beam_name, beam_dict in data_dict.items():
                beam_dict['dd'] = np.sum(beam_dict['radiation'] ** 2, axis=3)
                del beam_dict['radiation']
                np.save(f'simulations/{i}/radiation/{beam_name}', beam_dict['dd'])
        else:
            data_dict['all'] = {'fancy_name': 'Both Beams'} 
            for beam_name, beam_dict in data_dict.items():
                beam_dict['dd'] = np.load(f'simulations/{i}/radiation/{beam_name}.npy')

    for i, (scan_parameter, data_dict) in enumerate(data):
        i = 2
        for beam_name, beam_dict in data_dict.items():
            dd = beam_dict['dd']
            
            # double differential
            dd_x = np.sum(dd, axis=2) * phi_ys_step
            dd_y = np.sum(dd, axis=1) * phi_xs_step
            for j, (coordinate, dd_coord) in enumerate((('x', dd_x), ('y', dd_y))):
                fig, ax = plt.subplots()
                #ax.set_title(beam_dict['fancy_name'] + ' ' + scan_parameter_fancy_names[i])
                vmin, vmax = dd_coord.min(), dd_coord.max()
                if double_differential_vmin_vmax[0] != 'auto':
                    vmin = double_differential_vmin_vmax[0]
                if double_differential_vmin_vmax[1] != 'auto':
                    vmax = double_differential_vmin_vmax[1]
                hm = ax.pcolormesh(energies_midpoint / 1000, phi_xs_midpoint * 1e6, dd_coord.T, vmin=vmin, vmax=vmax, cmap=cmap)
                ax.set_xlim(energies.min() / 1000, energies.max() / 1000)
                ax.set_ylim(phi_xs.min() * 1e6, phi_xs.max() * 1e6)
                if energies_is_log:
                    ax.set_xscale('log')
                ax.set_xlabel('$\\epsilon$ (keV)', fontsize=1.5*fsinitial)
                ax.set_ylabel(f'$\\phi_{coordinate}$ ($\\mu$rad)', fontsize=1.5*fsinitial)
                cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap), ax=ax)
                cbar.set_label('$\\frac{dI}{d\\phi_{' + coordinate + '}d\\epsilon}$', fontsize=1.5*fsinitial)
                fig.savefig(f'{results_folder}/{i}/double_differential_{beam_name}_{coordinate}{overal_result_suffix}.png', dpi=400)
                plt.close(fig)

            # distribution
            dist = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
            vmin, vmax = dist.min(), dist.max()
            if dist_vmin_vmax[0] != 'auto':
                vmin = dist_vmin_vmax[0]
            if dist_vmin_vmax[1] != 'auto':
                vmax = dist_vmin_vmax[1]
            fig, ax = plt.subplots()
            #ax.set_title(beam_dict['fancy_name'] + ' ' + scan_parameter_fancy_names[i])
            ax.pcolormesh(phi_xs_midpoint * 1e6, phi_ys_midpoint * 1e6, dist.T, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_xlim(phi_xs.min() * 1e6, phi_xs.max() * 1e6)
            ax.set_ylim(phi_ys.min() * 1e6, phi_ys.max() * 1e6)
            ax.set_xlabel('$\\phi_x$ ($\\mu$rad)', fontsize=1.5*fsinitial)
            ax.set_ylabel('$\\phi_y$ ($\\mu$rad)', fontsize=1.5*fsinitial)
            cbar = fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap), ax=ax)
            cbar.set_label('$\\frac{dI}{d\\Omega}$ (eV)', fontsize=1.5*fsinitial)
            fig.savefig(f'{results_folder}/{i}/2d_angular_distribution_{beam_name}{overal_result_suffix}.png', dpi=400)
            plt.close(fig)
            
        fig, ax = plt.subplots()
        custom_lines = []
        beamfancynames = []
        for ii, beam_name in enumerate(['drive', 'witness', 'all']):
            for i, (scan_parameter, data_dict) in enumerate(data):
                dd = data_dict[beam_name]['dd']
                dist = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
                dist_y = np.sum(dist, axis=0) * phi_xs_step
                dist_x = np.sum(dist, axis=1) * phi_ys_step
                ax.plot(phi_xs * 1e6, dist_x, color=f'C{ii}', linestyle='-')
                ax.plot(phi_ys * 1e6, dist_y, color=f'C{ii}', linestyle='--')
                custom_lines.append(lines.Line2D([0], [0], color=f'C{ii}', lw=2, linestyle='-'))
                beamfancynames.append(data_dict[beam_name]['fancy_name'])
        custom_lines.append(lines.Line2D([0], [0], color='black', lw=2, linestyle='-'))
        custom_lines.append(lines.Line2D([0], [0], color='black', lw=2, linestyle='--'))
        ax.set_yscale('log')
        ax.legend(custom_lines, beamfancynames + ['$\\phi_x$', '$\\phi_y$'])
        #ax.legend()
        ax.set_xlabel('$\\phi$ ($\\mu$rad)', fontsize=1.5*fsinitial)
        ax.set_ylabel('$\\frac{dI}{d\\phi}$', fontsize=1.5*fsinitial)
        fig.savefig(f'{results_folder}/1d_dist{overal_result_suffix}.png', dpi=400)
        plt.close(fig)
        
    fig, ax = plt.subplots()
    for beam_name in ['drive', 'witness', 'all']:
        for i, (scan_parameter, data_dict) in enumerate(data):
            i = 2
            dd = data_dict[beam_name]['dd']
            spectrum = np.sum(dd, axis=(1,2)) * phi_xs_step * phi_ys_step
            ax.plot(energies / 1000, spectrum, label=data_dict[beam_name]['fancy_name'])#, label=scan_parameter_fancy_names[i])#, color='royalblue')
    if energies_is_log:
        ax.set_xscale('log')
    xscalename = 'log' if energies_is_log else 'lin'
    ax.legend()
    ax.set_xlabel('$\\epsilon$ (keV)', fontsize=1.5*fsinitial)
    ax.set_ylabel('$\\frac{dI}{d\\epsilon}$', fontsize=1.5*fsinitial)
    if energies_is_log:
        ax.set_yscale('log')
        yscalename = 'log'
    else:
        yscalename = 'lin'
    fig.savefig(f'{results_folder}/spectrum_{xscalename}_{yscalename}{overal_result_suffix}.png', dpi=400)
    plt.close(fig)

    with open(f'{results_folder}/total_energy{overal_result_suffix}.txt', 'w+') as f:
        f.write('simulation\tscan parameter\tbeam\ttotal energy (eV)\n')
        for i, (scan_parameter, data_dict) in enumerate(data):
            for beam_name, beam_dict in data_dict.items():
                dd = beam_dict['dd']
                tot = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis]) * phi_xs_step * phi_ys_step
                if scan_parameter is None:
                    f.write(f'{i}\t-\t{beam_name}\t{tot:.15e}\n')
                else:
                    f.write(f'{i}\t{scan_parameter:.15e}\t{beam_name}\t{tot:.15e}\n')

            
r'''

    # plot angular dist
    fig, ax = plt.subplots()
    custom_lines = []
    for i, dd in enumerate(data):
        dist = np.sum(0.5 * (dd[1:, :, :] + dd[:-1, :, :]) * (energies[1:] - energies[:-1])[:, np.newaxis, np.newaxis], axis=0)
        angles = np.linspace(0, max(max(phi_xs), max(phi_ys)), 200)
        totals = []
        for j, angle in enumerate(angles):
            mask = (phi_xs <= angle)[:, np.newaxis] & (phi_ys <= angle)[np.newaxis, :]
            totals.append(np.sum(dist[mask]) * phi_xs_step * phi_ys_step)
        ax.plot(angles * 1e6, 100 * np.array(totals) / totals[-1], label=names[i])
    ax.legend()
    ax.set_xlabel('Spectrometer Acceptance Angle ($\\mu$rad)')
    ax.set_y
    ('Percentage of Energy Deposited')
    ax.set_ylim(0, 100)
    fig.savefig(f'{results_folder}/1d_dist_alt{overal_result_suffix}.png', dpi=400)
    plt.close(fig)

    
'''
if __name__ == '__main__':
    main()
