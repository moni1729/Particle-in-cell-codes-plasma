import json
import numpy as np
from run import scan_parameters, charges
import glob
import os
import h5py as h5
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib 
import scipy.interpolate

fsinitial = matplotlib.rcParams['font.size']
matplotlib.rcParams['font.size'] *= 1.5
plt.rcParams["figure.figsize"][0] *= 1.2
plt.rcParams["figure.figsize"][1] *= 1.2

with open('qpinput.json', 'r') as f:
    prototype = json.load(f)
    n0 = prototype['simulation']['n0']
    kpn1 = 299792458 / np.sqrt(n0 * 100 * 100 * 100 * 1.602176634e-19 * 1.602176634e-19 / (9.109383701528e-31 * 8.854187812813e-12))

os.system('mkdir -p results')

for j, beam in enumerate(['drive', 'witness']):
    results = []

    for i, scan_parameter in enumerate(scan_parameters):
        print(f'Simulation {i+1} of {len(scan_parameters)}')
        filenames = list(filter(lambda x: 'locktest' not in x, glob.glob(f'simulations/{i}/Beam{(j+1):04d}/Raw/*')))
        filenames.sort(key=lambda x: int(x[-11:-3]))
        filenames = filenames
        n_files = len(filenames)
        
        zs = []
        emittances = []
        emitx = []
        emity = []
        sigma_x = []
        sigma_y = []
        sigma_z = []
        gammas = []
        g_sigmas = []
        mux = []
        muy = []
        charge = []
        emit90 = []

        for file in filenames:
            with h5.File(file, 'r') as f:
                zs.append(f.attrs['TIME'][0] * kpn1)
                x = (np.array(f['x1'])) * kpn1
                y = (np.array(f['x2'])) * kpn1
                z = (np.array(f['x3'])) * kpn1
                px = np.array(f['p1'])
                py = np.array(f['p2'])
                pz = np.array(f['p3'])
                gamma = np.sqrt(1 + px * px + py * py + pz * pz)
                emittances.append(np.sqrt(np.sqrt(np.linalg.det(np.cov([x, y, px, py])))))
                emitx.append(np.sqrt(np.linalg.det(np.cov([x, px]))))
                emity.append(np.sqrt(np.linalg.det(np.cov([y, py]))))
                sigma_x.append(np.std(x))
                sigma_y.append(np.std(y))
                sigma_z.append(np.std(z))
                gammas.append(gamma.mean())
                g_sigmas.append(gamma.std())
                mux.append(np.mean(x))
                muy.append(np.mean(y))
                charge.append(len(x) * charges[beam] / (64 ** 3))
                r = np.sqrt(x ** 2 + y ** 2)
                r_sorted = np.sort(r)
                r90 = r_sorted[(len(r_sorted) * 9) // 10]
                x90 = x[r < r90]
                y90 = y[r < r90]
                px90 = px[r < r90]
                py90 = py[r < r90]
                emit90.append(np.sqrt(np.sqrt(np.linalg.det(np.cov([x90, y90, px90, py90])))))
        
        zs = np.array(zs)
        emittances = np.array(emittances)
        emitx = np.array(emitx)
        emity = np.array(emity)
        sigma_x = np.array(sigma_x)
        sigma_y = np.array(sigma_y)
        sigma_z = np.array(sigma_z)
        gammas = np.array(gammas)
        g_sigmas = np.array(g_sigmas)
        mux = np.array(mux)
        muy = np.array(muy)
        charge = np.array(charge)
        emit90 = np.array(emit90)

        results.append({'zs': zs, 'emittances': emittances, 'emitx': emitx, 'emity': emity, 'sigma_x': sigma_x, 'sigma_y': sigma_y, 'sigma_z': sigma_z, 'gammas': gammas, 'g_sigmas': g_sigmas, 'mux': mux, 'muy': muy, 'charge': charge, 'emit90': emit90})

    # Plot Energy
    for i, scan_parameter in enumerate(scan_parameters):
        result = results[i]
        plt.plot(result['zs'] * 100, result['gammas'] * 0.51099895000 / 1000, label='$\\sigma_{\\mathrm{witness}, \\perp}' + f' = {scan_parameter*1e6:.1f}$ $\mu$m')
    plt.legend()
    plt.xlabel('$z$ (cm)', fontsize=fsinitial*2)
    plt.ylabel('Energy (GeV)', fontsize=fsinitial*2)
    plt.savefig(f'results/{beam}_energy.png', dpi=300)
    plt.clf()
    
    # Plot Energy
    for i, scan_parameter in enumerate(scan_parameters):
        result = results[i]
        plt.plot(result['zs'] * 100, scipy.interpolate.CubicSpline(result['zs'], result['gammas']).derivative(nu=1)(result['zs']) * 0.51099895000 / 1000, label='$\\sigma_{\\mathrm{witness}, \\perp}' + f' = {scan_parameter*1e6:.1f}$ $\mu$m')
    plt.legend()
    plt.xlabel('$z$ (cm)', fontsize=fsinitial*2)
    plt.ylabel('Accelerating Gradient (GeV/m)', fontsize=fsinitial*2)
    plt.savefig(f'results/{beam}_accelgrad.png', dpi=300)
    plt.clf()

    # Plot Emit 4d
    for i, scan_parameter in enumerate(scan_parameters):
        result = results[i]
        plt.plot(result['zs'] * 100, result['emittances'] * 1e6, label='$\\sigma_{\\mathrm{witness}, \\perp}' + f' = {scan_parameter*1e6:.1f}$ $\\mu$m')
    plt.legend()
    plt.xlabel('$z$ (cm)', fontsize=fsinitial*2)
    plt.ylabel('$\\sqrt{\\epsilon_{n, \\mathrm{4D}}}$ ($\\mu$m)', fontsize=fsinitial*2)
    plt.savefig(f'results/{beam}_emit.png', dpi=300)
    plt.clf()

    # Plot Emit 4d
    for i, scan_parameter in enumerate(scan_parameters):
        result = results[i]
        plt.plot(result['zs'] * 100, result['emit90'] * 1e6, label='$\\sigma_{\\mathrm{witness}, \\perp}' + f' = {scan_parameter*1e6:.1f}$ $\\mu$m')
    plt.legend()
    plt.xlabel('$z$ (cm)', fontsize=fsinitial*2)
    plt.ylabel('$\\sqrt{\\epsilon_{n, \\mathrm{4D}, 90\\%}}$ ($\\mu$m)', fontsize=fsinitial*2)
    plt.savefig(f'results/{beam}_emit90.png', dpi=300)
    plt.clf()

    # Plot Emit 2d
    custom_lines = []
    for i, scan_parameter in enumerate(scan_parameters):
        result = results[i]
        plt.plot(result['zs'] * 100, result['emitx'] * 1e6, color=f'C{i}', linestyle='-')
        plt.plot(result['zs'] * 100, result['emity'] * 1e6, color=f'C{i}', linestyle='--')
        custom_lines.append(lines.Line2D([0], [0], color=f'C{i}', lw=2, linestyle='-'))
    custom_lines.append(lines.Line2D([0], [0], color='black', lw=2, linestyle='-'))
    custom_lines.append(lines.Line2D([0], [0], color='black', lw=2, linestyle='--'))
    plt.legend(custom_lines, ['$\\sigma_{\\mathrm{witness}, \\perp}' + f' = {scan_parameter*1e6:.1f}$ $\\mu$m' for scan_parameter in scan_parameters] + ['$\\epsilon_x$ ($\\mu$m)', '$\\epsilon_y$ ($\\mu$m)'])
    plt.xlabel('$z$ (cm)', fontsize=fsinitial*2)
    plt.savefig(f'results/{beam}_emitxy.png', dpi=300)
    plt.clf()


    fig, ax = plt.subplots()
    custom_lines = []
    for i, scan_parameter in enumerate(scan_parameters):
        result = results[i]
        ax.plot(result['zs'] * 100, result['sigma_x'] * 1e6, color=f'C{i}', linestyle='-')
        ax.plot(result['zs'] * 100, result['sigma_y'] * 1e6, color=f'C{i}', linestyle='--')
        custom_lines.append(lines.Line2D([0], [0], color=f'C{i}', lw=2, linestyle='-'))
    custom_lines.append(lines.Line2D([0], [0], color='black', lw=2, linestyle='-'))
    custom_lines.append(lines.Line2D([0], [0], color='black', lw=2, linestyle='--'))
    custom_labels = ['$\\sigma_i' + f' = {scan_parameter*1e6:.1f}$ $\\mu$m' for scan_parameter in scan_parameters] + ['$\\sigma_x$ ($\\mu$m)', '$\\sigma_y$ ($\\mu$m)']
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    ax.legend(custom_lines, custom_labels, loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_xlabel('$z$ (cm)', fontsize=fsinitial*2)
    fig.savefig(f'results/{beam}_sigmaxy.png', dpi=300)
    plt.close(fig)


    for i, scan_parameter in enumerate(scan_parameters):
        result = results[i]
        plt.plot(result['zs'] * 100, result['sigma_z'] * 1e6, label=f'$\\Delta = {scan_parameter*1e6:.1f}$ $\\mu$m')
    plt.legend()
    plt.xlabel('$z$ (cm)', fontsize=fsinitial*2)
    plt.ylabel('$\\sigma_z$ ($\\mu$m)', fontsize=fsinitial*2)
    plt.savefig(f'results/{beam}_sigmaz.png', dpi=300)
    plt.clf()

    custom_lines = []
    for i, scan_parameter in enumerate(scan_parameters):
        result = results[i]
        plt.plot(result['zs'] * 100, result['mux'] * 1e6, color=f'C{i}', linestyle='-')
        plt.plot(result['zs'] * 100, result['muy'] * 1e6, color=f'C{i}', linestyle='--')
        custom_lines.append(lines.Line2D([0], [0], color=f'C{i}', lw=2, linestyle='-'))
    custom_lines.append(lines.Line2D([0], [0], color='black', lw=2, linestyle='-'))
    custom_lines.append(lines.Line2D([0], [0], color='black', lw=2, linestyle='--'))
    plt.legend(custom_lines, ['$\\sigma_{\\mathrm{witness}, \\perp}' + f' = {scan_parameter*1e6:.1f}$ $\\mu$m' for scan_parameter in scan_parameters] + ['$\\mu_x$ ($\\mu$m)', '$\\mu_y$ ($\\mu$m)'])
    plt.xlabel('$z$ (cm)', fontsize=fsinitial*2)
    plt.savefig(f'results/{beam}_muxy.png', dpi=300)
    plt.clf()

    for i, scan_parameter in enumerate(scan_parameters):
        result = results[i]
        plt.plot(result['zs'] * 100, 100 * result['g_sigmas'] / result['gammas'], label='$\\sigma_{\\mathrm{witness}, \\perp}' + f' = {scan_parameter*1e6:.1f}$ $\\mu$m')
    plt.legend()
    plt.xlabel('$z$ (cm)', fontsize=fsinitial*2)
    plt.ylabel('Energy Spread (%)', fontsize=fsinitial*2)
    #plt.ylim(0, 100)
    plt.savefig(f'results/{beam}_espread.png', dpi=300)
    plt.clf()

    for i, scan_parameter in enumerate(scan_parameters):
        result = results[i]
        plt.plot(result['zs'] * 100, 1e9 * result['charge'], label='$\\sigma_{\\mathrm{witness}, \\perp}' + f' = {scan_parameter*1e6:.1f}$ $\\mu$m')
    plt.legend()
    plt.xlabel('$z$ (cm)', fontsize=fsinitial*2)
    plt.ylabel('$Q$ (nC)', fontsize=fsinitial*2)
    plt.savefig(f'results/{beam}_charge.png', dpi=300)
    plt.clf()