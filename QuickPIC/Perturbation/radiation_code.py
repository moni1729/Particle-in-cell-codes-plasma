import os
import pybetarad
import matplotlib.pyplot as plt
import numpy as np

radiation_parameters = {
    'algorithm': 'lienard_wiechert',
    'truncated_cone_on': True,
    'truncated_cone_angle': 1e-4,
    'compute_boundary_terms': False,
    'energy_min': 1e3,
    'energy_max': 1e6,
    'energy_num': 50,
    'energy_scale': 'logarithmic',
    'angle_1_min': -0.0005,
    'angle_1_max': 0.0005,
    'angle_1_num': 50,
    'angle_2_min': -0.0005,
    'angle_2_max': 0.0005,
    'angle_2_num': 50,
    'angular_grid_type': 'projected'
}

os.system('/home/myadav/software/betarad/interfaces/quickpic/quickpic2betarad --threads 4 --particles 10 --beam 0 simulations/0 trajectories')

times, trajectories, weight_multiplier, variable_weights = pybetarad.read_trajectories('trajectories')
for i in range(6):
    plt.plot(times, trajectories[0, :, i])
    plt.savefig(f'blah_{i}.png', dpi=300)
    plt.clf()

trajectories = pybetarad.Trajectories('trajectories')
trajectories.plot_beam_parameters()

with open('radiation_input', 'w') as f:
    for key, item in radiation_parameters.items():
        f.write(f'{key} = {item}\n')

os.system('betarad-initialize radiation_input progress')
os.system('mpirun -np 4 betarad-compute progress trajectories')
os.system('betarad-finalize progress output radiation')
radiation = pybetarad.RadiationGrid('radiation')
os.system('rm radiation_input trajectories radiation progress output')

radiation.plot_dist_projected()
radiation.plot_spectrum(vs_analytic={
    'n0': 1.79e16,
    'plasma_length': 0.3,
    'gamma': 10e9 / 510998.9461,
    'spot_size': 5e-6,
    'beam_charge': 1.5e-9
})
radiation.plot_double_differential_x()
