#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# test that simulates the betatron motion of electrons in a focusing magnetic field:

import numpy as np
from pywarpx import picmi
#from picmi import ExternalField

# Number of time steps
max_steps = 50

print('status1')
# Physical constants
c = picmi.constants.c
q_e = picmi.constants.q_e

print('status2')

# Domain decomposition
max_grid_size = 64
blocking_factor = 32

# Create grid
grid = picmi.Cartesian2DGrid(
    number_of_cells=[64, 64],# 256],
    lower_bound=[-0.02, -0.02],# -0.02],
    upper_bound=[0.02, 0.02],# 0.02],
    lower_boundary_conditions = ['periodic', 'periodic'],# 'dirichlet'],
    upper_boundary_conditions = ['periodic', 'periodic'],# 'dirichlet'],
    lower_boundary_conditions_particles = ['periodic', 'periodic'],#, 'absorbing'],
    upper_boundary_conditions_particles = ['periodic', 'periodic'],# 'absorbing'],
    moving_window_velocity = [0., 0.],# c],
    warpx_max_grid_size = max_grid_size,
    warpx_blocking_factor = blocking_factor,

)
print('status3')


# Add beam electrons
q_tot = 1e-12
x_m = 0.
y_m = 0.
z_m = -28e-06
x_rms = 0.5e-06
y_rms = 0.5e-06
z_rms = 0.5e-06
ux_m = 0.
uy_m = 0.
uz_m = 500.
ux_th = 2.
uy_th = 2.
uz_th = 50.
gaussian_bunch_distribution = picmi.GaussianBunchDistribution(
    n_physical_particles=q_tot / picmi.constants.q_e,
    rms_bunch_size=[x_rms, y_rms, z_rms],
    rms_velocity=[c * ux_th, c * uy_th, c * uz_th],
    centroid_position=[x_m, y_m, z_m],
    centroid_velocity=[c * ux_m, c * uy_m, c * uz_m]
)
beam = picmi.Species(
    particle_type='electron',
    name='beam',
    initial_distribution=gaussian_bunch_distribution
)

print('status4')

# Set up the field(EM) solver
#sim.solver = picmi.ElectrostaticSolver(grid=grid)
solver = picmi.ElectromagneticSolver(
    grid = grid,
    method = 'Yee',
    cfl = 1.,
    divE_cleaning = 0,)
    #external_fields=[external_field],)

print('status5')

# Set up diagnostics (e.g., particle and field diagnostics)
diag_field_list = ['B', 'E', 'J', 'rho']
particle_diag = picmi.ParticleDiagnostic(
    name = 'particle_diag',
    period = 100,
    write_dir = 'Python_BRsimulation_e/',
    warpx_file_prefix = 'Python_BRsimulation_particle')
field_diag = picmi.FieldDiagnostic(
    name = 'field_diag',
    grid = grid,
    period = 100,
    data_list = diag_field_list,
    write_dir = 'Python_BRsimulation_e/',
    warpx_file_prefix = 'Python_BRsimulation_field')
particle_energy_diag = picmi.ParticleDiagnostic(
    name= 'particle_energy_diag',#'diag1',#'particle_energy_diag',
    period=1,
    species='',
    data_list=['weighting', 'ux', 'uy', 'uz', 'gamma'],
    write_dir='Python_BRsimulation_e/',  # You can set a different directory if needed
    warpx_file_prefix='Python_BRsimulation_radiation')


print('status6')

#Add mag field
external_field = picmi.ConstantAppliedField(Bx= 0.5, By= 0.)#, Bz= 0.)

# Create simulation object
sim = picmi.Simulation(
    solver = solver,
    #external_field=external_field,
    max_steps = max_steps,
    verbose = 1,
    particle_shape = 'cubic',
    warpx_use_filter = 1,
    warpx_serialize_initial_conditions = 1,
    warpx_do_dynamic_scheduling = 0,
    #external_fields=[external_field],
    #warpx_magnetic_field=external_field#solenoidal_magnetic_field
)


# Add beam electrons to simulation
sim.add_species(
    beam,
    layout = picmi.PseudoRandomLayout(grid = grid, n_macroparticles = 100))

print('status8')

# Add diagnostics
sim.add_diagnostic(particle_diag)
sim.add_diagnostic(field_diag)
sim.add_diagnostic(particle_energy_diag)

print('status9')

# Write input file that can be used to run with the compiled version
sim.write_input_file(file_name = 'inputs_2d_electron')

print('status10')

# Run the simulation
# Initialize inputs and WarpX instance
sim.initialize_inputs()
sim.initialize_warpx()

#sim.initialize()
print('status11')

# Advance simulation until last time step
sim.step(max_steps)
