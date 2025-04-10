# Idealized Particle Tracker + Liénard-Wiechert (LW) Radiation

This repository implements several methods for simulating betatron radiation from electron beams in plasma wakefield accelerators (PWFA),
each offering different levels of physical fidelity and computational complexity. 

analytical model
that describes the radiation spectrum from both single particles and full electron beams undergoing betatron oscillations in an
idealized ion channel. Assuming paraxial motion and Gaussian beam distributions, this model provides closed-form expressions for 
the total radiated power and spectral shape, characterized by normalized synchrotron functions. It is especially useful for 
benchmarking numerical simulations and understanding the fundamental scaling of radiation properties with beam parameters such as spot size, energy, and emittance.

#  Liénard–Wiechert (LW) code

Implemented in C++ and parallelized with Boost.MPI. In this model, macro-particles are sampled from a Gaussian distribution and tracked through idealized blowout regime fields
using a fourth-order Runge-Kutta (RK4) integrator. These fields include linear transverse focusing and a constant longitudinal accelerating field. 
Radiation is computed directly from the LW potentials without storing intermediate trajectory data, leading to efficient memory usage. 
The model outputs double-differential spectra over a three-dimensional grid of photon energies and observation angles. 


# QuickPIC

A quasi-static 3D particle-in-cell (PIC) code, which captures the self-consistent plasma response
to the beam while assuming slow beam evolution. QuickPIC does not directly compute radiation, so trajectories from a selected subset of simulation 
particles are exported and interpolated using cubic B-splines. These trajectories are then used to compute radiation with the same LW integration method
as in the previous model.

# OSIRIS
For more complex experimental scenarios involving plasma injection and ultra-low emittance beams, the repository supports simulations with the 
OSIRIS full PIC code combined with LW radiation post-processing. 

OSIRIS is employed to model scenarios like Trojan Horse injection, where a laser ionizes a high-threshold gas species within a preformed plasma blowout,
creating a cold witness beam. These simulations require detailed control over plasma and laser profiles and can include collinear laser injection 
and complex beam-plasma interactions. Particle sampling from OSIRIS requires careful consideration of variable particle weights. 
Sampled trajectories are post-processed with the LW method to compute radiation spectra. 

# EPOCH PIC code 
With a Monte Carlo QED radiation module. This method treats high-energy photon emission as a probabilistic process and is capable of capturing quantum 
effects such as recoil and strong-field radiation. EPOCH applies dispersion-reducing filters and current smoothing functions to mitigate 
numerical Cherenkov radiation (NCR), which can otherwise distort the spectra. Simulations require fine spatial resolution, 
often on the order of hundreds of grid cells in each dimension, to accurately resolve the small-scale features of matched beams in high-density plasmas. 
Despite the heavy computational load, EPOCH provides a fully self-consistent way to model radiation from both drive and witness beams in extreme PWFA regimes.

Together, these models provide a versatile toolkit for simulating and analyzing betatron radiation in modern accelerator experiments. 
They support parameter scans, validation against analytical theory, and integration with machine learning-based beam reconstruction techniques.

[Modeling betatron radiation using particle-in-cell codes for plasma wakefield accelerator diagnostics)](https://arxiv.org/abs/2303.04213)
