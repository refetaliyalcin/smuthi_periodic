#*****************************************************************************#
# This benchmark compares the transmittance of a plane wave impinging onto a  #
# periodic particle arrangement under various incident angles as computed     #
# with Smuthi to that computed with COMSOL (FEM).                             #
# The tetragonal Bravais lattice hosts fifteen spheres of various sizes per   #
# unit cell that are embedded in a dielectic thin film between a glass        #
# substrate and air.                                                          #
# The script runs with Smuthi version 1.2                                     #
#*****************************************************************************#

import smuthi.simulation as sim
import smuthi.initial_field as init
import smuthi.layers as lay
import smuthi.particles as part
import smuthi.fields as flds
import smuthi.periodicboundaries as pb
import smuthi.periodicboundaries.post_processing as pbpost
import numpy as np
from al2o3 import al2o3_n, ag_n, ag_k
from numpy import genfromtxt
from tqdm import tqdm
import sys
import os
import matplotlib.pyplot as plt


# set up parameter sweep
wavelengths = np.arange(300, 510, 10) 
transmittance = np.zeros(len(wavelengths))
reflectance = np.zeros(len(wavelengths))
absorbance = np.zeros(len(wavelengths))

# layer system
layer_system = lay.LayerSystem([0, 3000, 0], [1, 1, 1])
lmax = 2

unit_cell = genfromtxt('spheres_periodic.txt', delimiter=',')
number_of_spheres = np.size(unit_cell, 0)
x_sphere = unit_cell[:,0] 
y_sphere = unit_cell[:,1]
z_sphere = unit_cell[:,2]
s = unit_cell[:,3]

def spheres(wl, x_sphere, y_sphere,z_sphere, s,lmax_val):
    spheres_list = []
    k=0
    for i in s:
        r = s[k]        
        spheres_list.append(part.Sphere(position=[x_sphere[k],
                                                                y_sphere[k],
                                                                z_sphere[k]],
                                                      refractive_index=ag_n(wl) + ag_k(wl)* 1j,
                                                      radius=r,
                                                      m_max=lmax_val,
                                                      l_max=lmax_val))
        k=k+1
    return spheres_list

neffmax = 3
neffimag = 0.01
waypoints = [0, 0.8, 0.8 - 1j * neffimag, 2.1 - 1j * neffimag, 2.1, neffmax]
neff_discr = 1e-3  

for idx, wl in enumerate(tqdm(wavelengths, desc='Wavelength iterations  ', file=sys.stdout,
                            bar_format='{l_bar}{bar}| elapsed: {elapsed} ' 'remaining: {remaining}')):
    print(idx)
    print(wl)
    particle_list = spheres(wl, x_sphere, y_sphere, z_sphere, s, lmax)
    flds.default_Sommerfeld_k_parallel_array = flds.reasonable_Sommerfeld_kpar_contour(vacuum_wavelength=wl,
		                                                                           neff_waypoints=waypoints,
		                                                                           neff_resolution=neff_discr)
    flds.angular_arrays(angular_resolution=np.pi/360)

    # initial field
    initial_field = init.PlaneWave(vacuum_wavelength=wl,
                                   polar_angle=0,
                                   azimuthal_angle=0,
                                   polarization=0,
                                   amplitude=1,
                                   reference_point=[0, 0, 0])
    
    # define unit cell
    a1 = np.array([151, 0, 0], dtype=float)
    a2 = np.array([0, 151, 0], dtype=float) 
    pb.default_Ewald_sum_separation = pb.set_ewald_sum_separation(a1, a2)
    
    # run simulation
    simulation = sim.Simulation(initial_field=initial_field,
                                layer_system=layer_system,
                                particle_list=particle_list,
                                periodicity=(a1, a2),
                                ewald_sum_separation_parameter='default',
                                number_of_threads_periodic=-2) # all but 2 threads
    simulation.run()
    
    # plane wave expansion of total transmitted field
    pwe_total_T = pbpost.transmitted_plane_wave_expansion(initial_field,
                                                          particle_list,
                                                          layer_system,
                                                          a1, a2)
    # plane wave expansion of total reflected field
    pwe_total_R = pbpost.reflected_plane_wave_expansion(initial_field,
                                                        particle_list,
                                                        layer_system,
                                                        a1, a2)
    
    
    # farfield objects       
    ff_T = pbpost.periodic_pwe_to_ff_conversion(pwe_total_T,
                                                simulation.initial_field,
                                                simulation.layer_system)
    ff_R = pbpost.periodic_pwe_to_ff_conversion(pwe_total_R,
                                                simulation.initial_field,
                                                simulation.layer_system)
        
    # power flow per area
    initial_power = pbpost.initial_plane_wave_power_per_area(initial_field, layer_system)
    transmitted_power = pbpost.scattered_periodic_ff_power_per_area(ff_T)
    reflected_power = pbpost.scattered_periodic_ff_power_per_area(ff_R)
    # T, R, A
    transmittance[idx] = transmitted_power / initial_power
    reflectance[idx] = reflected_power / initial_power
    absorbance[idx] = 1 - transmittance[idx] - reflectance[idx]
    print(transmitted_power / initial_power)
    

    
fig, ax = plt.subplots()
ax.plot(wavelengths[:idx], transmittance[:idx],
        linewidth=2, color='C0', label='SMUTHI')


ax.set_xlim(300, 500)
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_ylabel('transmittance')
ax.set_xlabel('Wavelength [nm]')
ax.legend(loc='lower left')
plt.show()

print(transmittance, file=open("./periodic_results.txt", "w"))
print(reflectance, file=open("./periodic_results.txt", "a"))
