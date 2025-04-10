import json
import numpy as np
import copy
import os
import scipy.constants

spots = np.array([0]) * 1e-6
scan_parameters = spots
scan_parameter_name = 'σ'
charges = {'drive': 1.5e-9, 'witness': 0.5e-9}

if __name__ == '__main__':
    with open('qpinput.json', 'r') as f:
        prototype = json.load(f)
        n0 = prototype['simulation']['n0']
        kpn1 = 299792458 / np.sqrt(n0 * 100 * 100 * 100 * 1.602176634e-19 * 1.602176634e-19 / (9.109383701528e-31 * 8.854187812813e-12))

    for i, spot in enumerate(spots):
        instance = copy.deepcopy(prototype)
        instance['beam'][1]['peak_density'] = 0.5e-9 / (n0 * 1e6 * scipy.constants.elementary_charge * ((2 * np.pi) ** (3/2)) * spot * spot * 0.3058958756 * kpn1)
        instance['beam'][1]['sigma'][0] = spot / kpn1
        instance['beam'][1]['sigma'][1] = spot / kpn1
        instance['beam'][1]['sigma_v'][0] = 5e-6 / spot
        instance['beam'][1]['sigma_v'][1] = 5e-6 / spot
        os.system(f'mkdir -p simulations/{i}')
        with open(f'simulations/{i}/qpinput.json', 'w+') as f:
            json.dump(instance, f, indent=4)
        os.system(f'bash run.sh {i}')
