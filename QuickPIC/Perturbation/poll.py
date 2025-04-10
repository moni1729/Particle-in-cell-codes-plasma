#!/usr/bin/env python3
from run import scan_parameter_name, scan_parameters
import os
import json
import numpy as np

with open('qpinput.json', 'r') as f:
    prototype = json.load(f)
    dt = prototype['simulation']['dt']
    n0 = prototype['simulation']['n0']
    kpn1 = 299792458 / np.sqrt(n0 * 100 * 100 * 100 * 1.602176634e-19 * 1.602176634e-19 / (9.109383701528e-31 * 8.854187812813e-12))

for i, scan_parameter in enumerate(scan_parameters):
    idx = int(os.popen(f'ls simulations/{i}/Beam0001/Raw | wc -l').read())
    distmm = int(round(100 * dt * idx * kpn1))
    print(f'{scan_parameter_name} = {scan_parameter*1e6:.1f} μm: {distmm}cm')
