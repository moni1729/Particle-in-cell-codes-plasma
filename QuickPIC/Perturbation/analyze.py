from run import scan_parameters
import os
for i, scan_parameter in enumerate(scan_parameters):
    os.system(f'bash analyze.sh {i}')
