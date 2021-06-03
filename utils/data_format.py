'''
Run this script with your two topology data files to make sure they are in the right format
Run: data_format.py 'npy filename' 'topology (0 for track, 1 for cascade)'
    ex 'data_format.py track_data.npy 0'
Returns properly formatted and named data to be placed into Multitester/mcdata
'''

import sys
import numpy as np
import pandas as pd 

data = np.load(sys.argv[1])
topo = int(sys.argv[2])

names = ('ra',
         'dec',
         'angErr',
         'logE',
         'ow',
         'topo')

formats = ('<f8',
           '<f8',
           '<f8',
           '<f8',
           '<f8',
           '<i8')

dtype_new = np.dtype({'names': names, 'formats': formats})

for name in names:
    #checks if all needed fields are in your data
    if name not in data.dtype.names and name != 'topo':
        raise ValueError(f'Your data is missing {name}!')
    selected = pd.DataFrame(data)[[*names[:-1]]]
    selected['topo'] = topo 
    selected = selected.to_records(index = False)
    clean = np.array(selected.astype(dtype_new))

if topo:
    np.save('cascade_mc.npy', clean)
    print("Data successfully saved to 'cascade_mc.npy'.")
else: 
    np.save('tracks_mc.npy', clean)
    print("Data successfully saved to 'tracks_mc.npy'.")