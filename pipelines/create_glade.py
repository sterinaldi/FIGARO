from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import healpy as hp
import h5py
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import optparse as op
import configparser

def main():
    parser = op.OptionParser()
    parser.add_option("-i", type = "string", dest = "glade_file", help = "GLADE+ txt file")
    
    (options, args) = parser.parse_args()
    glade_file      = Path(optiosn.glade_file).resolve()
    glade_folder    = glade_file.parent
    
    dict_out={}

    # Load all the columns of interest, please not that usecols must me in creasing order
    chunk = pd.read_csv(glade_file, usecols=(7,8, 9, 10, 12, 18, 20, 25,27,28,29,30,31,34), header=None,
                     names=['objtype','ra', 'dec', 'B', 'Bflag','K','W1', 'bJ','zhelio','zcmb','pecflag','pecerr','zhelioerr','redflag']
                     , delim_whitespace=True, na_values="null")

    # Check the galaxies
    COND_GALAXY = (chunk['objtype']=='G')

    # Check if the redshift is measured and not obtained from dl
    COND_RED_ORIGIN = (chunk['redflag']==1) | (chunk['redflag']==3)

    # Check if peculiar corrections are applied. If not, saves the galaxy only if it is above redshift 0.5
    COND_PEC = (chunk['zcmb']>0.05) | (chunk['pecflag']==1)

    # Check if cmb redshift is positive
    COND_NEGATIVE_RED = chunk['zcmb']>0

    # Combine the above conditions, must be all true
    idx=np.where(COND_GALAXY & COND_RED_ORIGIN & COND_PEC & COND_NEGATIVE_RED)[0]

    # Saves the columns in a disctionary as numpy arrays
    for kk in ['objtype',"ra", "dec", "B", "Bflag", "bJ",'K','W1','zhelio','zcmb','pecflag','pecerr','zhelioerr','redflag']:
        dict_out[kk]=chunk[kk][idx].to_numpy()

    # Save GLADE catalog
    bands=['B', 'K','W1','bJ']
    nGal = len(dict_out['zcmb'])

    # Converts to radians
    dict_out['ra']*=np.pi/180
    dict_out['dec']*=np.pi/180

    ## Finds the numpy index, just as general info
    #nside = 1024
    #ind = hp.pixelfunc.ang2pix(nside,dict_out['dec']+np.pi/2,dict_out['ra'])

    # Limits for an all sky catalog
    ra_dec_lim = 0
    ra_min = 0.0
    ra_max = np.pi*2.0
    dec_min = -np.pi/2.0
    dec_max = np.pi/2.0

    # Just a test print. I wanted to check how many galaxies had B from bj. I got around 10%
    idx0=np.where(dict_out['Bflag']==0)[0]
    idx1=np.where(dict_out['Bflag']==1)[0]
    print('bJflag ratio 0/1', len(idx0)/len(idx1))

    # Saves the catalog in hdf5 format
    with h5py.File(Path(glade_path, "glade+.hdf5"), "w") as f:
        f.create_dataset("ra", data=dict_out['ra'])
        f.create_dataset("dec", data=dict_out['dec'])
        f.create_dataset("z", data=dict_out['zcmb'])
        f.create_dataset("sigmaz", data=(dict_out['zhelioerr']/dict_out['zhelio'])*dict_out['zcmb'])
        f.create_dataset("skymap_indices", data=ind)
        f.create_dataset("radec_lim", data=np.array([ra_dec_lim,ra_min,ra_max,dec_min,dec_max]))
        for j in range(len(bands)):
            f.create_dataset("m_{0}".format(bands[j]), data=dict_out[bands[j]])

if __name__ == '__main__':
    main()
