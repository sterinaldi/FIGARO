#from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
#from tqdm import tqdm
#import healpy as hp
import h5py
import pandas as pd
#from tqdm import tqdm
from pathlib import Path
import optparse as op

'''
GLADE+ columns (29/04/2022)

0    GLADE no    GLADE+ catalog number
1    PGC no    Principal Galaxies Catalogue number
2    GWGC name    Name in the GWGC catalog
3    HyperLEDA name    Name in the HyperLEDA catalog
4    2MASS name    Name in the 2MASS XSC catalog
5    WISExSCOS name    Name in the WISExSuperCOSMOS catalog (wiseX)
6    SDSS-DR16Q name    Name in the SDSS-DR16Q catalog
7    Object type flag    Q: the source is from the SDSS-DR16Q catalog, G:the source is from another catalog and has not been identified as a quasar
8    RA    Right ascension in degrees
9    Dec    Declination in degrees
10    B    Apparent B magnitude
11    B_err    Absolute error of apparent B magnitude
12    B flag    0: the B magnitude is measured, 1: the B magnitude is calculated from the B_J magnitude
13    B_Abs    Absolute B magnitude
14    J    Apparent J magnitude
15    J_err    Absolute error of apparent J magnitude
16    H    Apparent H magnitude
17    H_err    Absolute error of apparent H magnitude
18    K    Apparent K_s magnitude
19    K_err    Absolute error of apparent K_s magnitude
20    W1    Apparent W1 magnitude
21    W1_err    Absolute error of apparent W1 magnitude
22    W2    Apparent W2 magnitude
23    W2_err    Absolute error of apparent W2 magnitude
24    W1 flag    0: the W1 magnitude is measured, 1: the W1 magnitude is calculated from the K_s magnitude
25    B_J    Apparent B_J magnitude
26    B_J err    Absolute error of apparent B_J magnitude
27    z_helio    Redshift in the heliocentric frame
28    z_cmb    Redshift converted to the Cosmic Microwave Background (CMB) frame
29    z flag    0: the CMB frame redshift and luminosity distance values given in columns 25 and 28 are not corrected for the peculiar velocity, 1: they are corrected values
30    v_err    Error of redshift from the peculiar velocity estimation
31    z_err    Measurement error of heliocentric redshift
32    d_L    Luminosity distance in Mpc units
33    d_L err    Error of luminosity distance in Mpc units
34    dist flag    0: the galaxy has no measured redshift or distance value, 1: it has a measured photometric redshift from which we have calculated its luminosity distance, 2: it has a measured luminosity distance value from which we have calculated its redshift, 3: it has a measured spectroscopic redshift from which we have calculated its luminosity distance
35    M*    Stellar mass in 10^10 M_Sun units
36    M*_err    Absolute error of stellar mass in 10^10 M_Sun units
37    M* flag    0: if the stellar mass was calculated assuming no active star formation, 1: if the stellar mass was calculated assuming active star formation
38    Merger rate    Base-10 logarithm of estimated BNS merger rate in the galaxy in Gyr^-1 units
39    Merger rate error    Absolute error of estimated BNS merger rate in the galaxy
'''

def main():
    """
    Preprocess the .txt file downloaded from the GLADE+ website - http://glade.elte.hu - selecting only some columns.
    The output file is a lighter .hdf5 file, saved in the same folder as the .txt file, that has the structure required by figaro.threeDvolume.VolumeReconstruction.load_glade
    """
    parser = op.OptionParser()
    parser.add_option("-i", type = "string", dest = "glade_file", help = "GLADE+ txt file")
    
    (options, args) = parser.parse_args()
    glade_file      = Path(options.glade_file).resolve()
    glade_folder    = glade_file.parent
    
    dict_out={}

    # Load all the columns of interest, please note that usecols must be in creasing order
    chunk = pd.read_csv(glade_file,
                        usecols=(7, 8, 9, 10, 12, 18, 20, 25, 27, 28, 29, 30, 31, 34),
                        header=None,
                        names=['objtype', 'ra', 'dec', 'B', 'Bflag', 'K', 'W1', 'bJ', 'zhelio', 'zcmb', 'pecflag', 'pecerr', 'zhelioerr', 'redflag'],
                        delim_whitespace=True,
                        na_values="null",
                        )

    # Check the galaxies
    COND_GALAXY = (chunk['objtype']=='G')

    # Check if the redshift is measured and not obtained from dl
    COND_RED_ORIGIN = True#(chunk['redflag']==1) | (chunk['redflag']==3)

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

#    # Finds the numpy index, just as general info
#    nside = 1024
#    ind = hp.pixelfunc.ang2pix(nside,dict_out['dec']+np.pi/2,dict_out['ra'])

#     Limits for an all sky catalogs
#    ra_dec_lim = 0
#    ra_min = 0.0
#    ra_max = np.pi*2.0
#    dec_min = -np.pi/2.0
#    dec_max = np.pi/2.0

    # Just a test print. I wanted to check how many galaxies had B from bj. I got around 10%
#    idx0=np.where(dict_out['Bflag']==0)[0]
#    idx1=np.where(dict_out['Bflag']==1)[0]
#    print('bJflag ratio 0/1', len(idx0)/len(idx1))

    # Saves the catalog in hdf5 format
    with h5py.File(Path(glade_folder, "glade+.hdf5"), "w") as f:
        f.create_dataset("ra", data=dict_out['ra'])
        f.create_dataset("dec", data=dict_out['dec'])
        f.create_dataset("z", data=dict_out['zcmb'])
#        f.create_dataset("sigmaz", data=(dict_out['zhelioerr']/dict_out['zhelio'])*dict_out['zcmb'])
#        f.create_dataset("skymap_indices", data=ind)
#        f.create_dataset("radec_lim", data=np.array([ra_dec_lim,ra_min,ra_max,dec_min,dec_max]))
        for j in range(len(bands)):
            f.create_dataset("m_{0}".format(bands[j]), data=dict_out[bands[j]])

if __name__ == '__main__':
    main()
