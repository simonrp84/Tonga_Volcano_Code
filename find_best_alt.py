"""A script to find the best altitude match for a volcanic plume.
This uses the parallax effect between data from three satellites to estimate altitude.

Note that this script is not efficient, it will take a *long* time to run.
"""

import helpers as hlp
import numpy as np
import warnings
import sys

# Ignore projection warnings from satpy, deep space pixels confuse it.
warnings.filterwarnings('ignore')


def _do_print(listnum, pdist, palt, plat, plon, nprint=100):
    idx = np.argpartition(pdist, nprint)
    mlat = np.nanmean(plat)
    mlon = np.nanmean(plon)
    malt = np.nanmean(palt)
    mdis = np.nanmean(pdist)
    salt = np.nanstd(palt)
    sdis = np.nanstd(pdist)
    galt = np.nanmean(palt[idx[:20]])
    gdis = np.nanmean(pdist[idx[:20]])
    gstd = np.nanstd(palt[idx[:20]])

    print(f'  {listnum:2d}    | '
          f'{mlat:6.4f}  | '
          f'{mlon:6.4f} | '
          f'{malt:5.2f}  | '
          f'{mdis:5.2f}  | '
          f'{salt:5.2f}  | '
          f'{sdis:5.2f}  | '
          f'{galt:5.2f}  | '
          f'{gdis:5.5f}  | '
          f'{gstd:5.2f}  | ')


def run_2sat(ahi_ri, ami_ri, plist, invar=2, maxalt=60):
    """Find best altitude estimate using 2 satellites.
    Args:
    - ahi_ri: AHI position from raw data
    - ami_ri: AMI position from raw data
    - plist: List of positions to analyse
    - invar: The current position to analyse
    - maxalt: The maximum altitude to attempt (km)
    """
    varset = plist[invar]

    curnum = varset[0]
    # Get satellite position info
    obs_lon_ahi = varset[1]
    obs_lat_ahi = varset[2]
    obs_lon_ami = varset[3]
    obs_lat_ami = varset[4]

    # Simulate some gaussian noise on the data
    ahi_obs_lats, ahi_obs_lons = hlp.make_random_set(obs_lat_ahi, obs_lon_ahi)
    ami_obs_lats, ami_obs_lons = hlp.make_random_set(obs_lat_ami, obs_lon_ami)

    # Initialise output arrays
    palts = np.zeros(ahi_obs_lats.shape)
    pdists = np.zeros(ahi_obs_lats.shape)
    plats = np.zeros(ahi_obs_lats.shape)
    plons = np.zeros(ahi_obs_lats.shape)
    palts[:] = 1e9
    pdists[:] = 1e9
    plats[:] = 1e9
    plons[:] = 1e9

    # Loop from surface to maxalt
    for alt in np.arange(0, maxalt, 0.05):
        # Get parallax corrected position for each satellite
        ahi_la, ahi_lo = hlp.do_get_parallax(ahi_ri[0], ahi_ri[1], ahi_ri[2],
                                             ahi_ri[3], ahi_ri[4], ahi_ri[5],
                                             ahi_obs_lats, ahi_obs_lons,
                                             alt)
        ami_la, ami_lo = hlp.do_get_parallax(ami_ri[0], ami_ri[1], ami_ri[2],
                                             ami_ri[3], ami_ri[4], ami_ri[5],
                                             ami_obs_lats, ami_obs_lons,
                                             alt)
        # Find best distance matches
        palts, pdists, plats, plons = hlp.get_dist_2sat(ami_lo, ami_la,
                                                        ahi_lo, ahi_la,
                                                        alt, palts, pdists, plats, plons)
    # Print the result
    _do_print(curnum, pdists, palts, plats, plons)


def run_3sat(ahi_ri, ami_ri, abi_ri, plist, invar=2, maxalt=60):
    """Find best altitude estimate using 3 satellites.
    Args:
    - ahi_ri: AHI position from raw data
    - ami_ri: AMI position from raw data
    - abi_ri: ABI position from raw data
    - plist: List of positions to analyse
    - invar: The current position to analyse
    - maxalt: The maximum altitude to attempt (km)
    """

    varset = plist[invar]

    curnum = varset[0]
    # Get satellite position info
    obs_lon_ahi = varset[1]
    obs_lat_ahi = varset[2]
    obs_lon_ami = varset[3]
    obs_lat_ami = varset[4]
    obs_lon_abi = varset[5]
    obs_lat_abi = varset[6]

    # Simulate some gaussian noise on the data
    ahi_obs_lats, ahi_obs_lons = hlp.make_random_set(obs_lat_ahi, obs_lon_ahi)
    ami_obs_lats, ami_obs_lons = hlp.make_random_set(obs_lat_ami, obs_lon_ami)
    abi_obs_lats, abi_obs_lons = hlp.make_random_set(obs_lat_abi, obs_lon_abi)

    # Initialise output arrays
    palts = np.zeros(ahi_obs_lats.shape)
    pdists = np.zeros(ahi_obs_lats.shape)
    plats = np.zeros(ahi_obs_lats.shape)
    plons = np.zeros(ahi_obs_lats.shape)
    palts[:] = 1e9
    pdists[:] = 1e9
    plats[:] = 1e9
    plons[:] = 1e9

    # Loop from surface to maxalt
    for alt in np.arange(0, maxalt, 0.05):
        # Get parallax corrected position for each satellite
        ahi_la, ahi_lo = hlp.do_get_parallax(ahi_ri[0], ahi_ri[1], ahi_ri[2],
                                             ahi_ri[3], ahi_ri[4], ahi_ri[5],
                                             ahi_obs_lats, ahi_obs_lons,
                                             alt)
        ami_la, ami_lo = hlp.do_get_parallax(ami_ri[0], ami_ri[1], ami_ri[2],
                                             ami_ri[3], ami_ri[4], ami_ri[5],
                                             ami_obs_lats, ami_obs_lons,
                                             alt)
        abi_la, abi_lo = hlp.do_get_parallax(abi_ri[0], abi_ri[1], abi_ri[2],
                                             abi_ri[3], abi_ri[4], abi_ri[5],
                                             abi_obs_lats, abi_obs_lons,
                                             alt)
        # Find best distance matches
        palts, pdists, plats, plons = hlp.get_dist(ami_lo, ami_la,
                                                   ahi_lo, ahi_la,
                                                   abi_lo, abi_la,
                                                   alt, palts, pdists,
                                                   plats, plons)
    # Print the result
    _do_print(curnum, pdists, palts, plats, plons)


if __name__ == "__main__":
    per = int(sys.argv[2])
    if sys.argv[1] == '3sat':
        ahi_r, ami_r, abi_r = hlp.load_sat()
        print("DATA FOR MULTIPLE IMAGES (3-sat)")
        print(f"  NUM   | Mean Lon  | Mean Lat  | Mean H | Mean D | Stdv H | Stdv D | Smal H |  Smal D  | Smal S |")
        print(f"________|___________|___________|________|________|________|________|________|__________|________|")
        tplist = hlp.plist()
        run_3sat(ahi_r, ami_r, abi_r, tplist, invar=per)
    elif sys.argv[1] == '2sat':
        ahi_r, ami_r, abi_r = hlp.load_sat()
        print("\n")
        print("DATA FOR 04:30 IMAGE (2-sat)")
        print(f"  NUM   | Mean Lon  | Mean Lat  | Mean H | Mean D | Stdv H | Stdv D | Smal H |  Smal D  | Smal S |")
        print(f"________|___________|___________|________|________|________|________|________|__________|________|")
        tplist = hlp.plist2()
        run_2sat(ahi_r, ami_r, tplist, invar=per)
    else:
        print("No, bad choice:", sys.argv[1], sys.argv[2])
