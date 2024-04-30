"""Contains helper functions used in other routines.

Copyright: Simon R Proud, 2022
License: GNU GPL v3
"""

from satpy import Scene
from numba import jit
from glob import glob
import numpy as np


def load_sat():
    """This loads the satellite data from file, only required to get the area defs.
    Paths are hardcoded - you'll need to change these!"""
    g16_file = 'I:/sat_data/GOES/OR_ABI-L1b-RadC-M6C02_G16_s20221642356173_e20221642358546_c20221642358568.nc'
    g17_file = 'I:/sat_data/GOES/OR_ABI-L1b-RadF-M6C02_G17_s20221642350319_e20221642359386_c20221642359411.nc'
    g16_r = load_abi(g16_file)
    g17_r = load_abi(g17_file)

    return g16_r, g17_r


@jit(nopython=True)
def get_dist_2sat(sat1_lon, sat1_lat, sat2_lon, sat2_lat, calt, palts, pdists, plats, plons):
    """Find the good matches between parallax corrected data using two satellites.
    Args:
    - sat1_lon / sat1_lat: Lons and lats for corrected points, 1st satellite
    - sat1_lon / sat2_lat: Lons and lats for corrected points, 2nd satellite
    - calt: current altitude being processed
    - palts: Best estimate of altitude thus far
    - pdists: Smallest distances corresponding to altitudes above
    - plats: Latitudes corresponding to smallest distances
    - plats: Longitudes corresponding to smallest distances
    """

    # Compute distance between sat points
    dist1 = np.sqrt((sat1_lon - sat2_lon) * (sat1_lon - sat2_lon) +
                    (sat1_lat - sat2_lat) * (sat1_lat - sat2_lat))

    # Compute mean position
    mea_lat = (sat1_lat + sat2_lat) / 2
    mea_lon = (sat1_lon + sat2_lon) / 2

    # Where the computed position is better than current position, use the new one
    palts = np.where(dist1 < pdists, calt, palts)
    plats = np.where(dist1 < pdists, mea_lat, plats)
    plons = np.where(dist1 < pdists, mea_lon, plons)
    pdists = np.where(dist1 < pdists, dist1, pdists)
    return palts, pdists, plats, plons


@jit(nopython=True)
def get_dist(sat1_lon, sat1_lat, sat2_lon, sat2_lat, sat3_lon, sat3_lat, calt, palts, pdists, plats, plons):
    """Find the good matches between parallax corrected data using three satellites.
    Args:
    - sat1_lon / sat1_lat: Lons and lats for corrected points, 1st satellite
    - sat1_lon / sat2_lat: Lons and lats for corrected points, 2nd satellite
    - sat3_lon / sat3_lat: Lons and lats for corrected points, 3rd satellite
    - calt: current altitude being processed
    - palts: Best estimate of altitude thus far
    - pdists: Smallest distances corresponding to altitudes above
    - plats: Latitudes corresponding to smallest distances
    - plats: Longitudes corresponding to smallest distances
    """

    # Compute intra-satellite distances
    dist1 = np.sqrt((sat1_lon - sat2_lon) * (sat1_lon - sat2_lon) +
                    (sat1_lat - sat2_lat) * (sat1_lat - sat2_lat))
    dist2 = np.sqrt((sat1_lon - sat3_lon) * (sat1_lon - sat3_lon) +
                    (sat1_lat - sat3_lat) * (sat1_lat - sat3_lat))
    dist3 = np.sqrt((sat3_lon - sat2_lon) * (sat3_lon - sat2_lon) +
                    (sat3_lat - sat2_lat) * (sat3_lat - sat2_lat))

    # Compute main distance
    dist_main = np.sqrt(dist1 * dist1 + dist2 * dist2 + dist3 * dist3)

    # Compute mean position
    mea_lat = (sat1_lat + sat2_lat + sat3_lat) / 3
    mea_lon = (sat1_lon + sat2_lon + sat3_lon) / 3

    # Where the computed position is better than current position, use the new one
    palts = np.where(dist_main < pdists, calt, palts)
    plats = np.where(dist_main < pdists, mea_lat, plats)
    plons = np.where(dist_main < pdists, mea_lon, plons)
    pdists = np.where(dist_main < pdists, dist_main, pdists)

    return palts, pdists, plats, plons


def get_proj_data(in_scn):
    """Get the satellite position and Earth shape for a given projection / area
    Args:
    - in_scn: A single dataset in a satpy.Scene
    Returns:
    - sat_x: Satellite position on x axis (km)
    - sat_y: Satellite position on y axis (km)
    - sat_z: Satellite position on z axis (km)
    - rad_rat:
    - eqr_rad: Earth's equatorial radius (km)
    - pol_rad: Earth's polar radius (km)
    """
    in_area = in_scn.attrs['area']

    if 'a' in in_area.proj_dict:
        eqr_rad = in_area.proj_dict['a'] / 1000.
        fla_rad = in_area.proj_dict['rf']
    else:
        eqr_rad = 6378.1370
        fla_rad = 298.257222096

    if 'satellite_actual_altitude' in in_scn.attrs['orbital_parameters']:
        sat_h = (in_scn.attrs['orbital_parameters']['satellite_actual_altitude'] / 1000.) + eqr_rad
        sat_lon = np.deg2rad(in_scn.attrs['orbital_parameters']['satellite_actual_longitude'])
        sat_lat = np.deg2rad(in_scn.attrs['orbital_parameters']['satellite_actual_latitude'])
    elif 'satellite_nominal_longitude' in in_scn.attrs['orbital_parameters']:
        sat_h = (in_scn.attrs['orbital_parameters']['satellite_nominal_altitude'] / 1000.) + eqr_rad
        sat_lon = np.deg2rad(in_scn.attrs['orbital_parameters']['satellite_nominal_longitude'])
        sat_lat = np.deg2rad(in_scn.attrs['orbital_parameters']['satellite_nominal_latitude'])
    else:
        raise ValueError('No sat data available.')

    pol_rad = eqr_rad - (eqr_rad / fla_rad)
    rad_rat = (eqr_rad / pol_rad) * (eqr_rad / pol_rad)
    sat_lat_m = np.arctan(np.tan(sat_lat) / rad_rat)

    sat_x = sat_h * np.cos(sat_lat_m) * np.cos(sat_lon)
    sat_y = sat_h * np.cos(sat_lat_m) * np.sin(sat_lon)
    sat_z = sat_h * np.sin(sat_lat_m)

    return sat_x, sat_y, sat_z, rad_rat, eqr_rad, pol_rad


@jit(nopython=True)
def do_get_parallax(sat_x, sat_y, sat_z, rad_rat, eqr_rad, pol_rad, obs_lat, obs_lon, obs_alt):
    """Do the actual parallax correction, assuming nonspherical Earth.
    Args:
    - sat_x, sat_y, sat_z: Cartesian position of the satellite
    - rad_rat: Ratio of Earth's polar and equatorial radii in geometrical model
    - eqr_rad: The equatorial radius in km
    - pol_rad: The polar radius in km
    - obs_lat: The observation latitude (to be parallax corrected)
    - obs_lon: The observation longitude (to be parallax corrected)
    - obs_alt: The altitude above Earth's surface of the observation.
    Returns:
    new_lat: Parallax corrected latitude
    new_lon: Parallax corrected longitude
    """
    obs_lat = np.radians(obs_lat)
    obs_lon = np.radians(obs_lon)

    obs_h = obs_alt

    # Modified latitude due to non-sphericity of Earth
    obs_lat1 = np.arctan(np.tan(obs_lat) / rad_rat)

    # Radius of Earth centre at given point
    r = np.sqrt(eqr_rad * eqr_rad * np.cos(obs_lat1) * np.cos(obs_lat1) +
                pol_rad * pol_rad * np.sin(obs_lat1) * np.sin(obs_lat1))

    # X, Y, Z position of observation on surface
    omod_x = r * np.cos(obs_lat1) * np.cos(obs_lon)
    omod_y = r * np.cos(obs_lat1) * np.sin(obs_lon)
    omod_z = r * np.sin(obs_lat1)

    # Radius from Earth centre
    ob_mod = np.sqrt(omod_x * omod_x +
                     omod_y * omod_y +
                     omod_z * omod_z)

    mod_x = sat_x - omod_x
    mod_y = sat_y - omod_y
    mod_z = sat_z - omod_z

    ax_mod = np.sqrt(mod_x * mod_x +
                     mod_y * mod_y +
                     mod_z * mod_z)

    cosbeta = -(omod_x * mod_x +
                omod_y * mod_y +
                omod_z * mod_z) / (ob_mod * ax_mod)
    aa = 1.
    bb = -2 * ob_mod * cosbeta
    cc = ob_mod * ob_mod - (ob_mod + obs_h) * (ob_mod + obs_h)
    x = (-bb + np.sqrt(bb * bb - 4 * aa * cc)) / (2 * aa)

    axis_x = omod_x + (x / ax_mod) * mod_x
    axis_y = omod_y + (x / ax_mod) * mod_y
    axis_z = omod_z + (x / ax_mod) * mod_z

    new_lat = np.arctan(axis_z / np.sqrt((axis_x * axis_x) + (axis_y * axis_y)))
    new_lat = np.arctan(np.tan(new_lat) * rad_rat)
    new_lat = np.degrees(new_lat)
    new_lon = np.degrees(np.arctan2(axis_y, axis_x))

    return new_lat, new_lon


def load_ami(infile, chan='VI006'):
    """Load GK-2A data.
    Args:
    - infile: Single observation file.
    - chan: Optional, the channel to load. VI006 is preferred.
    Returns:
     x,y,z position of satellite.
     flattening ratio of Earth
     equatorial radius
     polar radius
     """
    scn_ami = Scene([infile], reader='ami_l1b')
    scn_ami.load([chan])
    return get_proj_data(scn_ami[chan])


def load_agri(infile, chan='C13'):
    """Load FY-4B data.
    Args:
    - infile: Single observation file.
    - chan: Optional, the channel to load. C13 is preferred.
    Returns:
     x,y,z position of satellite.
     flattening ratio of Earth
     equatorial radius
     polar radius
     """
    scn_agri = Scene([infile], reader='agri_fy4b_l1')
    scn_agri.load([chan])
    return get_proj_data(scn_agri[chan])


def load_abi(infile, chan='C03'):
    """Load GOES-17 data.
    Args:
    - infile: Single observation file.
    - chan: Optional, the channel to load. C03 is preferred.
    Returns:
     x,y,z position of satellite.
     flattening ratio of Earth
     equatorial radius
     polar radius
     """
    scn_abi = Scene([infile], reader='abi_l1b')
    scn_abi.load([chan])
    return get_proj_data(scn_abi[chan])


def load_ahi(indir, stub='0540', chan='B03'):
    """Load Himawari data.
    Args:
    - infile: Single observation file.
    - stub: String to search for in files, narrows down selection - use one timeslot
    - chan: Optional, the channel to load. B03 is preferred.
    Returns:
     x,y,z position of satellite.
     flattening ratio of Earth
     equatorial radius
     polar radius
     """
    ahi_files = glob(f'{indir}/*{stub}*.DAT')
    scn_ahi = Scene(ahi_files, reader='ahi_hsd')
    scn_ahi.load([chan])
    return get_proj_data(scn_ahi[chan])


def plist():
    """Return points list for 3-satellite comparison.
    Columns are:
    ID, AHI_LON, AHI_LAT, AMI_LON, AMI_LAT, ABI_LON, ABI_LAT
    """
    return [[1, 125.2723, 2.3255, 125.3427, 2.3301, 125.445, 2.3252],
            ]


def plist2():
    """Return points list for 2-satellite comparison at 04:30 UTC
    Columns are:
    ID, G16_LON, G16_LAT, G17_LON, G17_LAT
    """
    return [[-1, 125.2723, 2.3255, 125.3427, 2.3301],
           ]


def make_random_set(lat, lon, num=10000, nrange=0.02):
    """Add gaussian noise to data to simulate uncertainty in position.
    Args:
     - lat: Observation latitude
     - lon: Observation longitude
     - num: Optional, number of observations to simulate
     - nrange: Optional, standard deviation of random noise
    """

    olats = lat + np.random.normal(0, nrange, num)
    olons = lon + np.random.normal(0, nrange, num)
    return olats, olons
