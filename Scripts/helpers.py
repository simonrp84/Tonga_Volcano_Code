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
    return [[1, -175.01309, -20.8084, -174.79875, -20.81208, -175.49986, -20.80498],
            [2, -175.00004, -20.64605, -174.75817, -20.65286, -175.60272, -20.66277],
            [3, -175.05543, -20.10941, -174.87484, -20.12475, -175.73037, -20.13835],
            [4, -175.23897, -21.12437, -175.06042, -21.12223, -175.29307, -21.08983],
            [5, -174.80269, -20.60336, -174.53016, -20.5942, -175.55025, -20.6205],
            [6, -175.22942, -21.67993, -174.86853, -21.64087, -176.15363, -21.62233],
            [7, -174.67983, -20.58513, -174.13079, -20.61891, -175.92322, -20.57018],
            [8, -175.51812, -22.21504, -175.09047, -22.20602, -176.3926, -22.1952],
            [9, -174.5609, -20.71734, -174.04259, -20.77633, -175.56792, -20.72325],
            [10, -175.13169, -20.70911, -174.64724, -20.74245, -176.32941, -20.74081],
            [11, -174.09382, -20.40346, -173.75943, -20.43426, -174.9112, -20.39943],
            [12, -174.04246, -20.44589, -173.7052, -20.46034, -174.8885, -20.43645],
            [13, -176.37841, -19.9683, -176.13872, -20.01629, -177.16269, -19.97321],
            [14, -174.68168, -20.74929, -174.12363, -20.78002, -175.98361, -20.71822],
            [15, -174.59792, -20.83659, -174.0268, -20.8722, -175.93218, -20.82284],
            [16, -174.6249, -20.93171, -174.16609, -20.95132, -175.62642, -20.92053],
            [17, -175.44782, -20.44286, -175.05453, -20.44098, -176.43492, -20.64499],
            [18, -174.71568, -20.66946, -174.37345, -20.66672, -175.59777, -20.6435],
            [19, -174.99507, -20.73894, -174.64008, -20.74731, -175.90277, -20.74087],
            [20, -174.78325, -20.53439, -174.44103, -20.56232, -175.64823, -20.5482],
            [21, -176.20679, -20.23628, -175.89096, -20.24664, -177.02231, -20.13151],
            [22, -177.48367, -20.99698, -177.17383, -21.0054, -178.31301, -21.00248],
            [23, -174.60307, -20.19243, -174.26353, -20.21444, -175.42931, -20.18502],
            [24, -175.51284, -21.74557, -175.17488, -21.76797, -176.30915, -21.74314],
            [25, -175.48177, -19.23563, -175.14534, -19.25853, -176.27397, -19.25005],
            [26, -177.17804, -19.88003, -176.88732, -19.89131, -177.93785, -19.8802],
            [27, -178.20801, -23.19096, -177.91411, -23.20185, -179.00665, -23.18606],
            [28, -178.25918, -18.56383, -178.02022, -18.55635, -178.87424, -18.45995],
            ]


def plist2():
    """Return points list for 2-satellite comparison at 04:30 UTC
    Columns are:
    ID, G16_LON, G16_LAT, G17_LON, G17_LAT
    """
    return [[-1, -84.57275, 38.14747, -84.10212, 38.20120],
            [-2, -82.63691, 39.33295, -82.07619, 39.35938],
            [-3, -82.62477, 38.93836, -82.13572, 38.95985],
            [-4, -83.52820, 38.61363, -83.14601, 38.67604],
            [-5, -81.24375, 39.37122, -80.91327, 39.40788],
            [-6, -82.99750, 39.07736, -82.54086, 39.11643],
            [-7, -83.58308, 39.42462, -82.96944, 39.39078],
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
