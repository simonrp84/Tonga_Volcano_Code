from numba import jit
import numpy as np
import copy

@jit(nopython=True)
def int_temp_prof(temp_prof, lapse_rate=-6.5):
    """Alter a temperature profile to cool above tropopause, simulating overshooting top.
    Args:
    - temp_prof:  Numpy array with ECMWF profile (n_level x 3: Pressure, Temperature, Height)
    - lapse_rate: Optional float specifying above-tropopause lapse rate in K/km
    Returns:
    Modified temperature profile.

    NOTE: This is very simple and gives nonsense values at extreme altitudes, these can be ignored.
    """
    
    nbins = len(temp_prof)
    t = temp_prof[:, 1]
    h = temp_prof[:, 2]

    out_prof = temp_prof.copy()

    # Find tropopause. We don't do anything fancy here as we know it'll be the coldest point.
    # As in this ECMWF profile the troposphere and upper mesosphere have been manually removed.
    # Nominally this should be the first point in the array
    trop_pt = np.argmin(t)
    base_h = h[trop_pt]
    base_t = t[trop_pt]
    for i in range(trop_pt+1, nbins):
        hdiff = (h[i] - base_h) / 1000.
        out_prof[i, 1] = base_t + lapse_rate * hdiff
    
    return out_prof


@jit(nopython=True)
def find_hbin(bright_temp, temp_prof, temp_prof_int):
    """Find best estimate of temperature bin from brightness temperature.
    Args:
    - bright_temp: Float, the brightness temperature
    - temp_prof: Numpy array with ECMWF profile (n_level x 3: Pressure, Temperature, Height)
    - temp_prof_int: Numpy array with ECMWF profile as above, but extrapolated above tropopause for cooling at 6.5K/km (overshoot)
    Returns:
    - int with bin value from bright_temp
    """
    nbins = len(temp_prof)
    best_bin = -1
    switch_cool = False

    # Select profile to use depending on brightness temperature
    if bright_temp < np.nanmin(temp_prof):
        # Extrapolated for cooling
        prof_inuse = temp_prof_int
        switch_cool = True
    else:
        # Normal
        prof_inuse = temp_prof

    # Loop over bins
    for i in range(0, nbins-1):
        # This is the case for a warming (stratospheric) profile
        if not switch_cool:
            if (bright_temp > prof_inuse[i] and bright_temp < prof_inuse[i+1]):
                best_bin = i
                break
        # And this is the case for a cooling (overshoot) profile
        else:
            if (bright_temp < prof_inuse[i] and bright_temp > prof_inuse[i+1]):
                best_bin = i
                break

    return best_bin

@jit(parallel=True)
def est_height_and_pressure(bt_image, tprof, mask=None):
    """Estimate plume height and pressure using cloud top temp and ECMWF profile.
    Args:
    - bt_image: Numpy array with brightness temperatures for plume
    - tprof: Numpy array with ECMWF profile (n_level x 3: Pressure, Temperature, Height)
    - mask: Optional numpy array to mask non-plume pixels. '1' is assumed plume and '0' non-plume.
    Returns:
    - numpy arrays with height and pressure values

    WARNING: This is not efficient!
    """
    # Initialise empty mask if none is present
    if mask is None:
        mask = bt_image.copy()
        mask[:,:] = 0
    
    # Initialise output arrays
    shaper = bt_image.shape
    out_pres = np.zeros(shaper)
    out_height = np.zeros(shaper)
    out_pres[:,:] = np.nan
    out_height[:,:] = np.nan
    tprof2 = int_temp_prof(tprof)

    # Loop over pixels
    for x in range(0, shaper[0]):
        for y in range(0, shaper[1]):
            # Only process those that aren't masked
            if mask[x,y] == 1:
                # Find best bin
                bbin = find_hbin(bt_image[x, y], tprof[:, 1], tprof2[:, 1])
                # Get pres and height for that bin
                out_pres[x, y] = tprof[bbin, 0]
                out_height[x, y] = tprof[bbin, 2]

    return out_height, out_pres