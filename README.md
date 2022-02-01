# Tonga_Volcano_Code

This repository contains code and data used in the `The January 2022 eruption of Hunga Tonga-Hunga Ha’apai volcano reached the mesosphere` paper submitted in Feb 2022.
Here you will find in the top level directory:
 - Three python notebooks (`Make_FigN.ipynb`) that were used to generate figures 2, 3 and 4.
 - `Simple_BT_Altitude.ipynb`, a notebook that loops over Himawari-8 data to run the IR temperature-based retrieval of plume altitude.
 - `Points_Dual.csv`, which contains the retrieved altitudes and locations of pixels in the 04:30 plume analysis (using AMI and AHI).
 - `Points_Tri.csv`, which contains the positions and retrieved altitudes of pixels across multiple satellite timesteps using all three satellite sensors (AMI, ABI, AHI).
 - `TProfs.csv`, which contains the ECMWF and GFS temperature profiles, pressures and geometric altitudes. 
 - Various other files used in processing, most of which duplicate one or more of the above.
 
The `Height_Out` subdirectory contains TIF files for each Himawari-8 timestep. These files contain float32 values denoting the plume altitude retrieved using the IR brightness temperature method.

The `Heightmap` subdirectory contains TIF files for each 10 minute timestep covering the plume in daylight. These files contain float32 values denoting plume altitude retrieved using the stereoscopic method.

The `Figures` subdirectory contains PNG and EPS versions of the figures (or subfigures) used in the article.
The `Scripts` subdirectory contains:
 - `BT_Height_Helpers.py`: Helper functions for the IR-based temperature retrieval
 - `find_best_alt.py`: A script to run the parallax-based altitude retrieval. It can be run with: `python find_best_alt.py satnum pixnum` where `satnum` is either "3sat" or "2sat", describing whether to run the tri- or dual-satellite retrieval. `pixnum` denotes the pixel number (references in the appropriate csv file) to run.
 - `helpers.py`: Helper functions for the parallax-based altitude retrieval.

