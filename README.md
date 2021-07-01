# Multitester
Multitester is a package for using various topology-reliant neutrino point source detection methods

## Setup

To get started, place `tracks_mc.npy` and `cascade_mc.npy` from the following Dropbox link into `Multitester/mcdata/` https://www.dropbox.com/sh/qjl9xz0tpkam8ms/AAA4ECvAsTdGkys9fp969AOSa?dl=0.
Then install the conda environment from multitester.yml.

The data comes from the PStracks and DNNcascades datasets. These original, unformatted MC files are also present in the above dropbox. The tracks/cascade mc files have been formatted to uniform datatypes and cut any unnecessary variables.

## Tester Object

A tester object holds various point source methods and has the following functionalities:
* Data sample creation from MC (PS tracks + DNN cascades)
* TS calculations
* Width based SDAG process for creating a background TS distribution
* Hypothesis testing
* TC Space creation + discovery potential
* Healpy sky maps/ full sky scans
* Creation of KDE fS/fB  splines from MC data

__Each function in the multitester object has a header in the script explaining every input/output.__
A tester object stores the _exact_ names of the methods you want to use and a count of tracks and cascade events to use as a MC background.
Furthermore each tester contains an args dictionary (indexed by calling the object itself as the dictionary i.e. tester['var']). The values of this dictionary are either passed in or set up as defaults during the initialization phase. The current list of arguments is as follows:
* `delta_ang`: Angle (rad) around source at which to cut events from the TS calculation.
    * Defaults to 0, which uses the whole sky
* `dec_bands`: Bands in declination in which to compare TS in hypothesis testing. Passed in as a 2D array-like where `dec_bands[i]` gives an array-like with `[band_min, band_max]
    * Defaults to .2 rad bins in sinDec
    * [[-pi/2, pi/2]] used for a whole sky scan
* `Bt`: (Function-like) Spatial background pdf in sinDec created by using a KDE on track event MC from an E^-2 spectrum
* `Bc`: (Function-like) Spatial background pdf in sinDec created by using a KDE on cascade event MC from an E^-3.7 spectrum
* `Est`:(Function-like) Energy background pdf in logE created by using a KDE on track MC data from an E^-2 spectrum
* `Ebt`:(Function-like) Energy background pdf in logE created by using a KDE on track MC data from an E^-3.7 spectrum
* `Esc`:(Function-like) Energy background pdf in logE created by using a KDE on cascade MC data from an E^-2 spectrum
* `Ebc`:(Function-like) Energy background pdf in logE created by using a KDE on cascade MC data from an E^-3.7 spectrum
* `B`: (Function-like) Spatial background pdf in sinDec created by using a KDE on MC data from an E^-2 spectrum (Currently equivalent to Bt)
* `Es`: (Function-like) Energy background pdf in logE created by using a KDE on MC data from an E^-2 spectrum (Currently equivalent to Est)
* `Eb`: (Function-like) Energy background pdf in logE created by using a KDE on MC data from an E^-3.7 spectrum (Currently equivalent to Ebt)
* `Tau`:(Function-like) Topology pmf for signal cascades as a function of spectral index.
    * Gives the percent of signal event excpected to be cascades at a given index
    * 1-Tau gives the pmf for tracks    



## Files/ Directories

`package.py`- Python package script to be imported.
* Contains PS method functions, utility functions, and the tester object definition
* Imports many other python packages present in the provided conda environment

`bkg_gen.py`- [Intended to only be ran by `tester.run()`] Creates a background distribution/calculates p-values
* Loads in a pickled tester and creates background TS in accordance with the values defined in the tester.
* n a process defined by `tester.run()`, will compare loaded in signal trials against the portion of the background distribution created with this script.
* Will be ran nfile times by `tester.run()` for parallelization

`signal_gen.py`- [Intended to only be ran by `tester.run()`] Creates a distribution of source TS
* Saves a signal TS ditribution according to injection counts defined in `tester.run()`

`sigsum.py` - [Intended to only be ran by `tester.run()`] Combines p-values from each background distribution piece into a final p-value, and then significance.
* Saves a file of significances calculated from signal dist compared to background dist

`repackage.py` - Repackages separate background TS distributions into one sorted distribution.
* Loads in a pickled tester object and combines any background distributions in `./working` whose filenames contain the same name as the tester.

`sdag.py`- Script controlling the SLURM SDAG functionality. Pulled from https://github.com/abdulrahmanazab/sdag

`./utils/data_format.py`- Formats an IceCube numpy MC dataset for use with this package
* The provided `tracks_mc.npy` and `cascade_mc.npy` are already formatted in this way
* To format new IceCube MC files, run `python data_format.py {filename.npy} {topology}`
    * topology = 0 for tracks and 1 for cascades

`./working/`- Will contain all temporary files created by `tester.run()`

`./data/` - Each time a tester object is created in Multitester a subdirectory `./data/{tester.name}/` will be created to hold any files created for that object

`./plots/` - Each time a tester object is created in Multitester a subdirectory `./plots/{tester.name}/` will be created to hold any plot pngs created for that object

`./Testers/`- Holds pickled tester objects
