import scipy.special as spec
import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
import datetime
import sys
import os
import glob
from scipy.optimize import minimize
from scipy.special import factorial
from bisect import bisect
from tqdm import tqdm
import pickle
import scipy.interpolate
from scipy.stats import gaussian_kde
from scipy.interpolate import InterpolatedUnivariateSpline

# MISC UTILITY FUNCTIONS ---
#used in kent dist; spherical dot product
def sph_dot(th1,th2,phi1,phi2):
    return np.sin(th1)*np.sin(th2)*np.cos(phi1-phi2) + np.cos(th1)*np.cos(th2)


def sigmoid(x, c = 3, a = .5):
    return 1 / (1 + np.exp(-c * (x - a)))

def poisson(u, x):
    y = np.exp(-u) * (u**x)/(factorial(x))
    return y

def gaussian(x,sig,mu):
    return np.exp(-np.power(x-mu,2)/(2*np.power(sig,2)))

#Takes in the track x,y,error and an angle to measure at and returns a 'gaussian signal term'
#Uses Kent distribution/ spehrical gaussian [Rob]
def evPSFd(nue,numu):
    kappa = 1./(nue[2])**2
    log_dist = np.log(kappa) - np.log(2*np.pi) - kappa + kappa*sph_dot(np.pi/2-nue[1], np.pi/2-numu[1], nue[0], numu[0])
    return np.exp(log_dist)

#power law function
def Efunc(E, a, b):
    return a * E**b

def pd2sig(p):
    return np.sqrt(2)*spec.erfinv(1-p)

#p-value with bisecting sort algo.
def p_value(x, bkg):
    ##bisection sorting for speed
    j = bisect(bkg,x)
    return bkg[j:].shape[0]/ bkg.shape[0]

def sigmoid(x, c = 3, a =.5):
    return 1 / (1 + np.exp(-c * (x - a)))

#Event ratio prior code
#size := max number
# size = 30
# CTR = 2

#TCT, TCC = np.zeros([size+1,size+1]),np.zeros([size+1,size+1])
#for i in range(1, size):
#    TCC[:,i] = poisson(i/CTR, np.linspace(0,size,size+1))
#    TCT[i,:] = poisson(i*CTR, np.linspace(0,size,size+1))
#TC = (TCT + TCC)
#for i in range(TC.shape[0]):
#    for j in range(TC.shape[0]):
#        TC[i,j] *= (j+i)
#TC[0,0] = 1e-20
#TC /= np.sum(TC)

####################################################################################

# TS LIBRARY

#Takes in a lower energy bound, power slope, and # between 0 and 1 and returns an energy
#Energy sampler created by math magic
def EventE(a,g,cdf):
    output = ((-g+1) * cdf/(((-a**(-g+1)) / (-g+1))**-1)) + a**(-g+1)
    return output**(1/(-g+1))

'''
Input: Energy in GeV, topology of neutrino (track = 1, casc = 0) , signal boolean (false = background atmospheric distribution)

Output: Percent of events that this kind of event is at the given energy
'''
#Given sig/bkg and topology, gives the percent of events of this kind at an energy
def PercentE(E, t_c, signal = True):
    perc = np.zeros_like(E)
    E = 10**E

    track_b = Efunc(E, *params[0])
    track_s = Efunc(E, *params[1])
    casc_b = Efunc(E, *params[2])
    casc_s = Efunc(E, *params[3])

    summed = (track_b + track_s + casc_b + casc_s)

    for i in range(t_c.shape[0]):
        if t_c[i]:

            if signal:
                perc[i] = casc_s[i]/ summed[i]
            else:
                perc[i] = casc_b[i]/ summed[i]
            pass

        elif not t_c[i]:
            if signal:
                perc[i] = track_s[i]/ summed[i]
            else:
                perc[i] = track_b[i]/ summed[i]
            pass

    return perc


# EVENT GENERATION
######################################################################

#Pulls using ow and can inject events. [Rob]
def gen(n_Ev, g, topo = 0, inra=None,indec=None):
        if(g<=0):
            print("g (second arg) must be >0, negative sign for spectra is hard-coded")
            return
        if topo == 0:
            mc = np.load("./mcdata/tracks_mc.npy")
        elif topo == 1:
            mc = np.load("./mcdata/cascade_mc.npy")
        else:
            print("topo = 0 for tracks, topo = 1 for cascades")
            return
        p=mc["ow"]*np.power(np.power(10,mc["logE"]),-g)
        p/=np.sum(p)
        keySC=np.random.choice( np.arange(len(p)), n_Ev, p=p, replace=False)
        evs=np.copy(mc[keySC])

        if(inra!=None and indec!=None):
            #Note: this method was yanked from a skylab example and might not actually be great
            eta = np.random.uniform(0., 2.*np.pi, n_Ev)
            sigmags=np.random.normal(scale=evs["angErr"])

            evs["dec"] = indec + np.sin(eta) * sigmags
            evs["ra"] = inra + np.cos(eta) * sigmags

            changeDecs=evs['dec']> np.pi/2
            #over shooting in dec is the same as rotating arounf and subtracting the Dec from pi.
            evs['ra'][changeDecs]+=np.pi #rotate the point to the other side
            evs['dec'][changeDecs]=np.pi-evs['dec'][changeDecs] #move the Dec accordingly

            #undershooting in dec
            changeDecs=evs['dec']< -np.pi/2

            evs['ra'][changeDecs]+=np.pi #rotate the point to the other side
            evs['dec'][changeDecs]=-np.pi-evs['dec'][changeDecs] #move the Dec accordingly

            #under or overshooting in ra, a bit easier
            evs['ra'][evs['ra']>2*np.pi]-=2*np.pi
            evs['ra'][evs['ra']<0]+=2*np.pi
        return evs


######################################################################

# METHODS FOR SIGNAL DETECTION
# FOR EACH METHOD TO FUNCTION PROPERLY WITH THE HYPOTHESIS TESTING SCRIPTS THEY MUST RETURN TS AS RETURN VALUE [0]

#CLASSIC LLH
def LLH(tracks,cascades, ra, dec, args):

    if args['delta_ang'] != 0:
        #only considers events within a delta_ang rad declination band around the location
        mask = np.logical_and(tracks["dec"] > dec - args['delta_ang'], tracks["dec"] < dec + args['delta_ang'])
        tracks = tracks[mask]
    evs = np.concatenate([tracks,cascades])
    nev = evs.shape[0]

    #in case the band is empty
    if nev == 0:
        return 0,0

    B = (1/(2*np.pi)) * args['B'](evs['sinDec']) * args['Eb'](evs['logE'])
    S = evPSFd([evs['ra'],evs['dec'],evs['angErr']], [ra,dec]) * args['Es'](evs['logE'])

    fun = lambda n, S, B: -np.sum(np.log( (((n/(S.shape[0]))*S) + ((1 - n/(S.shape[0]))*B))))
    opt = minimize(fun, nev/2, (S,B), bounds = ((0,nev),))

    n_sig = float(opt.x)
    maxllh = -float(opt.fun)
    TS = 2*(maxllh - np.sum(np.log(B)))

    return TS, n_sig

#VERSION 1 OF TOPOLOGY AWARE LIKELIHOOD METHOD
#Eventually to be changed into (32) from the following write-up from Hans: https://www.overleaf.com/project/60d3ba3f4cd5cecf7e328b37
def TA(tracks,cascades, ra, dec, args):

    if args['delta_ang'] != 0:
        #only considers events within a delta_ang rad declination band around the location
        mask = np.logical_and(tracks["dec"] > dec - args['delta_ang'], tracks["dec"] < dec + args['delta_ang'])
        tracks = tracks[mask]

    evs = np.concatenate([tracks, cascades])
    nev = evs.shape[0]

    #computes track and cascade signal and background terms to be used in a combined LLH search
    #For now background probabilities are fixed in the method code
    track_B = (1/(2*np.pi)) * args['Bt'](tracks['sinDec']) * args['Ebt'](tracks['logE']) * 0.884
    casc_B = (1/(2*np.pi)) * args['Bc'](cascades['sinDec']) * args['Ebc'](cascades['logE']) * 0.116
    B = np.concatenate([track_B, casc_B])

    #For now signal~E^-2 is fixed
    track_S = evPSFd([tracks['ra'],tracks['dec'],tracks['angErr']], [ra,dec]) * args['Est'](tracks['logE']) * args['Tau'](2)
    casc_S = evPSFd([cascades['ra'],cascades['dec'],cascades['angErr']], [ra,dec]) * args['Esc'](cascades['logE']) * (1-args['Tau'](2))
    S = np.concatenate([track_S, casc_S])

    fun = lambda n, S, B: -np.sum(np.log( (((n/(S.shape[0]))*S) + ((1 - n/(S.shape[0]))*B))))
    opt = minimize(fun, nev/2, (S,B), bounds = ((0,nev),))

    n_sig = float(opt.x)
    maxllh = -float(opt.fun)
    TS = 2*(maxllh - np.sum(np.log(B)))

    return TS, n_sig

# DIFFERENT VARIENT ON CLASSIC LLH (DESCRIBED IN AN OLD POWERPOINT IN MSU ICECUBE DRIVE)
def RLLH(tracks,cascades,ra,dec, args):
    evs = np.concatenate([tracks,cascades])

    S = evPSFd([evs['ra'],evs['dec'],evs['angErr']], [ra,dec])
    B = (1/(2*np.pi)) * args['B'](evs['dec'])

    alpha = S > B
    ns = np.sum(alpha)
    S = S[alpha]

    TS = 2*np.sum(np.log(S/B))

    return TS, ns

# ROB'S MULTIMAP METHOD WITH Energy
def MM(tracks, cascades, ra, dec, args):
    St =  evPSFd([tracks['ra'],tracks['dec'],tracks['angErr']], [ra,dec]) * args['Est'](tracks['logE'])
    Sc = evPSFd([cascades['ra'],cascades['dec'],cascades['angErr']], [ra,dec]) * args['Esc'](cascades['logE'])
    TS = (np.sum(St)/tracks.shape[0]) * (np.sum(Sc) / cascades.shape[0])
    return TS,

def LLH_detector0(evs, ra, dec, args):
    nev = evs.shape[0]
    ns = np.arange(0,nev)
    B = (1/(2*np.pi)) * args['B'](evs['sinDec']) * args['Eb'](evs['logE'])

    S = evPSFd([evs['ra'],evs['dec'],evs['angErr']], [ra,dec]) * args['Es'](evs['logE'])
    nfit, sfit = np.meshgrid(ns, S)
    nfit, bfit = np.meshgrid(ns, B)

    lsky = np.log( (nfit/(nev))*sfit + (1 - nfit/(nev))*bfit )
    injected = (np.argmax(np.sum(lsky,axis=0)))
    maxllh = np.max(np.sum(lsky,axis=0))

    TS = 2*(maxllh - nev*np.log(B))

    return TS, injected, maxllh, lsky

def TAPrior(tracks, cascades, ra, dec, args):
    try:
        TC = args['Prior']
    except:
        print("Prior array missing from args!")
        return

    evs = np.concatenate([tracks,cascades])
    nev = evs.shape[0]

    #in case the band is empty
    if nev == 0:
        return 0,0

    #computes track and cascade signal and background terms to be used in a combined LLH search
    track_B = (1/(2*np.pi))  * args['Ebt'](tracks['logE']).astype('float128') * 0.884 * (args['Bt'](tracks['sinDec'])).astype('float128')
    casc_B = (1/(2*np.pi))   * args['Ebc'](cascades['logE']).astype('float128') * 0.116 * (args['Bc'](cascades['sinDec'])).astype('float128')
    B = np.concatenate([track_B, casc_B])

    track_S = evPSFd([tracks['ra'],tracks['dec'],tracks['angErr']], [ra,dec]).astype('float128') * args['Est'](tracks['logE']).astype('float128') * 0.29
    casc_S = evPSFd([cascades['ra'],cascades['dec'],cascades['angErr']], [ra,dec]).astype('float128') * args['Esc'](cascades['logE']).astype('float128') * 0.71
    S = np.concatenate([track_S, casc_S])

    # LLH calculations/maximizations
    fun = lambda n, S, B: -np.sum(np.log( ((n/(S.shape[0]))*S) + ((1 - n/(S.shape[0]))*B) ))
    opt = minimize(fun, nev/2, (S,B), bounds = ((0,None),))
    n_sig = float(opt.x)
    b = np.ones(1).astype('float128') * opt.fun
    maxllh = np.exp(-b)

    # null likelihood
    L0 = np.exp(np.sum(np.log(B)))

    # prior calculation
    if args['delta_ang'] != 0:
        if ntrack == 0:
            nst = 0
    else:
        nst = LLH_detector0(tracks, ra, dec, args = args)[1]
    nsc = LLH_detector0(cascades, ra, dec, args = args)[1]
    prior = TC[nst,nsc].astype('float128')

    alpha = 1-(1/prior)

    return (np.log(maxllh) - np.log(maxllh - alpha*L0))


class multi_tester():

    '''
    methods: [list] Consist of method names as strings (Written exactly as they appear in package)
    tracks: [int] # background tracks
    cascades: [int] # background cascades
    resolution: [int] Healpy grid resolution (NPIX = 2**resolution)

    args: [dict] Used to pass information to the methods. Keys are method specific strings (ex: 'Prior' to pass in a TC prior).
                Values vary depending on method-- use this dict to pass any arguments to methods. Can be accessed by indexing the tester object as a dict.
                List of args can be found in the README

    Takes in the above arguments, checks to make sure they're of the right form, then creates pdfs from MC and initializes the object
    '''
    def __init__(self, methods, tracks, cascades, resolution = 8, args = dict()):

        if type(args) != dict:
            raise ValueError("args should be a dictionary of values to pass to the method functions.")

        #test of user inputted bands
        #this test allows for discontinuous bands, but checks to make sure there is no overlap between declination bands
        for band in dec_bands:
            assert(band[0] >= -np.pi/2 and band[1] <= np.pi/2), "Declination is defined within [-pi/2,pi/2]."
            for testband in dec_bands:
                #tests the bands for any overlap
                if np.all(band != testband):
                    assert(not (band[0] > testband[0] and band[0] < testband[1])), "Your declination bands either overlap or are in an unsupported format."
                    assert(not (band[1] > testband[0] and band[1] < testband[1])), "Your declination bands either overlap or are in an unsupported format."
        dec_bands = (np.array(dec_bands))
        self.dec_bands = dec_bands

        #list of method names exactly as they are written in the package (ex "LLH_detector")
        self.Methods = methods

        print("Initialization in progress...")
        #if no band cut is specified then full sky is used
        if 'delta_ang' not in args:
            args['delta_ang'] = 0

        if 'dec_bands' not in args:
            args['dec_bands'] = np.arcsin(np.column_stack([np.arange(-1,1,.2),np.arange(-.8,1.2,.2)]))

        self.args = args

        #number of mc background tracks and cascades to create background TS for and test with
        #will be consistent for everything this object is used for
        self.track_count = tracks
        self.cascade_count = cascades

        #name data
        self.timetag = ("".join(filter(str.isdigit, str(datetime.datetime.now()))))
        self.pkl = 'multitester_' + self.timetag + '.pkl'
        self.name = "multitester_" + self.timetag

        #resolution of healpy grid
        self.NSIDE = 2**resolution
        self.NPIX = hp.nside2npix(self.NSIDE)

        self.create_pdfs()

        #creates the data directory for this test
        os.mkdir('./data/' + self.name)
        os.mkdir('./plots/' + self.name)

        print('Initialization complete.')
        print()
        return

    #allow the tester to be indexed directly rather than having to call tester.args
    #args acts as a container for the testers' objects
    def __getitem__(self, key):
        return self.args[key]

    def __setitem__(self, key, val):
        self.args[key] = val

    def __repr__(self):
        return f'Multi-tester object using the following methods: {self.Methods}\nBackground tracks: {self.track_count} Background Cascades: {self.cascade_count}\nWill be stored in the file "{self.pkl}" with tester name "{self.name}"'

    '''
    bkg_trials: [int] Total number of trials that will make up the background distribution
    filecount: [int] Number of files to split bkg creation into
    runtime: [str (hr:min:sec)] Time allocation for each file
    mem: [str (ex '50GB')] Memory allocation for each file

    ra,dec: [float (rad)] ra and dec to build background distribution at and inject source at for signal trials
    signal_trials: [int] Number of signal trials to calculate significances for
    ninj_tracks, ninj_cascades: [int] Number of each topology to inject as a source for signal bkg_trials
    outpath: [str] Path to store data in (Default ./results/)
    clean: [bool] Toggles whether or not the middle files are deleted after the data is repackaged
    dryrun: [bool] Toggles whether or not the files are automatically run or not

    If signal trials are specified creates the signal and background TS distributions and tests signal trials
        in order to calculate significances.
    Without an ra,dec specified, will generate background and signal trials randomly within the declination bands
        and compare to other events within each band.
    Creates background TS, signal TS, and significances files in the specified outpath
    '''
    def run(self, bkg_trials, filecount, runtime, mem = '50G', ra = None, dec = None, signal_trials = 0, ninj_tracks = 0, ninj_cascades = 0, clean = True, dryrun = False):

        self.signal_trials = signal_trials
        self.ra = ra
        self.dec = dec

        if type(runtime) != type(''):
            raise ValueError('Input runtime as a string of the form hr:min:seconds (ex 5:00:00 for 5 hours)')
            return

        if(ninj_tracks+ninj_cascades > 0 and signal_trials == 0):
            raise ValueError(f'{ninj_tracks} track events and {ninj_cascades} cascade events injected, but no signal trial count specified.')
            return
        if(signal_trials > 0 and ninj_tracks+ninj_cascades == 0):
            raise ValueError(f'Signal trial count specified, but no events injected')
            return

        #number of events per job
        self.filecount = filecount
        self.nper= bkg_trials//filecount
        #write bkg sbatchs
        for i in range(filecount):
            filstr=self.name+"_"+str(i)+".sb"
            f = open('./templates/bkgjob.txt', 'r')
            filetxt = f.read().format(time = runtime, mem = mem, tasknm = 'multitester_' + self.timetag, obj = self.pkl, tag = str(i))
            f.close()
            f = open('./working/' +filstr, 'w+')
            f.write(filetxt)
            f.close()

        #write repackage sbatch
        f = open('./templates/repack.txt', 'r')
        filetxt = f.read().format(tasknm = self.name + '_REPACK', obj = self.pkl)
        f.close()
        f = open('./working/repack_' + self.name + '.sb', 'w+')
        f.write(filetxt)
        f.close()

        #write clean script sbatch
        f = open('./templates/clean.txt', 'r')
        filetxt = f.read().format(tasknm = self.name + '_CLEAN', obj = self.pkl)
        f.close()
        f = open('./working/clean_' + self.name + '.sb', 'w+')
        f.write(filetxt)
        f.close()


        #case where width based significance calculation is triggered-- else can be done explicitly in a script after backround distribution is created
        if signal_trials:
            #a few checks
            if min([bkg_trials, filecount, signal_trials, ninj_tracks, ninj_cascades]) < 0:
                raise ValueError('Some inputted value is negative')
                return
            if ninj_tracks == 0 and ninj_cascades == 0:
                raise ValueError('Please input a number of tracks and cascades to inject for signal trials')
                return

            self.ninj_t = ninj_tracks
            self.ninj_c = ninj_cascades

            #write signal sbatch
            f = open('./templates/signaljob.txt', 'r')
            filetxt = f.read().format(time = runtime, mem = mem, tasknm = 'multitester_signal_' + self.name, obj = self.pkl)
            f.close()
            f = open('./working/signal_' + self.name + '.sb', 'w+')
            f.write(filetxt)
            f.close()

            #write signal summation sbatch
            f = open('./templates/signal_summation.txt', 'r')
            filetxt = f.read().format(time = runtime, mem = mem, tasknm = self.name + '_SIGNAL', obj = self.pkl)
            f.close()
            f = open('./working/sigsum_' + self.name + '.sb', 'w+')
            f.write(filetxt)
            f.close()

        #WRITE SDAG-----------
        #str of jobs and their sdag names
        joblist = ''
        #str of each bkgTS dist creating job for calling as parents/children
        jstr = ''

        for i in range(filecount):
            joblist += f'JOB J{i} ./working/{self.name}_{i}.sb\n'
            jstr += f'J{i} '
        joblist += f'JOB R ./working/repack_{self.name}.sb\n'
        if clean:
            joblist += f'JOB CLEAN ./working/clean_{self.name}.sb\n'

        if signal_trials:
            joblist += f'JOB S ./working/signal_{self.name}.sb\n'
            joblist += f'JOB SS ./working/sigsum_{self.name}.sb\n'

            contents = joblist + '\nPARENT S CHILD ' + jstr + '\nPARENT ' + jstr + 'CHILD R SS'
            if clean:
                contents += '\nPARENT R SS CHILD CLEAN'
        else:
            contents = joblist + '\nPARENT ' + jstr + 'CHILD R'
            if clean:
                contents += '\nPARENT R CHILD CLEAN'

        fout=open(f'./working/{self.name}.sdag',"w")
        fout.write(contents)
        fout.close()

        bkg_filename = self.name + "_BKG.npz"
        signal_filename = self.name  + "_SIGNAL.npz"

        self.bkg = bkg_filename
        self.signal = signal_filename

        #saves this object in a pkl file to be read in by other scripts
        dbfile = open('./Testers/' + self.pkl , 'wb+')
        pickle.dump(self, dbfile)
        dbfile.close()

        #start the program
        if not dryrun:
            os.system(f'python2 sdag.py ./working/{self.name}.sdag')

        return


    '''
    signal: [bool] Determines whether or not to read in a signal TS distribution too
            *Only let signal be True if signal generation was done in run() as well*

    Run this after the BKG creation process is finished.
    Loads in the background TS distribution created with run() into self.bkg and splits bkg into dec bands for hypothesis testing
    Saves the TS file into tester.bkg; Null distribution in tester.bkg['TS']
    '''
    def load_TS(self, signal):
        try:
            bkg_dist = np.load("./data/" + self.name + '/' + self.bkg)
            print("Background file successfully loaded into .bkg")
            self.bkg = bkg_dist

            print("Dividing background into declination bands...")
            #divides TS distribution into declination bands based on self.dec_bands for hypothesis testing
            self.bkg_bands = []
            for band in range(self.dec_bands.shape[0]):
                self.bkg_bands.append(np.column_stack([(self.bkg['TS'][:,j][np.logical_and(self.bkg['dec'][:,j] >= self.dec_bands[band][0],self.bkg['dec'][:,j] < self.dec_bands[band][1])]) for j in range(len(self.Methods))]))

            if signal:
                signal_dist = np.load("./data/"+ self.name + '/' + self.signal)
                print("Signal file successfully loaded into .bkg")
                self.signal = signal_dist

            print("Data loading complete.")
            return

        except:
            raise EnvironmentError('Are your TS files missing/ created yet?')


    """
    ra: [float] Right ascension to test the methods at[0, 2pi]
    dec: [float] Declination to test the methods at [pi/2, pi/2]
    ninj_t, ninj_c: [int] Number of tracks and cascades to inject as a source at the given ra, dec

    Creates a MC sky according to specs in object and evaluates each method at a given point in the sky
    If an event is injected, always tests on injection ra and declination
    Returns: Array with TS for each method
    """
    def test_methods(self, ra, dec, ninj_t = 0, ninj_c = 0):
        tracks = np.concatenate([gen(ninj_t, 2, 0, inra = ra, indec = dec),gen(self.track_count, 3.7, 0)])
        cascades = np.concatenate([gen(ninj_c, 2, 1, inra = ra, indec = dec),gen(self.cascade_count, 3.7, 1)])
        output = np.zeros(len(self.Methods))
        for i,method in enumerate(self.Methods):
            output[i] = eval(method + '(tracks, cascades, ra, dec, args = self.args)')[0]
        return output

    '''
    ninj_t, ninj_c: [int] Number of tracks and cascades to inject as a source at the given inj_ra, inj_dec
    inj_ra, inj_dec: [float] Right ascension/ Declination to inject the source at

    Creates a healpy sky with the object's track, cascade number and optional injection. Tests each method at every
        point in the sky to give a visual for the methods' performances.
    TS sky array is saved in self.sky and can be shown using show_sky
    '''
    def create_sky(self, ninj_t = 0, ninj_c = 0, inj_ra = np.random.uniform(0,2*np.pi), inj_dec = np.random.uniform(-np.pi/2,np.pi/2), bar = False):
        self.sky_ninj_t = ninj_t
        self.sky_ninj_c = ninj_c
        #angle array of every point on the sky
        m = np.arange(self.NPIX)
        theta, phi = np.deg2rad(hp.pix2ang(nside=self.NSIDE, ipix=m, lonlat = True))
        self.sky = np.zeros([self.NPIX, len(self.Methods)])

        tracks = np.concatenate([gen(ninj_t, 2, 0, inra = inj_ra, indec = inj_dec),gen(self.track_count, 3.7, 0)])
        cascades = np.concatenate([gen(ninj_c, 2, 1, inra = inj_ra, indec = inj_dec),gen(self.cascade_count, 3.7, 1)])

        if bar:
            for i in tqdm(range((self.NPIX))):
                for k, method in enumerate(self.Methods):
                    self.sky[i][k] = eval(method + '(tracks, cascades, theta[i], phi[i], args = self.args)')[0]
        else:
            for i in (range((self.NPIX))):
                for k, method in enumerate(self.Methods):
                    self.sky[i][k] = eval(method + '(tracks, cascades, theta[i], phi[i], args = self.args)')[0]

        return self.sky

    '''
    Saves a mollwide view of each method's TS sky in ./plots/name/
    '''
    def show_sky(self, inline = False):
        for i in range(len(self.Methods)):
            hp.mollview(self.sky[:,i])
            plt.title(f"TS Sky for {self.Methods[i]}- {self.track_count} track {self.cascade_count} casc background \n {self.sky_ninj_t}, {self.sky_ninj_c} injection")
            if inline:
                plt.show()
            else:
                plt.savefig('./plots/' + self.name + '/' + 'SKY_' + self.Methods[i])
        return

    """
    TS: [list/array] Test statistics for each method to calculate significances for in comparison to self.bkg
    dec: [list/array] Declinations of the given Test statistics

    Compares TS against a loaded background the calculate the significance level
    *Must have loaded in the background distribution*
    Returns: Significance array
    """
    #Not vectorized due to p-value using unvectorized bisect sort algo
    def calculate_sigma(self, TS, dec):
        TS = np.array(TS, dtype=float)
        dec = np.array(dec, dtype=float)
        #checks to make sure bkg is loaded into the object
        if type(self.bkg) == type(''):
            raise AttributeError("Load in your TS distributions first.")

        if TS.shape != dec.shape:
            raise ValueError("Makes sure your TS and declination arrays match!")

        #handles the single event case where you pass in [Method1 TS, Method2 TS, ...] and [Method1 dec, Method2 dec, ...]
        if len(TS.shape) < 2 and len(self.Methods) > 1:
            TS = TS.reshape(1,-1)
            dec = dec.reshape(1,-1)
        #handles the single method case where you pass in [TS1, TS2, ...] and [dec1, dec2, ...]
        elif len(TS.shape) < 2 and len(self.Methods) == 1:
            TS = TS.reshape(-1,1)
            dec = dec.reshape(-1,1)

        sigma = np.zeros_like(TS)
        for i in range(TS.shape[0]):
            for j in range(TS.shape[1]):
                #calculates the band number of the given declination
                band = np.argmax(np.array([np.logical_and(dec[i,j] >= band[0], dec[i,j] < band[1]) for band in self.dec_bands]))
                #calculates p_values by comparing TS to all bkg TS within the same declination band
                sigma[i,j] = p_value(TS[i,j], self.bkg_bands[band][:,j])

        return pd2sig(sigma)

    """
    Temporary (and bad) benchmarking function----
    Inputs: Test specifications
    Runs background trials on the HPCC (not yet in a job) for the method and prints out
    all the useful benchmarking info one would need to calculate the stuff for background files
    Returns: Total time to run N trials, Average time per trial, Array of each time per trial
    """
    def benchmarking(self, N, printout = False,  bar  = False):
        times = np.zeros(N)
        if bar:
            for i in tqdm(range(N)):
                t1 = datetime.datetime.now()
                self.test_methods(0,0)
                t2 = datetime.datetime.now()
                times[i] = (t2-t1).total_seconds()
        else:
            for i in range(N):
                t1 = datetime.datetime.now()
                self.test_methods(0,0)
                t2 = datetime.datetime.now()
                times[i] = (t2-t1).total_seconds()
        if printout:
            print(np.sum(times), np.mean(times), times)
        return np.sum(times), np.mean(times), times

    '''
    ntrials: [int] Number of trials to run at each injection combination
    dec: [float/list/array] Declination to test at in [-np.pi/2, np.pi/2] OR declination band to test within [min_dec, max_dec]
    size: [int] Max ninj_t and ninj_c to test at
    progress: [bool] Toggles whether progress is printed out
    plot: [bool] Toggles whether or not the TC space is plotted and saved in ./plots/{tester.name}/
    inline: [bool] If inline is True with plot == True then the plot will be shown using plt.show() instead of saved

    Tests each method at all injection combinations within [size,size] and returns an array of the percent of
        events that measure above 5sigma
    If plotted, returns a contour plot of the above data with a red line whose bottom line shows the discovery potential as a function of number of tracks and cascades
    '''
    def TC_space(self, ntrials, dec, size = 10, progress = False, plot = False, inline = False):

        space = np.zeros([size,size, len(self.Methods)])

        #TC space signifigance comparison
        if progress:
            for ninj_t in tqdm(range(size)):
                for ninj_c in range(size):

                    #if a band like [min, max] is passed in for declination, pulls uniformly from said band
                    if type(dec) == list or type(dec) == np.ndarray:
                        assert(np.array(dec).shape[0] == 2), "Pass in a declination band as [min, max]"
                        test_dec = np.random.uniform(*dec)
                    else:
                        test_dec = dec

                    TS = np.zeros([ntrials, len(self.Methods)])
                    TSdec = np.ones_like(TS)*test_dec
                    for i in range(ntrials):
                        ra = np.random.uniform(0, 2*np.pi)
                        TS[i] = self.test_methods(ra, test_dec, ninj_t, ninj_c)

                    sigs = self.calculate_sigma(TS, TSdec)
                    space[ninj_t, ninj_c] = np.sum([sigs >= 5], axis = 1)/ntrials
        else:
            for ninj_t in (range(size)):
                for ninj_c in range(size):

                    #if a band like [min, max] is passed in for declination, pulls uniformly from said band
                    if type(dec) == list or type(dec) == np.ndarray:
                        assert(np.array(dec).shape[0] == 2), "Pass in a declination band as [min, max]"
                        test_dec = np.random.uniform(*dec)
                    else:
                        test_dec = dec

                    TS = np.zeros([ntrials, len(self.Methods)])
                    TSdec = np.ones_like(TS)*test_dec
                    for i in range(ntrials):
                        ra = np.random.uniform(0, 2*np.pi)
                        TS[i] = self.test_methods(ra, test_dec, ninj_t, ninj_c)

                    sigs = self.calculate_sigma(TS, TSdec)
                    space[ninj_t, ninj_c] = np.sum([sigs >= 5], axis = 1)/ntrials

        if plot:
            for k in range(len(self.Methods)):
                TCset = plt.contourf(space[:,:,k], levels = np.linspace(0,1,21))
                plt.ylabel("Injected Tracks")
                plt.xlabel("Injected Cascades")
                plt.colorbar()
                try:
                    p1 = TCset.collections[10].get_paths()[0]
                    px,py= p1.vertices[:,0], p1.vertices[:,1]
                    plt.plot(px,py, color = 'red', label = f'{self.Methods[k]} Discovery Potential', linestyle = 'dashed')
                    plt.legend()
                except:
                    print(f"ERROR: 50% percentile missing from {self.Methods[k]}")

                plt.title(f"% of Injections Above 5sigma [{self.track_count},{self.cascade_count}] bkg: " + self.Methods[k])
                if not inline:
                    plt.savefig('./plots/' + self.name + '/' + 'TC_' + self.Methods[k])
                    plt.clf()
                else:
                    plt.show()

        return space

    '''
    Ran during initialization of a tester object to store pdfs in tester.args
    Loads in MC data to create Background Spatial pdfs and Energy pdfs for signal (-2 spectrum) and background (-3.7 spectrum)
    Creates pdfs for tracks and cascades separately for use in topology-implemented methods 
    '''
    def create_pdfs(self):
        #creates a large sample of MC signal and background events to determine pdfs from assuming a -2 spectrum for signal and -3.7 for background
        sig_t = gen(100000, 2, 0)
        bkg_t = gen(100000, 3.7, 0)
        sig_c = gen(100000, 2, 1)
        bkg_c = gen(100000, 3.7, 1)
        #for calculating topology ratio pdfs
        tracks_mc = np.load("./mcdata/tracks_mc.npy")
        cascs_mc = np.load("./mcdata/cascade_mc.npy")

        #makes sure any possible energy value falls within the range of interpolation
        E_x = np.linspace(min(sig_t['logE'].min(), sig_c['logE'].min(),bkg_t['logE'].min(),bkg_c['logE'].min()),
                      max(sig_t['logE'].max(), sig_c['logE'].max(),bkg_t['logE'].max(),bkg_c['logE'].max()), 1000)
        sindec_x = np.linspace(-1,1,1000)
        gamma_x = np.arange(1,5,.1)

        #Interpolation is MUCH faster to call than the scipy kde function, so we interpoalte over all kde fits for efficiency
        Bt = InterpolatedUnivariateSpline(sindec_x, (gaussian_kde(bkg_t['sinDec'])(sindec_x)), k = 3)
        Bc = InterpolatedUnivariateSpline(sindec_x, (gaussian_kde(bkg_c['sinDec'])(sindec_x)), k = 3)

        #Without any topology changes, we set the pdfs to all come from track events as you would in a normal PS search
        B = Bt
        print("Background spatial splines created")

        #signal and background energy pdfs
        #Signal track energy pdf
        Est = InterpolatedUnivariateSpline(E_x, (gaussian_kde(sig_t['logE'])(E_x)), k = 3)
        #Background track energy pdf
        Ebt = InterpolatedUnivariateSpline(E_x, (gaussian_kde(bkg_t['logE'])(E_x)), k = 3)

        #Signal cascade energy pdf
        Esc = InterpolatedUnivariateSpline(E_x, (gaussian_kde(sig_c['logE'])(E_x)), k = 3)
        #Background cascade energy pdf
        Ebc = InterpolatedUnivariateSpline(E_x, (gaussian_kde(bkg_c['logE'])(E_x)), k = 3)

        #Everything treated as a track for LLH
        #Energy background pdf- E^-3.7 spectrum
        Es = Est
        #Energy background pdf- (E^-2 spectrum fixed for now-- eventually this has to be 3D (gamma, E, declination)
        Eb = Ebt
        print("Energy splines created")

        percs = []

        #calculates the source topology ratio for every spectral index between 1 and 5
        for g in gamma_x:
            wc = np.power(cascs_mc['trueE'], -g) * cascs_mc['ow']
            wt = np.power(tracks_mc['trueE'], -g) * tracks_mc['ow']
            perc_casc = np.sum(wc)/(np.sum(wt)+np.sum(wc))
            percs.append(perc_casc)
        #*cascade* source percent contribution
        #(Use 1-Tau for singal track contribution)
        Tau = InterpolatedUnivariateSpline(gamma_x, percs,  k = 3)

        print("Topology contribution spline created")

        #fills in tester with splines for energy, background spatial term, and topology for both split topology and non split topology searches
        self.args['Bt'] = Bt
        self.args['Bc'] = Bc
        self.args['Est'] = Est
        self.args['Ebt'] = Ebt
        self.args['Esc'] = Esc
        self.args['Ebc'] = Ebc
        self.args['B'] = B
        self.args['Es'] = Es
        self.args['Eb'] = Eb
        self.args['Tau'] = Tau
