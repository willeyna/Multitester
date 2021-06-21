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

#loads in parameters for an Energy function brought in from a plot in https://arxiv.org/pdf/1306.2309.pdf
params = np.array([[ 7.83668562e+13, -2.29461080e+00],
                   [ 4.24024293e+05, -7.25775174e-01],
                   [ 5.77647391e+12, -2.27582695e+00],
                   [ 9.97827439e+04, -5.66062064e-01]])

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
#Uses Kent distribution/ spehrical gaussian
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

#size := max number
size = 30
CTR = 2

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

    B = (1/(2*np.pi)) * np.exp(args['B'](evs['sinDec'])) * args['Eb'](evs['logE'])
    S = evPSFd([evs['ra'],evs['dec'],evs['angErr']], [ra,dec]) * args['Es'](evs['logE'])

    fun = lambda n, S, B: -np.sum(np.log( (((n/(S.shape[0]))*S) + ((1 - n/(S.shape[0]))*B))))
    opt = minimize(fun, nev/2, (S,B), bounds = ((0,nev),))

    n_sig = float(opt.x)
    maxllh = -float(opt.fun)
    TS = 2*(maxllh - np.sum(np.log(B)))

    return TS, n_sig

# WIP LLH with Energy and Dec pdf declination band dependence/splits
# def LLH_DD(tracks,cascades, ra, dec, args):
#
#     if args['delta_ang'] != 0:
#         #only considers events within a delta_ang rad declination band around the location
#         mask = np.logical_and(tracks["dec"] > dec - args['delta_ang'], tracks["dec"] < dec + args['delta_ang'])
#         tracks = tracks[mask]
#
#     evs = np.concatenate([tracks, cascades])
#     nev = evs.shape[0]
#
#     #gets an array of the band number each event is in
#     track_bands = np.argmax(np.array([np.logical_and(tracks['dec'] >= band[0],tracks['dec'] < band[1]) for band in tester.dec_bands]), axis = 0)
#     cascade_bands = np.argmax(np.array([np.logical_and(cascades['dec'] >= band[0],cascades['dec'] < band[1]) for band in tester.dec_bands]), axis = 0)
#
#     #energyless B,S terms
#     track_B = (1/(2*np.pi)) * args['Bt'](tracks['dec']) * 0.884
#     casc_B = (1/(2*np.pi)) * args['Bc'](cascades['dec']) * 0.116
#
#     #multiplies by dec dep energy pdf
#     track_B *= np.array([args['Ebt'][track_bands[i]](tracks[i]['logE']) for i in range(tracks.shape[0])])
#     casc_B *= np.array([args['Ebc'][cascade_bands[i]](cascades[i]['logE']) for i in range(cascades.shape[0])])
#     B = np.concatenate([track_B, casc_B])
#
#     track_S = evPSFd([tracks['ra'],tracks['dec'],tracks['angErr']], [ra,dec]) * 0.29
#     casc_S = evPSFd([cascades['ra'],cascades['dec'],cascades['angErr']], [ra,dec]) * 0.71
#
#     #multiplies by dec dep energy pdf
#     track_S *= np.array([args['Est'][track_bands[i]](tracks[i]['logE']) for i in range(tracks.shape[0])])
#     casc_S *= np.array([args['Esc'][cascade_bands[i]](cascades[i]['logE']) for i in range(cascades.shape[0])])
#     S = np.concatenate([track_S, casc_S])
#
#     fun = lambda n, S, B: -np.sum(np.log( (((n/(S.shape[0]))*S) + ((1 - n/(S.shape[0]))*B))))
#     opt = minimize(fun, nev/2, (S,B), bounds = ((0,nev),))
#
#     n_sig = float(opt.x)
#     maxllh = -float(opt.fun)
#     TS = 2*(maxllh - np.sum(np.log(B)))
#
#     return TS, n_sig

#P(tau | gamma)
def TA(tracks,cascades, ra, dec, args):

    if args['delta_ang'] != 0:
        #only considers events within a delta_ang rad declination band around the location
        mask = np.logical_and(tracks["dec"] > dec - args['delta_ang'], tracks["dec"] < dec + args['delta_ang'])
        tracks = tracks[mask]

    evs = np.concatenate([tracks, cascades])
    nev = evs.shape[0]

    #computes track and cascade signal and background terms to be used in a combined LLH search
    track_B = (1/(2*np.pi)) * np.exp(args['Bt'](tracks['sinDec'])) * args['Ebt'](tracks['logE']) * 0.884
    casc_B = (1/(2*np.pi)) * np.exp(args['Bc'](cascades['sinDec'])) * args['Ebc'](cascades['logE']) * 0.116
    B = np.concatenate([track_B, casc_B])

    track_S = evPSFd([tracks['ra'],tracks['dec'],tracks['angErr']], [ra,dec]) * args['Est'](tracks['logE']) * 0.29
    casc_S = evPSFd([cascades['ra'],cascades['dec'],cascades['angErr']], [ra,dec]) * args['Esc'](cascades['logE']) * 0.71
    S = np.concatenate([track_S, casc_S])

    fun = lambda n, S, B: -np.sum(np.log( (((n/(S.shape[0]))*S) + ((1 - n/(S.shape[0]))*B))))
    opt = minimize(fun, nev/2, (S,B), bounds = ((0,nev),))

    n_sig = float(opt.x)
    maxllh = -float(opt.fun)
    TS = 2*(maxllh - np.sum(np.log(B)))

    return TS, n_sig


def SMTopoAw(tracks, cascades, ra, dec, args):

    if "Sigmoid" in args:
        a,c = args['Sigmoid'][0], args['Sigmoid'][1]
    else:
        a,c = 0.5, 2.2

    if args['delta_ang'] != 0:
        #only considers events within a delta_ang rad declination band around the location
        mask = np.logical_and(tracks["dec"] > dec - args['delta_ang'], tracks["dec"] < dec + args['delta_ang'])
        tracks = tracks[mask]
    evs = np.concatenate([tracks,cascades])
    nev = evs.shape[0]

    #in case the band is empty
    if nev == 0:
        return 0,0

    fS = PercentE(evs['logE'],evs['topo'], True)
    fB = PercentE(evs['logE'],evs['topo'], False)

    S = evPSFd([evs['ra'],evs['dec'],evs['angErr']],[ra,dec]) * sigmoid(fS, a = 0.5, c = 2.2)

    B = (1/(2*np.pi)) * np.exp(args['bkg_spline'](evs['dec'])) * fB

    fun = lambda n, S, B: -np.sum(np.log( (((n/(S.shape[0]))*S) + ((1 - n/(S.shape[0]))*B))))
    opt = minimize(fun, nev/2, (S,B), bounds = ((0,None),))

    injected = float(opt.x)
    maxllh = -float(opt.fun)

    TS = 2*(maxllh - np.sum(np.log(B)))

    return TS, injected

def Cascade_Filter(tracks, cascades, ra, dec, args):

    if args['delta_ang'] != 0:
        #only considers events within a delta_ang rad declination band around the location
        mask = np.logical_and(tracks["dec"] > dec - args['delta_ang'], tracks["dec"] < dec + args['delta_ang'])
        tracks = tracks[mask]
    ntrack = tracks.shape[0]

    #in case the band is empty
    if ntrack == 0:
        return 0,0

    S =  evPSFd([tracks['ra'],tracks['dec'],tracks['angErr']],[ra,dec])
    B = (1/(2*np.pi)) * np.exp(args['bkg_spline'](tracks['dec'])) * fB

    fun = lambda n, S, B: -np.sum(np.log( ((n/(S.shape[0]))*S) + ((1 - n/(S.shape[0]))*B) ))
    opt = minimize(fun, ntrack/2, (S,B), bounds = ((0,None),))

    maxllh = -float(opt.fun)
    TS = 2*(maxllh - ntrack*np.log(B))
    #Applies a cascade prior to the traditional LLH TS for tracks
    PRIOR =  np.sum(evPSFd([cascades['ra'],cascades['dec'],cascades['angErr']],[ra,dec]))

    TS *= PRIOR

    return TS, PRIOR

# DIFFERENT VARIENT ON CLASSIC LLH (DESCRIBED IN AN OLD POWERPOINT IN MSU ICECUBE DRIVE)
def RLLH(tracks,cascades,ra,dec, args):
    evs = np.concatenate([tracks,cascades])

    S = evPSFd([evs['ra'],evs['dec'],evs['angErr']], [ra,dec])
    B = (1/(2*np.pi)) * np.exp(args['bkg_spline'](evs['dec']))

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

# TOPOLOGY RATIO PRIOR APPLIED
# THIS VERSION DOES NOT USE knn FOR  SIGNAL COUNT; USES LLH MAXIMIZER
def TCP(tracks, cascades, ra, dec, args):
    if 'Prior' in args:
        TC = args['Prior']

    nsc = int(round(LLH_detector0(cascades, ra, dec)[1]))
    nst = int(round(LLH_detector0(tracks, ra, dec)[1]))
    prior = TC[nst,nsc]

    TS0 = LLH_detector(tracks, cascades, ra, dec, args)[0]
    TS = TS0 * prior
    return TS, prior, [nst,nsc], TS0

def TruePrior(tracks, cascades, ra, dec, args):
    if 'Prior' in args:
        TC = args['Prior']

    if args['delta_ang'] != 0:
        #only considers events within a delta_ang rad declination band around the location
        mask = np.logical_and(tracks["dec"] > dec - args['delta_ang'], tracks["dec"] < dec + args['delta_ang'])
        tracks = tracks[mask]
        ntrack = tracks.shape[0]

    evs = np.concatenate([tracks,cascades])
    nev = evs.shape[0]

    #in case the band is empty
    if nev == 0:
        return 0,0

    # spatial bkg and signal terms
    B = (1/(2*np.pi)) * np.exp(args['bkg_spline'](evs['dec']))
    S = evPSFd([evs['ra'],evs['dec'],evs['angErr']], [ra,dec])

    # LLH calculations/maximizations
    fun = lambda n, S, B: -np.sum(np.log( ((n/(S.shape[0]))*S) + ((1 - n/(S.shape[0]))*B) ))
    opt = minimize(fun, nev/2, (S,B), bounds = ((0,None),))
    n_sig = float(opt.x)
    maxllh = -float(opt.fun)

    # null likelihood
    L0 = np.sum(np.log(B)).astype('float128')

    # prior calculation
    if ntrack == 0:
        nst = 0
    else:
        nst = LLH_detector0(tracks, ra, dec, args = args)[1]
    nsc = LLH_detector0(cascades, ra, dec, args = args)[1]
    prior = TC[nst,nsc].astype('float128')

    offset = np.log(np.exp(maxllh) + np.exp(L0)*((1/prior) - 1))

    return np.exp(maxllh - offset),

def LLH_detector0(evs, ra, dec, args):
    nev = evs.shape[0]
    ns = np.arange(0,nev)
    B = (1/(2*np.pi)) * np.exp(args['B'](evs['sinDec'])) * args['Eb'](evs['logE'])

    S = evPSFd([evs['ra'],evs['dec'],evs['angErr']], [ra,dec]) * args['Es'](evs['logE'])
    nfit, sfit = np.meshgrid(ns, S)
    nfit, bfit = np.meshgrid(ns, B)

    lsky = np.log( (nfit/(nev))*sfit + (1 - nfit/(nev))*bfit )
    injected = (np.argmax(np.sum(lsky,axis=0)))
    maxllh = np.max(np.sum(lsky,axis=0))

    TS = 2*(maxllh - nev*np.log(B))

    return TS, injected, maxllh, lsky

def LLH0(tracks, cascades, ra, dec, args):
    if args['delta_ang'] != 0:
        #only considers events within a delta_ang rad declination band around the location
        mask = np.logical_and(tracks["dec"] > dec - args['delta_ang'], tracks["dec"] < dec + args['delta_ang'])
        tracks = tracks[mask]
    evs = np.concatenate([tracks, cascades])
    nev = evs.shape[0]

    #in case the band is empty
    if nev == 0:
        return 0,0,0,0

    ns = np.arange(0,nev)
    B = (1/(2*np.pi)) * args['B'](evs['sinDec']) * args['Eb'](evs['logE'])

    S = evPSFd([evs['ra'],evs['dec'],evs['angErr']], [ra,dec]) * args['Es'](evs['logE'])

    nfit, sfit = np.meshgrid(ns, S)
    nfit, bfit = np.meshgrid(ns, B)

    lsky = np.log( (nfit/(nev))*sfit + (1 - nfit/(nev))*bfit )
    injected = (np.argmax(np.sum(lsky,axis=0)))
    maxllh = np.max(np.sum(lsky,axis=0))

    TS = 2*(maxllh - np.sum(np.log(B)))

    return TS, injected, lsky, nev

def TAPrior(tracks, cascades, ra, dec, args):
    try:
        TC = args['Prior']
    except:
        print("Prior array missing from args!")
        return

#     if args['delta_ang'] != 0:
#         #only considers events within a delta_ang rad declination band around the location
#         mask = np.logical_and(tracks["dec"] > dec - args['delta_ang'], tracks["dec"] < dec + args['delta_ang'])
#         tracks = tracks[mask]
#         ntrack = tracks.shape[0]

    evs = np.concatenate([tracks,cascades])
    nev = evs.shape[0]

    #in case the band is empty
    if nev == 0:
        return 0,0

    #computes track and cascade signal and background terms to be used in a combined LLH search
    track_B = (1/(2*np.pi))  * args['Ebt'](tracks['logE']).astype('float128') * 0.884 * np.exp(args['Bt'](tracks['sinDec'])).astype('float128')
    casc_B = (1/(2*np.pi))   * args['Ebc'](cascades['logE']).astype('float128') * 0.116 * np.exp(args['Bc'](cascades['sinDec'])).astype('float128')
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
    dec_bands: [list/array] Dec bands [min,max] for which to test events in (Cannot overlap) (Default 10 bins evenly spaced in sin dec)
                            Singal TS generated in these bands are compared only to others within their bands in calculating significances
    spline_bins: [int] Number of bins to use in creation dec. depedent bkg pdf (Used only in LLH functions)
    args: [dict] Used to pass information to the methods. Keys are method specific strings (ex: 'Prior' to pass in a TC prior).
                values vary depending on method.

    Takes in the above arguments, checks to make sure they're of the right form, and initializes the object
    '''
    def __init__(self, methods, tracks, cascades, resolution = 8, dec_bands = np.arcsin(np.column_stack([np.arange(-1,1,.2),np.arange(-.8,1.2,.2)])), spline_bins = 20, args = dict()):

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

        #makes sure declination bands are passed in args to any method that needs them
        if len({'LLH', 'SMTopoAw', "Cascade_Filter", "TCP", "TruePrior", "LLH0"}.intersection(methods)) != 0:
            if 'delta_ang' not in args:
                raise ValueError("Specify your desired band width for use in LLH-dependent functions")

        self.args = args
        self.args['dec_bands'] = dec_bands
        print("Initialization in progress...")

        #methods that do not rely on any topological information have split = False
        if len({'LLH', 'LLH0', "TruePrior", 'TAPrior'}.intersection(methods)) != 0:
            self.create_pdfs(split = False)

        #methods that want split topology pdfs
        if len({'TA', 'MM'}.intersection(methods)) != 0:
            self.create_pdfs(split = True)


        #number of mc background tracks and cascades to create background TS for and test with
        #will be consistent for everything this object is used for
        self.track_count = tracks
        self.cascade_count = cascades

        #resolution of healpy grid
        self.NSIDE = 2**resolution
        self.NPIX = hp.nside2npix(self.NSIDE)
        #status on whether the program has been run for this object
        self.ran = False
        self.name = "_".join(self.Methods)

        print('Initialization complete.')
        return

    def __repr__(self):
        if self.ran:
            desc = f'''Multi-tester object using the following methods: {self.Methods}
Background tracks: {self.track_count} Background Cascades: {self.cascade_count}
Ran with filename: {self.name}
            '''
            try:
              desc += f'testing an injection of {self.ninj_t} tracks and {self.ninj_c} cascades.'
            except:
              pass
        else:
            desc = f'''Multi-tester object using the following methods: {self.Methods}
Background tracks: {self.track_count} Background Cascades: {self.cascade_count}
            '''
        return desc

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

        self.timetag = ("".join(filter(str.isdigit, str(datetime.datetime.now()))))
        self.pkl = 'multitester_' + self.timetag + '.pkl'

        #number of events per job
        self.filecount = filecount
        self.nper= bkg_trials//filecount
        #write bkg sbatchs
        filnam="multitester_" + self.timetag
        self.name = filnam
        for i in range(filecount):
            filstr=filnam+"_"+str(i)+".sb"
            f = open('./templates/bkgjob.txt', 'r')
            filetxt = f.read().format(time = runtime, mem = mem, tasknm = 'multitester_' + self.timetag, obj = self.pkl, tag = str(i))
            f.close()
            f = open('./working/' +filstr, 'w+')
            f.write(filetxt)
            f.close()

        #write repackage sbatch
        f = open('./templates/repack.txt', 'r')
        filetxt = f.read().format(tasknm = 'multitester_' + self.timetag + '_REPACK', obj = self.pkl)
        f.close()
        f = open('./working/repack_' + filnam + '.sb', 'w+')
        f.write(filetxt)
        f.close()

        #write clean script sbatch
        f = open('./templates/clean.txt', 'r')
        filetxt = f.read().format(tasknm = 'multitester_' + self.timetag + '_CLEAN', obj = self.pkl)
        f.close()
        f = open('./working/clean_' + filnam + '.sb', 'w+')
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
            filetxt = f.read().format(time = runtime, mem = mem, tasknm = 'multitester_signal_' + self.timetag, obj = self.pkl)
            f.close()
            f = open('./working/signal_' + filnam + '.sb', 'w+')
            f.write(filetxt)
            f.close()

            #write signal summation sbatch
            f = open('./templates/signal_summation.txt', 'r')
            filetxt = f.read().format(time = runtime, mem = mem, tasknm = 'multitester_' + self.timetag + '_SIGNAL', obj = self.pkl)
            f.close()
            f = open('./working/sigsum_' + filnam + '.sb', 'w+')
            f.write(filetxt)
            f.close()

        #WRITE SDAG-----------
        #str of jobs and their sdag names
        joblist = ''
        #str of each bkgTS dist creating job for calling as parents/children
        jstr = ''

        for i in range(filecount):
            joblist += f'JOB J{i} ./working/{filnam}_{i}.sb\n'
            jstr += f'J{i} '
        joblist += f'JOB R ./working/repack_{filnam}.sb\n'
        if clean:
            joblist += f'JOB CLEAN ./working/clean_{filnam}.sb\n'

        if signal_trials:
            joblist += f'JOB S ./working/signal_{filnam}.sb\n'
            joblist += f'JOB SS ./working/sigsum_{filnam}.sb\n'

            contents = joblist + '\nPARENT S CHILD ' + jstr + '\nPARENT ' + jstr + 'CHILD R SS'
            if clean:
                contents += '\nPARENT R SS CHILD CLEAN'
        else:
            contents = joblist + '\nPARENT ' + jstr + 'CHILD R'
            if clean:
                contents += '\nPARENT R CHILD CLEAN'

        fout=open(f'./working/{filnam}.sdag',"w")
        fout.write(contents)
        fout.close()

        bkg_filename = 'multitester_' + self.timetag + "_BKG.npz"
        signal_filename = 'multitester_' + self.timetag + "_SIGNAL.npz"

        self.bkg = bkg_filename
        self.signal = signal_filename
        self.ran = True

        #creates the data directory for this test
        os.mkdir('./data/' + self.name)

        #saves this object in a pkl file to be read in by other scripts
        dbfile = open('./Testers/multitester_' + self.timetag + '.pkl', 'wb+')
        pickle.dump(self, dbfile)
        dbfile.close()

        #start the program
        if not dryrun:
            os.system(f'python2 sdag.py ./working/{filnam}.sdag')

        return


    '''
    signal: [bool] Determines whether or not to read in a signal TS distribution too
            Only let signal be True if signal generation was done in run() as well

    Run this after the BKG creation process is finished.
    Loads in the background TS distribution created with run() into self.bkg
    TS can then be found with self.bkg['TS']
    '''
    def load_TS(self, signal):
        try:
            bkg_dist = np.load("./data/" + self.name + '/' + self.bkg)
            print("Background file successfully loaded into .bkg")
            self.bkg = bkg_dist

            if signal:
                signal_dist = np.load("./data/"+ self.name + '/' + self.signal)
                print("Signal file successfully loaded into .bkg")
                self.signal = signal_dist

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
    Saves a mollwide view of each method's TS sky in ./plots
    '''
    def show_sky(self, inline = False):
        for i in range(self.sky.shape[1]):
            hp.mollview(self.sky[:,i])
            if inline:
                plt.show()
            else:
                plt.savefig('./plots/SKY' + self.name+ self.Methods[i])
        return

    """
    TS: [list/array] Test statistics for each method to calculate significances for in comparison to self.bkg
    dec: [list/array] Declinations of the given Test statistics

    Compares TS against a loaded background the calculate the significance level
    *Must have loaded in the background distribution*
    Returns: Significance array
    """
    #Not vectorized
    def calculate_sigma(self, TS, dec):
        TS = np.array(TS).reshape(-1)

        if type(self.bkg) == type(''):
            raise AttributeError("Load in your TS distributions first.")
        sigma = np.zeros(len(self.Methods))

        band = np.argmax(np.array([np.logical_and(dec >= band[0],dec < band[1]) for band in self.dec_bands]))
        for j in range(len(self.Methods)):
            #generates the array of every background event in the declination band of the signal trial
            band_bkg = self.bkg['TS'][:,j][np.logical_and(self.bkg['dec'][:,j] >= self.dec_bands[band][0],self.bkg['dec'][:,j] < self.dec_bands[band][1])]
            sigma[j] = p_value(TS[j],band_bkg)

        return pd2sig(sigma)

    """
    WIP Benchmarking function----
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
    plt: [bool] Toggles whether or not the TC space is plotted and saved

    Tests each method at all injection combinations within [size,size] and returns an array of the percent of
        events that measure above 5sigma
    '''
    def TC_space(self, ntrials, dec, size = 10, progress = False, plot = False):

        space = np.zeros([size,size, len(self.Methods)])

        #TC space signifigance comparison
        for ninj_t in range(size):
            for ninj_c in range(size):

                #if a band like [min, max] is passed in for declination, pulls uniformly from said band
                if type(dec) == list or type(dec) == np.ndarray:
                    assert(np.array(dec).shape[0] == 2), "Pass in a declination band as [min, max]"
                    test_dec = np.random.uniform(*dec)
                else:
                    test_dec = dec

                sigs = np.zeros([ntrials, len(self.Methods)])
                for i in range(ntrials):
                    ra = np.random.uniform(0, 2*np.pi)
                    TS = self.test_methods(ra, test_dec, ninj_t, ninj_c)
                    sigs[i] = self.calculate_sigma(TS, test_dec)
                space[ninj_t, ninj_c] = np.sum([sigs >= 5], axis = 1)/ntrials

                if progress:
                    print(f"{ninj_t}.{int(10*ninj_c/size)}/{size}", end = '\r')
        if plot:
            for k in range(len(self.Methods)):
                plt.contourf(space[:,:,k], origin = 'lower')
                plt.xlabel("Injected Tracks")
                plt.ylabel("Injected Cascades")
                plt.colorbar()
                plt.title("Sensitivity for Topologically Varied Injections: " + self.Methods[k])
                plt.savefig('./plots/TC_SPACE_' + self.name + '_' + self.Methods[k])

        return space

    '''
    Reads in the mcdata files and uses a gaussian kernel density estimator to make a  pdf for declination taking into
        account the declination dependence of events in your sample
    Returns a pdf that can be evaluated over [-pi/2,pi/2]
    ## Later alter to create splines for both topologies
    '''
    def create_pdfs(self, split = False):
        #creates a large sample of MC signal and background events to determine pdfs from
        sig_t = gen(100000, 2, 0)
        bkg_t = gen(100000, 3.7, 0)
        sig_c = gen(100000, 2, 1)
        bkg_c = gen(100000, 3.7, 1)

        #makes sure any possible energy value falls within the range of interpolation
        E_x = np.linspace(min(sig_t['logE'].min(), sig_c['logE'].min(),bkg_t['logE'].min(),bkg_c['logE'].min()),
                        max(sig_t['logE'].max(), sig_c['logE'].max(),bkg_t['logE'].max(),bkg_c['logE'].max()), 1000)

        #if splitting pdfs by topology for a topology aware analysis
        if split:

            #log in interpolation makes sure no pdf values are negative
            track_hist, bins = np.histogram(bkg_t['sinDec'], density=True, bins = 10)
            Bt = InterpolatedUnivariateSpline((bins[1:] + bins[:-1]) / 2., np.log(track_hist), k=3)

            casc_hist, bins = np.histogram(bkg_c['sinDec'], density=True, bins = 10)
            Bc = InterpolatedUnivariateSpline((bins[1:] + bins[:-1]) / 2., np.log(casc_hist), k=3)

            print("Split topology background spatial splines created")
            # if dec_dep:
            #
            #     #lists to store the splines for each dec band
            #     Est = []
            #     Ebt = []
            #     Esc = []
            #     Ebc = []
            #
            #     for band in self.dec_bands:
            #
            #         #signal and background energy pdfs
            #         #makes sure any possible energy value falls within the range of interpolation
            #         E_x = np.linspace(min(tracks['logE'].min(), cascs['logE'].min()), max(tracks['logE'].max(), cascs['logE'].max()), 1000)
            #
            #         mask = np.logical_and(sig_t['dec'] >= band[0], sig_t['dec'] < band[1])
            #         Est.append(InterpolatedUnivariateSpline(E_x, (gaussian_kde(sig_t['logE'][mask])(E_x)), k = 3))
            #
            #         mask = np.logical_and(bkg_t['dec'] >= band[0], bkg_t['dec'] < band[1])
            #         Ebt.append(InterpolatedUnivariateSpline(E_x, (gaussian_kde(bkg_t['logE'][mask])(E_x)), k = 3))
            #
            #         mask = np.logical_and(sig_c['dec'] >= band[0], sig_c['dec'] < band[1])
            #         Esc.append(InterpolatedUnivariateSpline(E_x, (gaussian_kde(sig_c['logE'][mask])(E_x)), k = 3))
            #
            #         mask = np.logical_and(bkg_c['dec'] >= band[0], bkg_c['dec'] < band[1])
            #         Ebc.append(InterpolatedUnivariateSpline(E_x, (gaussian_kde(bkg_c['logE'][mask])(E_x)), k = 3))
            #
            #     print("Energy splines created")

            #signal and background energy pdfs
            Est = InterpolatedUnivariateSpline(E_x, (gaussian_kde(sig_t['logE'])(E_x)), k = 3)
            Ebt = InterpolatedUnivariateSpline(E_x, (gaussian_kde(bkg_t['logE'])(E_x)), k = 3)

            Esc = InterpolatedUnivariateSpline(E_x, (gaussian_kde(sig_c['logE'])(E_x)), k = 3)
            Ebc = InterpolatedUnivariateSpline(E_x, (gaussian_kde(bkg_c['logE'])(E_x)), k = 3)
            print("Split topology energy splines created")

            self.args['Bt'] = Bt
            self.args['Bc'] = Bc
            self.args['Est'] = Est
            self.args['Ebt'] = Ebt
            self.args['Esc'] = Esc
            self.args['Ebc'] = Ebc

        else:
            signal = np.concatenate([sig_t, sig_c])
            background = np.concatenate([bkg_t, bkg_c])

            #background pdf creation based on dec dist
            sin_decs = np.concatenate([bkg_t['sinDec'], bkg_c['sinDec']])

            #log in interpolation makes sure no pdf values are negative
            hist, bins = np.histogram(sin_decs, density=True, bins = 10)
            B = InterpolatedUnivariateSpline((bins[1:] + bins[:-1]) / 2., np.log(hist), k=3)

            print("Background spatial spline created")

            #Interpolation is MUCH faster to call than the scipy kde function, so we interpoalte over the kde
            #signal and background energy pdfs
            Es = InterpolatedUnivariateSpline(E_x, (gaussian_kde(signal['logE'])(E_x)), k = 3)
            Eb = InterpolatedUnivariateSpline(E_x, (gaussian_kde(background['logE'])(E_x)), k = 3)

            print("Energy splines created")

            self.args['B'] = B
            self.args['Es'] = Es
            self.args['Eb'] = Eb
