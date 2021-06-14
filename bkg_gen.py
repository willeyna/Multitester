from package import *

tester = pickle.load(open('./Testers/' + sys.argv[1],'rb'))
tag = (sys.argv[2])

bkg_TS = np.zeros([tester.nper, len(tester.Methods)])
declinations = np.zeros(tester.nper)

t1= datetime.datetime.now()
for i in (np.arange(tester.nper)):
    if tester.ra is not None and tester.dec is not None:
        ra,dec = tester.ra,tester.dec
    else:
        ra,dec = np.random.uniform(0, 2*np.pi), np.random.choice([np.random.uniform(*band) for band in tester.dec_bands])
    bkg_TS[i] = tester.test_methods(ra, dec)
    declinations[i] = dec
t2= datetime.datetime.now()
dt=(t2-t1).total_seconds()

declinations = declinations[bkg_TS.argsort(axis=0)]
bkg_TS.sort(axis = 0);

if tester.signal_trials:
    #compare signal trials to this background sample
    signal = np.load('./data/'+ tester.name + '/' +tester.signal)
    signal_TS = signal['TS']
    sig_dec = signal['dec']

    #array that counts how many background trials are in the dec. bin of each signal trial
    nin_bin = np.zeros([tester.signal_trials, len(tester.Methods)])

    #counts the number of background events larger than each signal event in this distribution fragment
    psum = np.zeros([tester.signal_trials, len(tester.Methods)])

    band_masks = np.array([np.logical_and(declinations >= band[0],declinations < band[1]) for band in tester.dec_bands])
    #mask for signal declinations for every band-- used in finding which band each signal event is in
    signal_masks = np.array([np.logical_and(sig_dec>= band[0],sig_dec < band[1]) for band in tester.dec_bands])

    for i in range(tester.signal_trials):
        #chooses the right mask for the band of the chosen event
        mask = band_masks[np.argmax(signal_masks[:,i])]
        for j in range(len(tester.Methods)):
            #generates the array of every background event in the declination band of the signal trial
            filt_bkg = bkg_TS[:,j][mask[:,j]]
            psum[i,j] = p_value(signal_TS[i,j],filt_bkg)*filt_bkg.shape[0]

            nin_bin[i,j] = filt_bkg.shape[0]

    np.savez('./working/'+ tester.name + "_BKG" + tag + ".npz", TS = bkg_TS, dec = declinations, psum = psum, nin_bin = nin_bin, runtime = dt)
else:
    np.savez('./working/'+ tester.name + "_BKG" + tag + ".npz", TS = bkg_TS, dec = declinations, runtime = dt)
