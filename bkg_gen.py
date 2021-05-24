from package import *

tester = pickle.load(open('./Testers/' + sys.argv[1],'rb'))
tag = (sys.argv[2])

bkg_TS = np.zeros([tester.nper, len(tester.Methods)])
declinations = np.zeros(tester.nper)

t1= datetime.datetime.now()
for i in (np.arange(tester.nper)):
    ra,dec = np.random.uniform(0, 2*np.pi), np.random.choice([np.random.uniform(*band) for band in tester.dec_bands])
    bkg_TS[i] = tester.test_methods(ra, dec)
    declinations[i] = dec
t2= datetime.datetime.now()
dt=(t2-t1).total_seconds()

bkg_TS.sort(axis = 0);

if tester.signal_trials:
    #compare signal trials to this background sample
    signal = np.load('./data/'+tester.signal_name)
    signal_TS = signal['TS']
    sig_dec = signal['dec']

    #array that counts how many background trials are in the dec. bin of each signal trial
    nin_bin = np.zeros([tester.signal_trials, len(tester.Methods)])

    #counts the number of background events larger than each signal event in this distribution fragment
    psum = np.zeros([tester.signal_trials, len(tester.Methods)])
    for i in range(len(tester.Methods)):
        for j,ev in enumerate(signal_TS[:,i]):
            #pulls current signal event's declination
            dec = sig_dec[j]
            #generates the array of every background event in the declination band of the signal trial
            filt_bkg = bkg_TS[:,i][[np.logical_and(declinations >= band[0],declinations < band[1]) for band in tester.dec_bands if dec >= band[0] and dec < band[1]][0]]
            psum[j,i] = p_value(ev,filt_bkg)*filt_bkg.shape[0]

            nin_bin[j,i] = filt_bkg.shape[0]
    np.savez('./working/'+ tester.name + "_BKG" + tag + ".npz", TS = bkg_TS, dec = declinations, psum = psum, nin_bin = nin_bin, runtime = dt)
else:
    np.savez('./working/'+ tester.name + "_BKG" + tag + ".npz", TS = bkg_TS, dec = declinations, runtime = dt)
