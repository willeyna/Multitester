from package import *

tester = pickle.load(open('./Testers/' + sys.argv[1],'rb'))
tag = (sys.argv[2])

bkg_TS = np.zeros([tester.nper, len(tester.Methods)])

t1= datetime.datetime.now()
for i in (np.arange(tester.nper)):
    bkg_TS[i] = tester.test_methods(tester.ra, tester.dec)
t2= datetime.datetime.now()
dt=(t2-t1).total_seconds()

if tester.signal_trials: 
    #compare signal trials to this background sample
    signal_TS = np.load('./working/'+tester.name + "_SIGNAL.npz")
    #counts the number of background events larger than each signal event in this distribution fragment
    psum = np.array([p_value(ev,bkg_TS)*bkg_TS.shape[0] for ev in signal_TS])
    np.savez('./working/'+ tester.name + "_BKG" + tag + ".npz", TS = bkg_TS, psum = psum, runtime = dt)
else:
    np.savez('./working/'+ tester.name + "_BKG" + tag + ".npz", TS = bkg_TS, runtime = dt)