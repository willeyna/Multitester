from package import *

tester = pickle.load(open('./Testers/' + sys.argv[1],'rb'))

signal_TS = np.zeros([tester.nper, len(tester.Methods)])

t1= datetime.datetime.now()
for i in (np.arange(self.signal_trials)):
    signal_TS[i] = tester.test_methods(tester.ra, tester.dec, ninj_t = tester.ninj_t, ninj_c = tester.ninj_c, inj_ra = tester.inj_ra, inj_dec = tester.inj_dec)
t2= datetime.datetime.now()
dt=(t2-t1).total_seconds()

np.savez('./working/' + tester.name + "_SIGNAL.npz", TS = signal_TS, runtime = dt)
