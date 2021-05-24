from package import *

tester = pickle.load(open('./Testers/' + sys.argv[1],'rb'))

signal_TS = np.zeros([tester.signal_trials, len(tester.Methods)])
declinations = np.zeros(tester.signal_trials)

t1= datetime.datetime.now()
for i in (np.arange(tester.signal_trials)):
    ra,dec = np.random.uniform(0, 2*np.pi), np.random.choice([np.random.uniform(*band) for band in bands])
    signal_TS[i] = tester.test_methods(ra, dec, ninj_t = tester.ninj_t, ninj_c = tester.ninj_c)
    declinations[i] = dec
t2= datetime.datetime.now()
dt=(t2-t1).total_seconds()

np.savez('./data/' + tester.name + "_SIGNAL.npz", TS = signal_TS, dec = declinations, runtime = dt)
