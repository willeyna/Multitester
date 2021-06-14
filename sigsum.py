from package import *

tester = pickle.load(open('./Testers/' + sys.argv[1],'rb'))

signal = np.load('./data/'+tester.signal)

pv_sum = np.zeros([tester.signal_trials, len(tester.Methods)])
total_nin_bin = np.zeros([tester.signal_trials, len(tester.Methods)])
try:
    for i in range(tester.filecount):
        tag = str(i)
        data = np.load('./working/'+ tester.name + "_BKG" + tag + ".npz")
        pv_sum += data['psum']
        total_nin_bin += data['nin_bin']
except:
    raise EnvironmentError('Error reading in all background files. Did one break or run out of time?')

#array of p-values for every signal trial
p_value = pv_sum/total_nin_bin
sigma = pd2sig(p_value)

np.savez('./data/' + tester.name + '/' + tester.name + "_SIGMA.npz", sigma = sigma, dec = signal['dec'])
