from package import *

tester = pickle.load(open('./Testers/' + sys.argv[1],'rb'))

pv_sum = np.zeros(tester.signal_trials)
try:
    for i in range(self.filecount):
        tag = str(i)
        data = np.load('./working/'+ tester.name + "_BKG" + tag + ".npz")
        pv_sum += data['psum']
except:
    raise EnvironmentError('Error reading in all background files. Did one break or run out of time?')
    
#array of p-values for every signal trial 
p_value = pv_sum/(tester.nper*tester.filecount)
sigma = pd2sig(p_value)

np.savez('./data/' + tester.name + "_SIGMA.npz", sigma = sigma)