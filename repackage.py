from package import *

tester = pickle.load(open('./Testers/' + sys.argv[1],'rb'))

fils=glob.glob("./working/*.npz")
filtered_fils = []

for fil in fils:
    if tester.timetag in fil and "BKG" in fil:
        filtered_fils.append(fil)
fils = filtered_fils

#filters
if len(fils) != tester.filecount:
    print("Number of background files found and filecount stated differ..")

outdat=np.load(fils[0])['TS']

declinations=np.load(fils[0])['dec']

cputime = 0
statstr=" "
for j,fil in enumerate(fils[1:]):
    print(len(statstr)*" ",end="\r")
    statstr=f"merging file {j+1}, {fil}"
    print(statstr,end="\r")
    file=np.load(fil)
    outdat = np.concatenate((outdat,file["TS"]))
    declinations = np.concatenate((declinations,file["dec"]))
    cputime += file['runtime']

print("sorting data")

decs = []
#there may be a nice vectorized way to handle this, but the 2d arrays make indexing difficult
for i in range(len(tester.Methods)):
    decs.append(declinations[:,i][outdat[:,i].argsort(axis=0)])
declinations = np.column_stack(decs)
outdat.sort(axis=0)

print("packing data")

np.savez('./data/' +tester.name +'/' + tester.name + "_BKG.npz", TS = outdat, dec = declinations, time = cputime)
print(f'file saved to ./data/{tester.name}/{tester.name}_BKG.npz')
