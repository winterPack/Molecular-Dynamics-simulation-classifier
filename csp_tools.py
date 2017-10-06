import numpy as np
import os

def load_data_range(angles = range(19), runIds = range(50), verbose=False):
	"load CSP data: 19 tilt angles (labeled 0-18) and 50 runs (labeled 0-49)"
	X = []
	y = []
	csv_repo = '/share3/hydra_export/winter/CSP_Neural_Net/tilt_pts_csp_50x50x50'
	for a in angles:
		for r in runIds:
			csv_file = "tilt{}_run{}.csv".format(a*5,r+1)
			csv_file = os.path.join(csv_repo,csv_file)
			csv_data = np.loadtxt(csv_file,delimiter=',')
			csv_data = np.reshape(csv_data,(50,50,50,1))
			X.append(csv_data)
			y.append(a)
			if verbose:
				print(csv_file)
	return np.array(X), np.array(y)

def random_roll(X): #roll/ shift
    res = np.zeros_like(X)
    for i in range(X.shape[0]):
        roll_vec = np.random.randint(0,50,4)
        res[i] = np.roll(X[i],roll_vec,(0,1,2,3))  # np.__version__ >= 1.12.0
    return res
