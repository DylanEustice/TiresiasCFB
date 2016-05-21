import os
import json
import shutil
import errno
import pandas as pd
import numpy as np

# global paths
IO_DIR = os.path.join('data', 'inout_fields')
COMP_TEAM_DATA = os.path.join('data', 'compiled_team_data')
PRM_DIR = os.path.join('data', 'network_params')
DATA_DIR = os.path.join('data', 'data_sets')


def debug_assert(condition):
	try:
		assert(condition)
	except AssertionError:
		import pdb
		pdb.set_trace()
		print "Debugger set. Enter 'u' to go up in stack frame"


def ensure_path(path):
	"""
	Make sure os path exists, create it if not
	"""
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise


def dump_json(data, fname, fdir='.', indent=4):
	"""
	Save data to file. 
	NOTE: Writes as text file, not binary.
	"""
	ensure_path(fdir)
	with open(os.path.join(fdir, fname), 'w') as f:
		json.dump(data, f, indent=indent, sort_keys=True)


def load_json(fname, fdir='.'):
	"""
	Reads data from file. 
	NOTE: Reads from text file, not binary.
	"""
	with open(os.path.join(fdir, fname), 'r') as f:
		return json.load(f)


def copy_dir(src, dst):
	"""
	Attempt to copy directory, on failure copy file. Will overwrite
	any files in dst.
	"""
	# Remove destination directory if already exists
	if os.path.exists(dst):
		shutil.rmtree(dst)
	# Copy directory over
	try:
		shutil.copytree(src, dst)
	except OSError as exc:
		if exc.errno == errno.ENOTDIR:
			shutil.copy(src, dst)
		else:
			raise Exception()


def grab_scraper_data(src=os.path.join('..','BarrelRollCFBData','data'),
					  dst=os.path.join('data')):
	"""
	Copy in data directory from BarrelRollCFBData
	"""
	copy_dir(src, dst)


def load_team_DataFrame(team_id, path_to_data='.'):
	fname = str(team_id) + '_DataFrame.df'
	return pd.read_pickle(os.path.join(path_to_data, 'data', 'compiled_team_data', fname))


def standardize_data(data, std=None, mean=None):
	"""
	Standardize numpy array data using formula:
		x_out = (x - x_mean) / x_std
	Assumes data is M x N where M is the observations and
	N is the data type.
	"""
	if std is None and mean is None:
		return np.divide(data - data.mean(axis=0), data.std(axis=0))
	elif std is not None and mean is not None:
		return np.divide(data - mean, std)
	else:
		UserWarning("Must enter both STD and MEAN, only one entered.")
		return None


def normalize_data(data, min_=None, max_=None):
	"""
	Normalize numpy array data using formula:
		x_out = (x - x_min) / x_max
	Assumes data is M x N where M is the observations and
	N is the data type.
	"""
	if min_ is None and max_ is None:
		return np.divide(data - data.min(axis=0), data.max(axis=0))
	elif min_ is not None and max_ is not None:
		return np.divide(data - min_, max_)
	else:
		UserWarning("Must enter both STD and MEAN, only one entered.")
		return None


def moving_avg(data, n=10):
	"""
	Calculate n long moving average
	"""
	ret = np.cumsum(data)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n-1:] / n


def get_winner_acc(net, data):
	"""
	Given a neural network to predict game outcomes,
	calculate the % the winner is correct
	"""
	out = net.sim(data['inp'])
	tar_idx = data['tar'][:,1] > data['tar'][:,0]
	out_idx = out[:,1] > out[:,0]
	return 1.*np.sum(tar_idx == out_idx) / tar_idx.shape[0]


def idv_out_mse(out, tar):
	"""
	Return an array of the MSE for each individual output
	"""
	diff = out - tar
	return np.mean(np.power(diff, 2), axis=0)


def idv_out_bias(out, tar):
	"""
	Return an array of the bias for each individual output
	"""
	diff = out - tar
	return np.mean(diff, axis=0)
	