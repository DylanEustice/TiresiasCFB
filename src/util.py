import os
import json
import shutil
import errno
import pandas as pd
import numpy as np
import re
import src.default_parameters as default


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


def load_all_dataFrame():
	"""
	Load data then filter bad data and FCS games
	"""
	all_data = pd.read_pickle(os.path.join(default.comp_team_dir, 'all.df'))
	all_data = all_data[all_data['this_Score'] != '-']
	all_data = all_data[all_data['other_conferenceId'] != '-1']
	return all_data


def load_schedule():
	"""
	Load future games
	"""
	schedule = pd.read_pickle(os.path.join(default.comp_team_dir, 'schedule.df'))
	return schedule


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


def linear_regression(X, y):
	"""
	Preform linear regression on inputs X with output y
	"""
	X = np.matrix(X)
	y = np.matrix(y)
	return (X.T * X)**-1 * X.T * y


def elo_mean(x, fields, elo_fields=default.all_elo_fields):
	"""
	Preforms a mean on a data series, but using the last indexed
	elo value instead of averaging
	"""
	if x.shape[0] <= 1:
		return x
	u = np.zeros([1,x.shape[1]])
	for i in range(x.shape[1]):
		u[0,i] = np.mean(x[:,i]) if fields[i] not in elo_fields else x[-1,i]
	return u


def test_corr(all_data, fields):
	"""
	IN DEV
	"""
	A = np.array(all_data[fields[0]])
	B = np.array(all_data[fields[1]])
	# Remove dashed
	ixKeep = np.logical_and(A != '-', B != '-')
	A, B = A[ixKeep].astype(float), B[ixKeep].astype(float)
	# Remove NaN
	is_num = lambda x: np.logical_not(np.isnan(x))
	ixKeep = np.logical_and(is_num(A), is_num(B))
	A, B = A[ixKeep], B[ixKeep]
	corrcoef = np.corrcoef(A, B)[0,1]
	return corrcoef


def flip_this_other(fields):
	"""
	Flips locations of 'this' and 'other' in field array
	"""
	out_fields = ['' for f in fields]
	for i, f in enumerate(fields):
		if re.match('this', f):
			out_fields[i] = re.sub('this', 'other', f)
		elif re.match('other', f):
			out_fields[i] = re.sub('other', 'this', f)
		else:
			out_fields[i] = f
	return out_fields



def get_pct_correct(y_true, y_pred):
	"""
	Assumes these are M x 2 score arrays
	"""
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	correct = (y_true[:,0] > y_true[:,1]) == (y_pred[:,0] > y_pred[:,1])
	return np.mean(correct)