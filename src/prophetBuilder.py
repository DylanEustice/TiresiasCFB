import numpy as np
import pandas as pd
import neurolab as nl
import matplotlib.pyplot as plt
import os
import copy
import pickle
import datetime
import src.util as util
from src.team import build_all_teams
import src.default_parameters as default


class Params:
	def __init__(self, io_name, io_dir, min_games, min_date, max_date, trainf, lyr,
		train_pct, lr, epochs, update_freq, show, minmax, hid_lyr, ibias, inp_avg,
		norm_func, date_diff, home_only):
		self.io_name = io_name
		self.io_dir = io_dir
		self.min_games = min_games
		self.min_date = min_date
		self.max_date = max_date
		self.trainf = trainf
		self.lyr = lyr
		self.train_pct = train_pct
		self.lr = lr
		self.epochs = epochs
		self.update_freq = update_freq
		self.show = show
		self.minmax = minmax
		self.hid_lyr = hid_lyr
		self.ibias = ibias
		self.inp_avg = inp_avg
		self.norm_func = norm_func
		self.date_diff = date_diff
		self.home_only = home_only
		self.kwargs = {}

	@classmethod
	def load(prms, fname, fdir=default.prm_dir):
		this = prms(*[None]*19)
		with open(os.path.join(fdir, fname),"r") as f:
			this = pickle.load(f)
		return this

	def save(self, fname, fdir=default.prm_dir):
		with open(os.path.join(fdir, fname),"w") as f:
			pickle.dump(self, f)

	def print_self(self):
		members = self.get_members()
		for m in members:
			print m, ":", getattr(self, m)

	def get_members(self):
		return [attr for attr in dir(self) if not attr.startswith("__")]


def build_dataset(prm):
	"""
	prm:	paramters file
	"""
	# Load data
	all_data = util.load_all_dataFrame()
	teams = build_all_teams(all_data=all_data)
	# Read I/O fields
	io_fields = util.load_json(prm.io_name, fdir=prm.io_dir)
	inp_fields = io_fields['inputs']
	tar_fields = io_fields['outputs']
	# Keyword args for data averaging
	if prm.inp_avg is np.mean:
		kwargs = dict([('axis',0)])
	elif prm.inp_avg is util.elo_mean:
		kwargs = dict([('fields',inp_fields)])
	# Extract game training data
	games = []
	teams_dict = dict([(t.tid, t) for t in teams])
	for t in teams:
		print t.tid
		games.extend(t.get_training_data(prm, teams_dict, inp_fields, tar_fields, **kwargs))
	# Build dataset
	full_dataset = {}
	full_dataset['raw_inp'] = np.vstack([g['inp'] for g in games])
	full_dataset['raw_tar'] = np.vstack([g['tar'] for g in games])
	full_dataset['norm_inp'] = prm.norm_func(full_dataset['raw_inp'].astype('double'))
	full_dataset['norm_tar'] = prm.norm_func(full_dataset['raw_tar'].astype('double'))
	full_dataset['norm_func'] = str(prm.norm_func)
	full_dataset['games'] = games
	# Partition dataset and return
	dataset = partition_data(full_dataset, train_pct=prm.train_pct)
	return dataset


def partition_data(data, train_pct=0.5):
	"""
	Randomly partition games for training and validation
	games:		list of games to be partitioned
	data:		input/target raw/normalized data
	train_pct:	percentage of data used for training
	"""
	games = data['games']
	gids = list(set(g['id'] for g in games))
	rnd_vec = np.random.random(len(gids))
	train_idx = rnd_vec < train_pct
	train_gids = {id_ for i, id_ in enumerate(gids) if train_idx[i]}
	# partition
	train = {}
	train['norm_inp'] = data['norm_inp'][np.array([g['id'] in train_gids for g in games]),:]
	train['norm_tar'] = data['norm_tar'][np.array([g['id'] in train_gids for g in games]),:]
	train['raw_inp'] = data['raw_inp'][np.array([g['id'] in train_gids for g in games]),:]
	train['raw_tar'] = data['raw_tar'][np.array([g['id'] in train_gids for g in games]),:]
	test = {}
	test['norm_inp'] = data['norm_inp'][np.array([g['id'] not in train_gids for g in games]),:]
	test['norm_tar'] = data['norm_tar'][np.array([g['id'] not in train_gids for g in games]),:]
	test['raw_inp'] = data['raw_inp'][np.array([g['id'] not in train_gids for g in games]),:]
	test['raw_tar'] = data['raw_tar'][np.array([g['id'] not in train_gids for g in games]),:]
	out_data = {'train': train, 'test': test}
	return out_data


def build_prms_file(prm_name, io_name, io_dir=default.io_dir, min_games=6,
	hid_lyr=10, trainf=nl.train.train_gdm, lyr=[nl.trans.SoftMax(),nl.trans.PureLin()],
	train_pct=0.5, lr=0.001, epochs=100, update_freq=20, show=20, minmax=1.0,
	ibias=1.0, inp_avg=np.mean, norm_func=util.normalize_data, min_date=datetime.datetime(1900,1,1),
	date_diff=datetime.timedelta(weeks=26), home_only=True):
	"""
	Build Params object and save to file
	  Note: min_games = -1 -> autoencoder
	  		min_games = 0  -> same game
	"""
	prms = Params(io_name, io_dir, min_games, min_date, trainf, lyr, train_pct, lr, 
		epochs, update_freq, show, minmax, hid_lyr, ibias, inp_avg, norm_func, date_diff,
		home_only)
	with open(os.path.join(default.prm_dir, prm_name), 'w') as f:
		pickle.dump(prms, f)
	return prms


def train_net_from_prms(prm=None, prm_name=None, data_file=None, fdir=default.data_dir):
	"""
	Load .prm file and train network based on those parameters
	"""
	if prm is None:
		assert(prm_name is not None)
		prm = Params.load(prm_name)
	if data_file is None:
		# Build data and train
		net, dataset, error = train_net_from_scratch(prm)
		# Save if desired
		fname = raw_input('Save built data? Enter N or file name (*.pkl): ')
		if fname != 'N':
			with open(os.path.join(fdir, fname), 'w') as f:
				pickle.dump(dataset, f)
	else:
		# Read data and train
		with open(os.path.join(fdir, data_file), 'r') as f:
			dataset = pickle.load(f)
		net, _, error = train_net_from_scratch(prm, dataset=dataset)
	return net, dataset, error


def train_net_from_scratch(prm, dataset=None):
	"""
	Setup data, setup network, and train network given inputs
	"""
	if dataset is None:
		dataset = build_dataset(prm)
	# Train network
	net = setup_network(dataset['train'], prm)
	net, error = train_network(net, dataset, prm)
	return net, dataset, error


def setup_network(train_data, prm):
	"""
	train_data: used to set input and output layer sizes
	prm:		paramters file
	"""
	lyr_rng = zip([0.]*train_data['norm_inp'].shape[1], [1.]*train_data['norm_inp'].shape[1])
	net = nl.net.newff(lyr_rng, [prm.hid_lyr, train_data['norm_tar'].shape[1]], prm.lyr)
	net.trainf = prm.trainf
	for l in net.layers:
		l.initf = nl.init.InitRand([-prm.minmax, prm.minmax], 'wb')
	net.layers[0].np['b'][:] = prm.ibias
	return net


def train_network(net, data, prm):
	"""
	Train neural network
	net:	network to be trained
	data:	training and testing data
	prm:	paramters file
	"""
	train = data['train']
	test = data['test']
	error_test = []
	error_train = []
	msef = nl.error.MSE()
	best_net = copy.deepcopy(net)
	best_error = float('inf')
	test_data_acc = dict([('inp', data['test']['norm_inp']), ('tar', data['test']['norm_tar'])])
	train_data_acc = dict([('inp', data['train']['norm_inp']), ('tar', data['train']['norm_tar'])])
	for i in range(prm.epochs / prm.update_freq):
		if net.trainf is not nl.train.train_bfgs:
			error_tmp = net.train(train['norm_inp'], train['norm_tar'],
				epochs=prm.update_freq,	show=prm.show, lr=prm.lr)
		else:
			error_tmp = net.train(train['norm_inp'], train['norm_tar'],
				epochs=prm.update_freq,	show=prm.show)
		error_train.append(msef(train['norm_tar'], net.sim(train['norm_inp'])))
		error_test.append(msef(test['norm_tar'], net.sim(test['norm_inp'])))
		if error_test[-1] < best_error:
			best_net = copy.deepcopy(net)
		print "Iters    : {} / {}".format(i*prm.update_freq, prm.epochs)
		print "Train    : {}".format(error_train[-1])
		print " Test Acc: {}".format(util.get_winner_acc(net, train_data_acc))
		print " Test    : {}".format(error_test[-1])
		print " Test Acc: {}".format(util.get_winner_acc(net, test_data_acc))
	error = {}
	error['train'] = error_train
	error['test'] = error_test
	plot_data(best_net, error, train, test)
	return best_net, error


def plot_data(net, error, train, test, alpha=0.1):
	"""
	"""
	plt.ion()
	plt.figure()
	x = range(len(error['train']))
	plt.plot(x, error['train'], x, error['test'])
	fig = plt.figure()
	out = (net.sim(train['norm_inp']), net.sim(test['norm_inp']))
	idx = (np.argsort(np.ndarray.flatten(train['norm_tar'])), 
		   np.argsort(np.ndarray.flatten(test['norm_tar'])))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	# train
	ax1.plot(np.ndarray.flatten(train['norm_tar'])[idx[0]], 'o', alpha=alpha)
	ax1.plot(np.ndarray.flatten(out[0])[idx[0]], 'o', alpha=alpha)
	# test
	ax2.plot(np.ndarray.flatten(test['norm_tar'])[idx[1]], 'o', alpha=alpha)
	ax2.plot(np.ndarray.flatten(out[1])[idx[1]], 'o', alpha=alpha)


def partition_data_to_teams(all_data):
	"""
	Partitions data from full pandas array into a
	dictionary with a key for each team ID found
	"""
	team_ids = all_data['this_TeamId']
	unq_ids = np.unique(team_ids)
	data_by_team = {}
	for id_ in unq_ids:
		data_by_team[id_] = all_data[all_data['this_TeamId']==id_]
	return data_by_team