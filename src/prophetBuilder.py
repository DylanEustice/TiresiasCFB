import numpy as np
import pandas as pd
import neurolab as nl
import matplotlib.pyplot as plt
import os
import copy
import pickle
from src.util import *

# global paths
IO_DIR = os.path.join('data', 'inout_fields')
COMP_TEAM_DATA = os.path.join('data', 'compiled_team_data')
PRM_DIR = os.path.join('data', 'network_params')


class Params:
	def __init__(self, io_name, io_dir, n_prev_games, min_date, trainf, lyr,
		train_pct, lr, epochs, update_freq, show, minmax, hid_lyr, ibias, inp_avg):
		self.io_name = io_name
		self.io_dir = io_dir
		self.n_prev_games = n_prev_games
		self.min_date = min_date
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

	@classmethod
	def load(prms, fname, fdir=PRM_DIR):
		this = prms(*[None]*15)
		with open(os.path.join(fdir, fname),"r") as f:
			this = pickle.load(f)
		return this

	def save(self, fname, fdir=PRM_DIR):
		with open(os.path.join(fdir, fname),"w") as f:
			pickle.dump(self, f)

	def print_self(self):
		members = self.get_members()
		for m in members:
			print m, ":", getattr(self, m)

	def get_members(self):
		return [attr for attr in dir(self) if not attr.startswith("__")]





class Game:
	def __init__(self, this_inp_data, other_inp_data, n_prev, this_inp_fields,
		other_inp_fields, out_fields, avg_inp_callback, tar_data=None, **kwargs):
		"""
		Class holding information for game to be predicted
		inp_data:			all input data, converted to input form according to the #
							of games prior used for prediction and the input fields
		n_prev:				# of games prior used for prediction
		inp_fields:			fields of input data used for prediction
		out_fields: 		fields of output data being predicted
		avg_inp_callback:	function used to average input games
		tar_data:			target data to predict (None if game has not occurred)
		kwargs:				used for avg_inp_callback
		"""
		self.n_prev = n_prev
		self.this_inp_fields = this_inp_fields
		self.other_inp_fields = other_inp_fields
		self.out_fields = out_fields
		self.this_inp_data = self.filter_inp(this_inp_data, this_inp_fields)
		self.other_inp_data = self.filter_inp(other_inp_data, other_inp_fields)
		self.avg_inp_f = avg_inp_callback
		self.avg_inp_kwargs = kwargs
		# Get target data if exists
		if tar_data is not None:
			self.id = tar_data['Id']
			self.tar_data = tar_data[out_fields]
			self.date = tar_data['DateUtc']
			self.tids = tar_data[['this_TeamId', 'other_TeamId']]
		else:
			self.id = None
			self.tar_data = None
			self.date = None
			self.tids = None
		# Average inputs (set to tar_data option for autoencoder)
		if self.n_prev > -1:
			self.inp_data = self.all_inp()
		else:
			self.inp_data = self.tar_data
			

	def filter_inp(self, inp_data, inp_fields):
		"""
		Return up to the previous 'n_prev' valid games. Valid games are those
		with no NaN values in input fields. Assume the previous games are given
		such that their dates are sorted in ascending order and the number of 
		previous games is not 0.
		"""
		inp_data = inp_data.replace('-', np.nan)
		inp_data_np = np.asarray(inp_data[inp_fields])
		inp_data_np	= inp_data_np[~np.isnan(inp_data_np).any(axis=1)]
		return inp_data_np[0:min(inp_data_np.shape[0], self.n_prev), :]

	def all_inp(self):
		return np.hstack([self.avg_inp(use_this=True), self.avg_inp(use_this=False)])

	def avg_inp(self, use_this=True):
		data = self.this_inp_data if use_this else self.other_inp_data
		return self.avg_inp_f(data, **self.avg_inp_kwargs)


def train_net_from_scratch(io_name, io_dir=IO_DIR, n_prev_games=6, min_date=None,
	hid_lyr=10, trainf=nl.train.train_gdm, lyr=[nl.trans.SoftMax(),nl.trans.PureLin()],
	train_pct=0.5, lr=0.001, epochs=100, update_freq=20, show=20, minmax=1.0,
	ibias=1.0, inp_avg=np.mean):
	"""
	Setup data, setup network, and train network given inputs
	"""
	data = setup_train_data(io_name, io_dir=io_dir, n_prev_games=n_prev_games,
		min_date=min_date, inp_avg=inp_avg)
	train, test = partition_data(data['games'], data['norm_inp'],
		data['norm_tar'], train_pct=train_pct)
	net = setup_network(train, hid_lyr=hid_lyr, trainf=trainf, lyr=lyr,
		minmax=minmax, ibias=ibias)
	net, error = train_network(net, train, test, lr=lr, epochs=epochs,
		update_freq=update_freq, show=show)
	return net, data, train, test, error


def setup_train_data(io_name, io_dir=IO_DIR, n_prev_games=6, min_date=None,
	inp_avg=np.mean):
	"""
	io_name:		name of file listing I/O field names
	io_dir:			directory of I/O file
	n_prev_games:	max number of previous games to consider as input
	min_date:		last date of games to pull data from
	"""
	# build all games and get data arrays
	games = build_data(io_name, io_dir=io_dir, n_prev_games=n_prev_games,
		min_date=min_date, inp_avg=inp_avg)
	raw_inp = np.vstack([g.inp_data for g in games])
	raw_tar = np.vstack([g.tar_data for g in games])
	# standardize data
	norm_inp = normalize_data(raw_inp.astype('double'))
	norm_tar = normalize_data(raw_tar.astype('double'))
	out_data = {}
	out_data['games'] = games
	out_data['norm_inp'] = norm_inp
	out_data['norm_tar'] = norm_tar
	out_data['raw_inp'] = raw_inp
	out_data['raw_tar'] = raw_tar
	return out_data


def build_data(io_name, io_dir=IO_DIR, n_prev_games=6, min_date=None,
	inp_avg=np.mean):
	"""
	Read from compiled dataFrame and builds up a list of games which
	occurred after min_date. Format data according to the I/O file
	name provided. Games are used for prediction/training.
	"""
	# Read I/O fields
	io_fields = load_json(io_name, fdir=io_dir)
	inp_fields = io_fields['inputs']
	out_fields = io_fields['outputs']
	# Read data and seperate by teams
	all_data = pd.read_pickle(os.path.join(COMP_TEAM_DATA, 'all.df'))
	data_by_team = partition_data_to_teams(all_data)
	# Build sets of inputs and target outputs for all possible games
	games = []
	for tid, tgames in data_by_team.iteritems():
		print "{0:g},".format(tid), 
		# all previous games are potential inputs
		for _, game in tgames.iterrows():
			# ensure game is after min_date, score field isn't blank,
			# and opponent is in history
			is_recent = game['DateUtc'] >= min_date
			out_data = game[out_fields]
			has_output = not any(out_data == '-') and all(pd.notnull(out_data))
			opp_avail = game['other_TeamId'] in data_by_team
			if not (is_recent and has_output and opp_avail):
				continue
			# get previous games played by both teams
			other_all_games = data_by_team[game['other_TeamId']]
			this_prev_games = tgames[game['DateUtc'] > tgames['DateUtc']]
			other_prev_games = other_all_games[game['DateUtc']
							   > other_all_games['DateUtc']]
			if this_prev_games.shape[0] <= 0 or other_prev_games.shape[0] <= 0:
				continue
			# build game class instance and append
			new_game = Game(this_prev_games, other_prev_games, n_prev_games,
				inp_fields, inp_fields, out_fields, inp_avg, tar_data=game, axis=0)
			if (new_game.this_inp_data.shape[0] > 0 and 
				new_game.other_inp_data.shape[0] > 0):
				games.append(new_game)
	print '\nDone.\n'
	return games


def partition_data(games, inp, tar, train_pct=0.5):
	"""
	Randomly partition games for training and validation
	games:		list of games to be partitioned
	inp:		input data
	tar:		target data
	train_pct:	percentage of data used for training
	"""
	gids = list(set(g.id for g in games))
	rnd_vec = np.random.random(len(gids))
	train_idx = rnd_vec < train_pct
	train_gids = {id_ for i, id_ in enumerate(gids) if train_idx[i]}
	# partition
	train = {}
	train['inp'] = inp[np.array([g.id in train_gids for g in games]),:]
	train['tar'] = tar[np.array([g.id in train_gids for g in games]),:]
	test = {}
	test['inp'] = inp[np.array([g.id not in train_gids for g in games]),:]
	test['tar'] = tar[np.array([g.id not in train_gids for g in games]),:]
	return train, test


def setup_network(train_data, hid_lyr=10, trainf=nl.train.train_gdm,
	lyr=[nl.trans.SoftMax(),nl.trans.PureLin()], minmax=1.0, ibias=1.0):
	"""
	train_data: used to set input and output layer sizes
	hid_lyr:	size of the hidden network layer
	trainf:		training function
	lyr:		layer activation functions
	"""
	lyr_rng = zip(train_data['inp'].min(axis=0), train_data['inp'].max(axis=0))
	net = nl.net.newff(lyr_rng, [hid_lyr, train_data['tar'].shape[1]], lyr)
	net.trainf = trainf
	for l in net.layers:
		l.initf = nl.init.InitRand([-minmax, minmax], 'wb')
	net.layers[0].np['b'][:] = 1.0
	return net


def train_network(net, train, test, lr=0.001, epochs=100, update_freq=20, show=20):
	"""
	Train neural network
	net:			network to be trained
	train:			training data
	test:			validation data
	lr:				learning rate
	epochs:			training epochs
	update_freq:	frequency which error is checked and outputted
	show:			frequency which neurolab outputs training updates
	"""
	error_test = []
	error_train = []
	msef = nl.error.MSE()
	best_net = copy.deepcopy(net)
	best_error = float('inf')
	for i in range(epochs / update_freq):
		error_tmp = net.train(train['inp'], train['tar'], epochs=update_freq,
			show=show, lr=lr)
		error_train.append(msef(train['tar'], net.sim(train['inp'])))
		error_test.append(msef(test['tar'], net.sim(test['inp'])))
		if error_test[-1] < best_error:
			best_net = copy.deepcopy(net)
		print "Iters: {}".format(i*update_freq)
		print "Train: {}".format(error_train[-1])
		print " Test: {}".format(error_test[-1])
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
	out = (net.sim(train['inp']), net.sim(test['inp']))
	idx = (np.argsort(np.ndarray.flatten(train['tar'])), 
		   np.argsort(np.ndarray.flatten(test['tar'])))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	# train
	ax1.plot(np.ndarray.flatten(train['tar'])[idx[0]], 'o', alpha=alpha)
	ax1.plot(np.ndarray.flatten(out[0])[idx[0]], 'o', alpha=alpha)
	# test
	ax2.plot(np.ndarray.flatten(test['tar'])[idx[1]], 'o', alpha=alpha)
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