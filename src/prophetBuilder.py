import numpy as np
import pandas as pd
import neurolab as nl
import matplotlib.pyplot as plt
import os
import copy
import pickle
import datetime
from src.util import *
from src.team import *

# global paths
IO_DIR = os.path.join('data', 'inout_fields')
COMP_TEAM_DATA = os.path.join('data', 'compiled_team_data')
PRM_DIR = os.path.join('data', 'network_params')


class Params:
	def __init__(self, io_name, io_dir, min_games, min_date, trainf, lyr, train_pct, lr,
		epochs, update_freq, show, minmax, hid_lyr, ibias, inp_avg, norm_func, date_diff,
		home_only):
		self.io_name = io_name
		self.io_dir = io_dir
		self.min_games = min_games
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
		self.norm_func = norm_func
		self.date_diff = date_diff
		self.home_only = home_only
		self.kwargs = {}

	@classmethod
	def load(prms, fname, fdir=PRM_DIR):
		this = prms(*[None]*18)
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
		if self.this_inp_data.shape[0] == 0 or self.other_inp_data.shape[0] == 0:
			return
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
		if self.n_prev > 0:
			return inp_data_np[0:min(inp_data_np.shape[0], self.n_prev), :]
		else:
			return inp_data_np

	def all_inp(self):
		try:
			return np.hstack([self.avg_inp(use_this=True), self.avg_inp(use_this=False)])
		except:
			import pdb; pdb.set_trace()

	def avg_inp(self, use_this=True):
		data = self.this_inp_data if use_this else self.other_inp_data
		return self.avg_inp_f(data, **self.avg_inp_kwargs)


def build_dataset(prm):
	"""
	prm:	paramters file
	"""
	# Load data
	all_data = load_all_dataFrame()
	teams = build_all_teams(all_data=all_data)
	# Read I/O fields
	io_fields = load_json(prm.io_name, fdir=prm.io_dir)
	inp_fields = io_fields['inputs']
	tar_fields = io_fields['outputs']
	# Keyword args for data averaging
	if prm.inp_avg is np.mean:
		kwargs = dict([('axis',0)])
	elif prm.inp_avg is elo_mean:
		kwargs = dict([('fields',this_inp_fields)])
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
	part_dataset = partition_data(full_dataset, train_pct=prm.train_pct)
	return part_dataset


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


def build_prms_file(prm_name, io_name, io_dir=IO_DIR, min_games=6,
	hid_lyr=10, trainf=nl.train.train_gdm, lyr=[nl.trans.SoftMax(),nl.trans.PureLin()],
	train_pct=0.5, lr=0.001, epochs=100, update_freq=20, show=20, minmax=1.0,
	ibias=1.0, inp_avg=np.mean, norm_func=normalize_data, min_date=datetime.datetime(1900,1,1),
	date_diff=datetime.timedelta(weeks=26), home_only=True):
	"""
	Build Params object and save to file
	  Note: min_games = -1 -> autoencoder
	  		min_games = 0  -> same game
	"""
	prms = Params(io_name, io_dir, min_games, min_date, trainf, lyr, train_pct, lr, 
		epochs, update_freq, show, minmax, hid_lyr, ibias, inp_avg, norm_func, date_diff,
		home_only)
	with open(os.path.join(PRM_DIR, prm_name), 'w') as f:
		pickle.dump(prms, f)


def train_net_from_prms(prm=None, prm_name=None, data_file=None, fdir=DATA_DIR):
	"""
	Load .prm file and train network based on those parameters
	"""
	if prm is None:
		assert(prm_name is not None)
		prm = Params.load(prm_name)
	if data_file is None:
		# Build data and train
		net, part_data, error = train_net_from_scratch(prm)
		# Save if desired
		fname = raw_input('Save built data? Enter N or file name (*.pkl): ')
		if fname != 'N':
			with open(os.path.join(fdir, fname), 'w') as f:
				pickle.dump(part_data, f)
	else:
		# Read data and train
		with open(os.path.join(fdir, data_file), 'r') as f:
			part_data = pickle.load(f)
		net, _, error = train_net_from_scratch(prm, part_data=part_data)
	return net, part_data, error


def train_net_from_scratch(prm, part_data=None):
	"""
	Setup data, setup network, and train network given inputs
	"""
	if part_data is None:
		# Get partitioned data
		all_data = setup_train_data(prm)
		part_data = partition_data(all_data, train_pct=prm.train_pct)
	# Train network
	net = setup_network(part_data['train'], prm)
	net, error = train_network(net, part_data, prm)
	return net, part_data, error


def build_data(prm):
	"""
	Read from compiled dataFrame and builds up a list of games which
	occurred after min_date. Format data according to the I/O file
	name provided. Games are used for prediction/training.
	"""
	# Read I/O fields
	io_fields = load_json(prm.io_name, fdir=prm.io_dir)
	inp_fields = io_fields['inputs']
	out_fields = io_fields['outputs']
	# Decide keyword argument to use
	if prm.inp_avg is np.mean:
		kwargs = dict([('axis',0)])
	elif prm.inp_avg is elo_mean:
		kwargs = dict([('fields',inp_fields)])
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
			is_recent = game['DateUtc'] >= prm.min_date
			out_data = game[out_fields]
			has_output = not any(out_data == '-') and all(pd.notnull(out_data))
			opp_avail = game['other_TeamId'] in data_by_team
			if not (is_recent and has_output and opp_avail):
				continue
			# get previous games played by both teams
			other_all_games = data_by_team[game['other_TeamId']]
			if prm.n_prev_games != 0:
				this_prev_games = tgames[game['DateUtc'] > tgames['DateUtc']]
				other_prev_games = other_all_games[game['DateUtc'] > other_all_games['DateUtc']]
			else:
				this_prev_games = tgames[game['DateUtc'] == tgames['DateUtc']]
				other_prev_games = other_all_games[game['DateUtc'] == other_all_games['DateUtc']]
			if this_prev_games.shape[0] <= 0 or other_prev_games.shape[0] <= 0:
				continue
			# build game class instance and append
			new_game = Game(this_prev_games, other_prev_games, prm.n_prev_games,
				inp_fields, inp_fields, out_fields, prm.inp_avg, tar_data=game, **kwargs)
			if new_game.this_inp_data.shape[0] > 0 and new_game.other_inp_data.shape[0] > 0:
				games.append(new_game)
	print '\nDone.\n'
	return games


def setup_network(train_data, prm):
	"""
	train_data: used to set input and output layer sizes
	prm:		paramters file
	"""
	lyr_rng = zip(train_data['norm_inp'].min(axis=0), train_data['norm_inp'].max(axis=0))
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
		print "Iters: {} / {}".format(i*prm.update_freq, prm.epochs)
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