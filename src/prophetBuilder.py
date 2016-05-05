import numpy as np
import pandas as pd
import neurolab as nl
import matplotlib.pyplot as plt
import argparse
import os
from src.util import load_json, standardize_data, moving_avg

# global paths
IO_DIR = os.path.join('data', 'inout_fields')
COMP_TEAM_DATA = os.path.join('data', 'compiled_team_data')


class Game:
	def __init__(self, this_inp_data, other_inp_data, n_prev, this_inp_fields,
				 other_inp_fields, out_fields, tar_data=None):
		"""
		Class holding information for game to be predicted
		inp_data:	all input data, converted to input form according to the #
					of games prior used for prediction and the input fields
		n_prev:		# of games prior used for prediction
		inp_fields:	fields of input data used for prediction
		out_fields: fields of output data being predicted
		tar_data:	data for game to be predicted (None if game has not occurred)
		"""
		self.n_prev = n_prev
		self.this_inp_fields = this_inp_fields
		self.other_inp_fields = other_inp_fields
		self.out_fields = out_fields
		self.this_inp_data = self.filter_inp(this_inp_data, this_inp_fields)
		self.other_inp_data = self.filter_inp(other_inp_data, other_inp_fields)
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


def train_network(io_name, io_dir=IO_DIR, n_prev_games=6, min_date=None):
	# get data
	games, inp, tar, raw_inp, raw_tar = setup_train_data(
		io_name, io_dir=io_dir,	n_prev_games=n_prev_games, min_date=min_date)
	# randomly partition games for training and validation
	gids = list(set(g.id for g in games))
	rnd_vec = np.random.random(len(gids))
	train_per = 0.5
	train_idx = rnd_vec < train_per
	train_gids = {id_ for i, id_ in enumerate(gids) if train_idx[i]}
	# partition
	train = {}
	train['inp'] = inp[np.array([g.id in train_gids for g in games]),:]
	train['tar'] = tar[np.array([g.id in train_gids for g in games]),:]
	test = {}
	test['inp'] = inp[np.array([g.id not in train_gids for g in games]),:]
	test['tar'] = tar[np.array([g.id not in train_gids for g in games]),:]
	# set up network parameters
	hid_lyr = 10
	lyr = [nl.trans.SoftMax(), nl.trans.PureLin()]
	epochs = 2000
	show = 50
	lr = 0.00001
	update_freq = 20
	trainf = nl.train.train_gdm
	# set up network
	lyr_rng = zip(train['inp'].min(axis=0), train['inp'].max(axis=0))
	net = nl.net.newff(lyr_rng, [hid_lyr, train['tar'].shape[1]], lyr)
	net.trainf = trainf
	# train network
	error_test = []
	error_train = []
	msef = nl.error.MSE()
	for i in range(epochs / update_freq):
		error_tmp = net.train(train['inp'], train['tar'], epochs=update_freq, show=show, lr=lr)
		error_train.append(msef(train['tar'], net.sim(train['inp'])))
		error_test.append(msef(test['tar'], net.sim(test['inp'])))
		print "Iters: {}".format(i*update_freq)
		print "Train: {}".format(error_train[-1])
		print " Test: {}".format(error_test[-1])
	# plotting
	plt.ion()
	plt.figure()
	x = range(epochs / update_freq)
	plt.plot(x, error_train, x, error_test)
	fig = plt.figure()
	out = (net.sim(train['inp']), net.sim(test['inp']))
	idx = (np.argsort(np.ndarray.flatten(train['tar'])), 
		   np.argsort(np.ndarray.flatten(test['tar'])))
	ax1 = fig.add_subplot(121)
	ax2 = fig.add_subplot(122)
	# train
	ax1.plot(np.ndarray.flatten(train['tar'])[idx[0]], 'o')
	ax1.plot(np.ndarray.flatten(out[0])[idx[0]], 'o')
	ax1.plot(moving_avg(np.ndarray.flatten(out[0])[idx[0]]), 'o')
	# test
	ax2.plot(np.ndarray.flatten(test['tar'])[idx[1]], 'o')
	ax2.plot(np.ndarray.flatten(out[1])[idx[1]], 'o')
	ax2.plot(moving_avg(np.ndarray.flatten(out[1])[idx[1]]), 'o')
	return net, error, train, test


def setup_train_data(io_name, io_dir=IO_DIR, n_prev_games=6, min_date=None):
	# build all games and get data arrays
	games = build_data(io_name, io_dir=io_dir, n_prev_games=n_prev_games, min_date=min_date)
	this_inp = np.vstack([g.this_inp_data.mean(axis=0) for g in games])
	other_inp = np.vstack([g.other_inp_data.mean(axis=0) for g in games])
	raw_inp = np.hstack([this_inp, other_inp])
	raw_tar = np.vstack([g.tar_data for g in games])
	# standardize data
	norm_inp = standardize_data(raw_inp.astype('double'))
	norm_tar = standardize_data(raw_tar.astype('double'))
	return games, norm_inp, norm_tar, raw_inp, raw_tar


def build_data(io_name, io_dir=IO_DIR, n_prev_games=6, min_date=None):
	"""
	Read from compiled dataFrame and builds up a list of games which
	occurred after min_date. Format data according to the I/O file
	name provided. Games are used for prediction/training.
	"""
	# Read I/O fields
	io_fields = read_io(io_name, fdir=io_dir)
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
			has_score = not any(game[out_fields] == '-')
			opp_avail = game['other_TeamId'] in data_by_team
			if not (is_recent and has_score and opp_avail):
				continue
			# get previous games played by both teams
			other_all_games = data_by_team[game['other_TeamId']]
			this_prev_games = tgames[game['DateUtc'] > tgames['DateUtc']]
			other_prev_games = other_all_games[game['DateUtc'] > other_all_games['DateUtc']]
			if this_prev_games.shape[0] <= 0 or other_prev_games.shape[0] <= 0:
				continue
			# build game class instance and append
			new_game = Game(this_prev_games, other_prev_games, n_prev_games,
							inp_fields, inp_fields, out_fields, game)
			if (new_game.this_inp_data.shape[0] > 0 and 
				new_game.other_inp_data.shape[0] > 0):
				games.append(new_game)
	print '\nDone.\n'
	return games


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


def read_io(fname, fdir=IO_DIR):
	"""
	Reads saved JSON file with input and output field specifiers
	"""
	io_fields = load_json(fname, fdir=fdir)
	return io_fields