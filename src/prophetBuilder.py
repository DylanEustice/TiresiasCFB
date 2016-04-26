import numpy as np
import pandas as pd
import neurolab as nl
import matplotlib.pyplot as plt
import argparse
import os
from src.util import load_json

# global paths
IO_DIR = os.path.join('data', 'inout_fields')
COMP_TEAM_DATA = os.path.join('data', 'compiled_team_data')


class Game:
	def __init__(self, inp_data, n_prev, inp_fields, out_fields=None, tar_data=None):
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
		self.inp_fields = inp_fields
		self.out_fields = out_fields
		self.inp_data = self.filter_all_inp(inp_data)
		self.tar_data = None if tar_data is None else tar_data[out_fields]

	def filter_all_inp(self, all_inp):
		"""
		Return up to the previous 'n_prev' valid games. Valid games are those
		with no NaN values in input fields. Assume the previous games are given
		such that their dates are sorted in ascending order and the number of 
		previous games is not 0.
		"""
		all_inp = all_inp.replace('-', np.nan)
		all_inp_arr = np.asarray(all_inp[self.inp_fields])
		inp_arr	= all_inp_arr[~np.isnan(all_inp_arr).any(axis=1)]
		return inp_arr[0:min(inp_arr.shape[0], self.n_prev), :]


def build_data(io_name, io_dir=IO_DIR, n_prev_games=6):
	"""
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
		# all previous games are potential inputs
		for _, game in data.iterrows():
			prev_games = tgames[game['DateUtc'] > tgames['DateUtc']]
			if prev_games.shape[0] > 0:
				new_game = Game(prev_games, n_prev_games, inp_fields, out_fields, game)


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