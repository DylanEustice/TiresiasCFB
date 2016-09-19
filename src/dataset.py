import src.default_parameters as default
import src.util as util
from src.team import build_all_teams
import numpy as np
import pandas as pd
import datetime
import pickle
import os


class Dataset:
	def __init__(self, name, name_dir=default.data_dir):
		# Optionally load in parameters
		if name is None:
			return
		self._name = name
		self._info = util.load_json(name+'.json', fdir=name_dir)
		# Params
		self.min_date = datetime.datetime(*self._info['min_date'])
		self.max_date = datetime.datetime(*self._info['max_date'])
		self.min_games = self._info['min_games']
		self.home_only = self._info['home_only']
		self.date_diff = datetime.timedelta(weeks=self._info['date_diff'])
		self.train_pct = self._info['train_pct']
		# Games / teams
		self._games = util.load_all_dataFrame()
		self._teams = build_all_teams(all_data=self._games)
		# I/O fields
		io_fields = util.load_json(self._info['io_name'], fdir=default.io_dir)
		self.inp_fields = io_fields['inputs']
		self.tar_fields = io_fields['outputs']
		# Averaging
		if self._info['avg_func'] == 'mean':
			self.avg_func = util.elo_mean
			self.avg_func_args = [self.inp_fields]
			self.avg_func_kwargs = dict([('elo_fields', default.all_elo_fields)])
		else:
			assert False, "Unrecognized averaging function!"
		# Normalization
		if self._info['norm_func'] == 'standardize_data':
			self._norm_func = util.standardize_data
		elif self._info['norm_func'] == 'normalize_data':
			self._norm_func = util.normalize_data
		else:
			assert False, "Unrecognized normalizing function!"
		# Build
		self._build_dataset()


	@classmethod
	def load(dataset, name, fdir=default.data_dir):
		this = dataset(*[None])
		try:
			with open(os.path.join(fdir, name+'.ds'),"r") as f:
				this = pickle.load(f)
		except IOError:
			this = Dataset(name)
		return this


	def save(self, fdir=default.data_dir):
		with open(os.path.join(fdir, self._name+'.ds'),"w") as f:
			pickle.dump(self, f)


	def _build_dataset(self):
		self._build_train_games()
		self._set_full_dataset()
		self._partition_dataset()


	def _build_train_games(self):
	# Extract game training data
		self._train_games = []
		teams_dict = dict([(t.tid, t) for t in self._teams])
		for t in self._teams:
			print t.tid
			self._train_games.extend(t.get_training_data(self, teams_dict))


	def _set_full_dataset(self):
		self.all_raw_inp = np.vstack([g['inp'] for g in self._train_games]).astype('double')
		self.all_raw_tar = np.vstack([g['tar'] for g in self._train_games]).astype('double')
		self.all_norm_inp, self.inp_norm_params = self._norm_func(self.all_raw_inp)
		self.all_norm_tar, self.tar_norm_params = self._norm_func(self.all_raw_tar)


	def _partition_dataset(self):
		# Get partition indeces
		gids = np.unique([g['id'] for g in self._train_games])
		rnd_vec = np.random.random(gids.shape[0])
		train_idx = rnd_vec < self.train_pct
		train_gids = {id_ for i, id_ in enumerate(gids) if train_idx[i]}
		ixTrain = np.array([g['id'] in train_gids for g in self._train_games])
		ixTest = np.logical_not(ixTrain)
		# Partition train data
		self.train_norm_inp = self.all_norm_inp[ixTrain,:]
		self.train_norm_tar = self.all_norm_tar[ixTrain,:]
		self.train_raw_inp = self.all_raw_inp[ixTrain,:]
		self.train_raw_tar = self.all_raw_tar[ixTrain,:]
		# Partition test data
		self.test_norm_inp = self.all_norm_inp[ixTest,:]
		self.test_norm_tar = self.all_norm_tar[ixTest,:]
		self.test_raw_inp = self.all_raw_inp[ixTest,:]
		self.test_raw_tar = self.all_raw_tar[ixTest,:]


	def print_date_range(self):
		print "{} to {}".format(self._min_date, self._max_date)


	def print_io_fields(self):
		print "Input:"
		for f in self._inp_fields:
			print "  {}".format(f)
		print "Output:"
		for f in self._tar_fields:
			print "  {}".format(f)