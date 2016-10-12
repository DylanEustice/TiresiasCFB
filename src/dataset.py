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
		self.name = name
		self._info = util.load_json(name+'.json', fdir=name_dir)
		# Params
		self.min_date = datetime.datetime(*self._info['min_date'])
		self.max_date = datetime.datetime(*self._info['max_date'])
		self.min_games = self._info['min_games']
		self.home_only = self._info['home_only']
		self.date_diff = datetime.timedelta(weeks=self._info['date_diff'])
		self.train_pct = self._info['train_pct']
		self.max_train_date = datetime.datetime(*self._info['max_train_date'])
		# Games / teams
		self._games = util.load_all_dataFrame()
		self._teams = build_all_teams(all_data=self._games)
		# I/O fields
		io_fields = util.load_json(self._info['io_name'], fdir=default.io_dir)
		self.inp_fields = io_fields['inputs']
		self.tar_fields = io_fields['outputs']
		self.inp_post = io_fields['input_postprocess'] if 'input_postprocess' in io_fields else None
		self.tar_post = io_fields['output_postprocess'] if 'output_postprocess' in io_fields else None
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
		self._build_raw_games()
		self._postprocess_games()
		self._set_games_data()
		self._set_full_dataset()
		self._partition_dataset()
		self._calculate_linear_regression()

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
		with open(os.path.join(fdir, self.name+'.ds'),"w") as f:
			pickle.dump(self, f)

	def _axis_isnan(self, A):
		return np.any(np.isnan(A), axis=1)

	def _build_raw_games(self):
	# Extract game training data
		self._raw_games = []
		teams_dict = dict([(t.tid, t) for t in self._teams])
		for t in self._teams:
			print t.tid
			self._raw_games.extend(t.get_training_data(self, teams_dict, use_schedule=False))
			self._raw_games.extend(t.get_training_data(self, teams_dict, use_schedule=True))

	def _postprocess_games(self):
		for game in self._raw_games:
			game['inp'] = self._postprocess_arr(self.inp_post, game['inp'][0,:])
			game['tar'] = self._postprocess_arr(self.tar_post, game['tar'][0,:])

	def _postprocess_arr(self, post_arr, orig_arr):
		if not post_arr:
			return orig_arr
		new_arr = np.zeros([1, len(post_arr)])
		for i, post in enumerate(post_arr):
			eval_str = self._gen_eval_str_from_post(orig_arr, post)
			new_arr[0,i] = eval(eval_str)
		return new_arr

	def _gen_eval_str_from_post(self, arr, post):
		s = ''
		for p in post:
			try:
				s += p
			except TypeError:
				s += str(arr[p])
		return s

	def _set_games_data(self):
		self.games_data = {}
		game_to_dict = lambda g: dict([('inp', g['inp']), ('tar', g['tar'])])
		for g in self._raw_games:
			self.games_data[g['id']] = game_to_dict(g)

	def _set_full_dataset(self):
		self.all_raw_inp = np.vstack([g['inp'] for g in self._raw_games]).astype('double')
		self.all_raw_tar = np.vstack([g['tar'] for g in self._raw_games]).astype('double')
		dates = np.array([g['date'] for g in self._raw_games])
		not_or = lambda A, B: np.logical_not(np.logical_or(A, B))
		ixValid = not_or(self._axis_isnan(self.all_raw_inp), self._axis_isnan(self.all_raw_tar))
		ixValid = np.logical_and)=(ixValid, dates < self.max_train_date)
		_, self.inp_norm_params = self._norm_func(self.all_raw_inp[ixValid,:])
		self.all_norm_inp, _ = self._norm_func(self.all_raw_inp, params=self.inp_norm_params)
		_, self.tar_norm_params = self._norm_func(self.all_raw_tar[ixValid,:])
		self.all_norm_tar, _ = self._norm_func(self.all_raw_tar, params=self.tar_norm_params)

	def _partition_dataset(self):
		# Get partition indeces
		gids = np.unique([g['id'] for g in self._raw_games])
		rnd_vec = np.random.random(gids.shape[0])
		train_idx = rnd_vec < self.train_pct
		train_gids = {id_ for i, id_ in enumerate(gids) if train_idx[i]}
		ixTrain = np.array([g['id'] in train_gids for g in self._raw_games])
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

	def _calculate_linear_regression(self):
		ixRawNaN = np.logical_or(self._axis_isnan(self.train_raw_inp),
								 self._axis_isnan(self.train_raw_tar))
		ixNormNaN = np.logical_or(self._axis_isnan(self.train_norm_inp),
								  self._axis_isnan(self.train_norm_tar))
		ixValid = np.logical_not(np.logical_or(ixRawNaN, ixNormNaN))
		self.B_raw,_,_,_ = np.linalg.lstsq(self.train_raw_inp[ixValid,:], self.train_raw_tar[ixValid,:])
		self.B_raw = np.matrix(self.B_raw)
		self.B_norm,_,_,_ = np.linalg.lstsq(self.train_norm_inp[ixValid,:], self.train_norm_tar[ixValid,:])
		self.B_norm = np.matrix(self.B_norm)

	def print_date_range(self):
		print "{} to {}".format(self._min_date, self._max_date)

	def print_io_fields(self):
		print "Input:"
		for f in self._inp_fields:
			print "  {}".format(f)
		print "Output:"
		for f in self._tar_fields:
			print "  {}".format(f)