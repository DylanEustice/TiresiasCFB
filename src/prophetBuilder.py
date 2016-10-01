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
from src.dataset import Dataset
import neurolab as nl


class Network:
	def __init__(self, dsname, trainf='train_gdm', lyr=[nl.trans.TanSig(),nl.trans.TanSig()],
		lr=0.0001, epochs=100, update_freq=20, show=20, minmax=1.0, hid_lyr=10, ibias=1.0, norm=True):
		self.ds = Dataset.load(dsname)
		self.norm = norm
		self.set_train_data()
		# training parameters
		self.trainf = getattr(nl.train, trainf)
		self.lyr = lyr
		self.lr = lr
		self.epochs = epochs
		self.update_freq = update_freq
		self.show = show
		self.minmax = minmax
		self.hid_lyr = hid_lyr
		self.ibias = ibias
		# neural network
		self.set_train_kwargs()
		self.net = self.setup_network()

	@classmethod
	def load(network, fname, fdir=default.prm_dir):
		this = network(*[None]*11)
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

	def set_train_data(self):
		if self.norm:
			self.train_inp = self.ds.train_norm_inp
			self.train_tar = self.ds.train_norm_tar
			self.test_inp = self.ds.test_norm_inp
			self.test_tar = self.ds.test_norm_tar
		else:
			self.train_inp = self.ds.train_raw_inp
			self.train_tar = self.ds.train_raw_tar
			self.test_inp = self.ds.test_raw_inp
			self.test_tar = self.ds.test_raw_tar

	def set_train_kwargs(self):
		self.train_kwargs = dict([('epochs', self.update_freq), ('show', self.show)])
		if self.trainf is not nl.train.train_bfgs:
			self.train_kwargs['lr'] = self.lr

	def setup_network(self):
		lyr_rng = zip([0.]*self.train_inp.shape[1], [1.]*self.train_inp.shape[1])
		net = nl.net.newff(lyr_rng, [self.hid_lyr, self.train_tar.shape[1]], self.lyr)
		net.trainf = self.trainf
		for l in net.layers:
			l.initf = nl.init.InitRand([-self.minmax, self.minmax], 'wb')
		net.layers[0].np['b'][:] = self.ibias
		net.errorf = nl.error.MSE()
		return net

	def train_network(self):
		"""
		"""
		error_test = []
		error_train = []
		msef = nl.error.MSE()
		best_net = copy.deepcopy(self.net)
		best_error = float('inf')
		test_data_acc = dict([('inp', self.test_inp), ('tar', self.test_tar)])
		train_data_acc = dict([('inp', self.train_inp), ('tar', self.train_tar)])
		for i in range(self.epochs / self.update_freq):
			error_tmp = self.net.train(self.train_inp, self.train_tar, **self.train_kwargs)
			error_train.append(msef(self.train_tar, self.net.sim(self.train_inp)))
			error_test.append(msef(self.test_tar, self.net.sim(self.test_inp)))
			if error_test[-1] < best_error:
				best_net = copy.deepcopy(self.net)
			print "Iters    : {} / {}".format(i*self.update_freq, self.epochs)
			print "Train    : {}".format(error_train[-1])
			print " Test    : {}".format(error_test[-1])
			print "Train Acc: {}".format(util.get_winner_acc(self.net, self.train_inp, self.train_tar))
			print " Test Acc: {}".format(util.get_winner_acc(self.net, self.test_inp, self.test_tar))
		self.net = best_net
		error = dict([('train',error_train), ('test',error_test)])
		self.plot_sim(error=error)

	def plot_sim(self, error=None, alpha=0.1):
		"""
		"""
		plt.ion()
		if error:
			fig_error = plt.figure()
			x = range(len(error['train']))
			plt.plot(x, error['train'], x, error['test'])
		fig_sim = plt.figure()
		out = (self.net.sim(self.train_inp), self.net.sim(self.test_inp))
		idx = (np.argsort(np.ndarray.flatten(self.train_tar)), 
			   np.argsort(np.ndarray.flatten(self.test_tar)))
		ax1_error = fig_sim.add_subplot(121)
		ax2_error = fig_sim.add_subplot(122)
		# train
		ax1_error.plot(np.ndarray.flatten(self.train_tar)[idx[0]], 'o', alpha=alpha)
		ax1_error.plot(np.ndarray.flatten(out[0])[idx[0]], 'o', alpha=alpha)
		# test
		ax2_error.plot(np.ndarray.flatten(self.test_tar)[idx[1]], 'o', alpha=alpha)
		ax2_error.plot(np.ndarray.flatten(out[1])[idx[1]], 'o', alpha=alpha)



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


def train_network(net, train_inp, train_tar, test_inp, test_tar, prm):
	"""
	Train neural network
	net:	network to be trained
	data:	training and testing data
	prm:	paramters file
	"""
	train = train
	test = test
	error_test = []
	error_train = []
	msef = nl.error.MSE()
	best_net = copy.deepcopy(net)
	best_error = float('inf')
	test_data_acc = dict([('inp', test_inp), ('tar', test_tar)])
	train_data_acc = dict([('inp', train_inp), ('tar', train_tar)])
	for i in range(prm.epochs / prm.update_freq):
		if net.trainf is not nl.train.train_bfgs:
			error_tmp = net.train(train_inp, train_tar, epochs=prm.update_freq,	show=prm.show, lr=prm.lr)
		else:
			error_tmp = net.train(train_inp, train_tar, epochs=prm.update_freq,	show=prm.show)
		error_train.append(msef(train_tar, net.sim(train_inp)))
		error_test.append(msef(test_tar, net.sim(test_inp)))
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
