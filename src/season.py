import src.util as util
from src.team import build_all_teams
import src.default_parameters as default
import numpy as np
from dateutil import parser
import datetime
import pytz
import os
from src.eloCruncher import Pr_elo


class Season:

	def __init__(self, dataset, year=default.this_year, sim_type='raw_lin_regress'):
		"""
		sim_type: raw_lin_regress, norm_lin_regress, pure_elo
		"""
		# Season info
		self._year = year
		self._year_dir = os.path.join('data', str(self._year))
		self._teams = build_all_teams(years=range(2005, year+1))
		self._schedule = util.load_schedule(year=year)
		self._lines = util.load_json('lines.json', fdir=self._year_dir)
		self._sim_type = sim_type
		# Data
		self._dataset = dataset
		all_data = util.load_all_dataFrame()
		self._games = all_data[all_data['Season'] == self._year]
		if self._dataset is not None:
			self._name = self._dataset.name + '_' + self._sim_type
		else:
			self._name = self._sim_type
		if self._sim_type == 'pure_elo':
			self._scores = False
		else:
			self._scores = True

	@property
	def teams(self):
		return self._teams

	def _predictions_name(self, week):
		path = os.path.join('Predictions', 'Week_'+str(int(week)), self._name)
		util.ensure_path(os.path.join(self._year_dir, path))
		return os.path.join(path, 'Predictions.json')

	def save_week_predictions(self, week):
		"""
		"""
		results = self.sim_week(week)
		fname = self._predictions_name(week)
		util.dump_json(results, fname, fdir=self._year_dir)

	def sim_week(self, week):
		"""
		"""
		this_week = self._schedule[self._schedule['Week']==week]
		gids = np.unique(this_week['Id'])
		results = dict([(gid,{}) for gid in gids])
		for gid in gids:
			# Get game info
			game = this_week[this_week['Id']==gid]
			date = parser.parse(str(game['DateUtc'].values[0])).replace(tzinfo=pytz.UTC)
			home_tid = game[game['is_home']]['this_TeamId'].values[0]
			away_tid = game[game['is_home']]['other_TeamId'].values[0]
			# Get teams
			try:
				prev_date = date - datetime.timedelta(days=1)
				home = self._teams[self._teams.index(int(home_tid))]
				away = self._teams[self._teams.index(int(away_tid))]
				home.set_elos_to_date(prev_date)
				away.set_elos_to_date(prev_date)
			except ValueError:
				# Eiher team not in valid teams
				del results[gid]
				continue
			scores = self._sim_game(gid, home, away)
			# Set data
			results[gid]['DateUtc'] = str(date)
			results[gid]['Home_TeamId'] = home_tid
			results[gid]['Away_TeamId'] = away_tid
			results[gid]['Home_TeamName'] = home.name
			results[gid]['Away_TeamName'] = away.name
			# Betting
			try:
				results[gid]['Spread'] = self._lines[gid]['Spread']
				results[gid]['OverUnder'] = self._lines[gid]['OverUnder']
			except KeyError:
				results[gid]['Spread'] = np.nan
				results[gid]['OverUnder'] = np.nan
			if self._scores:
				results[gid]['Home_Score'] = scores[0]
				results[gid]['Away_Score'] = scores[1]
				spread = 'Home' if scores[0]+results[gid]['Spread'] > scores[1] else 'Away'
				overunder = 'Over' if sum(scores) > results[gid]['OverUnder'] else 'Under'
				results[gid]['Bet'] = [spread, overunder]
			else:
				results[gid]['Home_Pr'] = scores
		return results

	def week_sim_acc(self, weeks, predictions=None):
		"""
		"""
		# Load predictions
		if predictions is None:
			predictions = {}
			if not isinstance(weeks, list):
				weeks = [weeks]
			for week in weeks:
				fname = self._predictions_name(week)
				new_predictions = util.load_json(fname, fdir=self._year_dir)
				for key, data in new_predictions.iteritems():
					predictions[key] = data
		predicted_gids = [int(gid) for gid in predictions.keys()]
		# Load results
		results_gids = np.unique(self._games['Id'])
		# Find their intersection
		gids = np.intersect1d(predicted_gids, results_gids)
		# Get results
		results = {}
		if self._scores:
			results['pred_score'], results['act_score'] = np.zeros([len(gids), 2]), np.zeros([len(gids), 2])
			results['pred_diff_adj'], results['act_diff_adj'] = np.zeros(len(gids)), np.zeros(len(gids))
			results['pred_sum_adj'], results['act_sum_adj'] = np.zeros(len(gids)), np.zeros(len(gids))
		else:
			results['pred_pct'], results['act_diff'] = np.zeros(len(gids)), np.zeros(len(gids))
		results['Id'] = []
		results['Home_TeamName'], results['Away_TeamName'] = [], []
		results['Home_TeamId'], results['Away_TeamId'] = [], []
		for i, gid in enumerate(gids):
			# Get game
			this_pred = predictions[str(int(gid))]
			game = self._games[self._games['Id'].values == gid]
			ixUse = game['is_home'].values.astype(bool)
			if self._scores:
				# Get predictions
				results['pred_score'][i,:] = [this_pred['Home_Score'], this_pred['Away_Score']]
				results['pred_diff_adj'][i] = results['pred_score'][i,0]+this_pred['Spread'] - results['pred_score'][i,1]
				results['pred_sum_adj'][i] = sum(results['pred_score'][i,:]) - this_pred['OverUnder']
				# Get actual
				results['act_score'][i,:] = [game['this_Score'].values[ixUse], game['other_Score'].values[ixUse]]
				results['act_diff_adj'][i] = results['act_score'][i,0]+this_pred['Spread'] - results['act_score'][i,1]
				results['act_sum_adj'][i] = sum(results['act_score'][i,:]) - this_pred['OverUnder']
			else:
				results['pred_pct'][i] = this_pred['Home_Pr']
				results['act_diff'][i] = game['this_Score'].values[ixUse] - game['other_Score'].values[ixUse]
			# Log
			results['Id'].append(gid)
			results['Home_TeamName'].append(this_pred['Home_TeamName'])
			results['Away_TeamName'].append(this_pred['Away_TeamName'])
			results['Home_TeamId'].append(this_pred['Home_TeamId'])
			results['Away_TeamId'].append(this_pred['Away_TeamId'])
		return results

	def print_week_acc(self, week):
		"""
		"""
		results = self.week_sim_acc(week)
		correct = {}
		# Straight up
		if self._scores:
			pred = results['pred_score'][:,0] > results['pred_score'][:,1]
			act = results['act_score'][:,0] > results['act_score'][:,1]
			diff = np.diff(results['act_score']) - np.diff(results['pred_score'])
			corr_game = pred == act
			# Spread
			ixNumSpread = np.logical_not(np.isnan(results['pred_diff_adj']))
			pred = results['pred_diff_adj'][ixNumSpread] > 0
			act = results['act_diff_adj'][ixNumSpread] > 0
			corr_spread = pred == act
			# Overunder
			ixNumOverUnd = np.logical_not(np.isnan(results['pred_diff_adj']))
			pred = results['pred_sum_adj'][ixNumOverUnd] > 0
			act = results['act_sum_adj'][ixNumOverUnd] > 0
			corr_overUnd = pred == act
			# Bias
			bias_home = np.mean(results['act_score'][:,0] - results['pred_score'][:,0])
			bias_away = np.mean(results['act_score'][:,1] - results['pred_score'][:,1])
			bias = np.mean(diff)
			# Abs error
			abserr_home = np.mean(np.abs(results['act_score'][:,0] - results['pred_score'][:,0]))
			abserr_away = np.mean(np.abs(results['act_score'][:,1] - results['pred_score'][:,1]))
			abserr = np.mean(np.abs(diff))
			# MSE
			mse_home = np.mean(np.power(results['act_score'][:,0] - results['pred_score'][:,0],2))
			mse_away = np.mean(np.power(results['act_score'][:,1] - results['pred_score'][:,1],2))
			mse = np.mean(np.power(diff,2))
			# Print
			nGame = len(corr_game)
			nSpread = sum(ixNumSpread)
			nOverUnd = sum(ixNumOverUnd)
			print "Straight Up: {:2f} ({} and {})".format(100*np.mean(corr_game),
				sum(corr_game), nGame-sum(corr_game))
			print "     Spread: {:2f} ({} and {})".format(100*np.mean(corr_spread),
				sum(corr_spread), nSpread-sum(corr_spread))
			print "  OverUnder: {:2f} ({} and {})".format(100*np.mean(corr_overUnd),
				sum(corr_overUnd), nOverUnd-sum(corr_overUnd))
			print "       Bias: {:2f}, {:2f} (Home), {:2f} (Away))".format(bias, bias_home, bias_away)
			print "    Abs Err: {:2f}, {:2f} (Home), {:2f} (Away))".format(abserr, abserr_home, abserr_away)
			print "        MSE: {:2f}, {:2f} (Home), {:2f} (Away))".format(mse, mse_home, mse_away)
		else:
			home_pred = results['pred_pct'] > 0.5
			home_act = results['act_diff'] > 0
			corr_game = [hp==ha for hp,ha in zip(home_pred, home_act)]
			# Print
			nGame = len(corr_game)
			print "Straight Up: {:2f} ({} and {})".format(100*np.mean(corr_game),
				sum(corr_game), nGame-sum(corr_game))

	def _sim_game(self, gid, home, away):
		"""
		"""
		if self._sim_type == 'norm_lin_regress':
			inp = self._dataset.games_data[float(gid)]['norm_inp']
			B = self._dataset.B_norm
			scores = np.array(inp * B)[0]
			scores,_ = self._dataset._norm_func(scores, params=self._dataset.tar_norm_params, do_norm=False)
			return scores
		if self._sim_type == 'raw_lin_regress':
			inp = self._dataset.games_data[float(gid)]['raw_inp']
			B = self._dataset.B_raw
			scores = np.array(inp * B)[0]
			return scores
		elif self._sim_type == 'pure_elo':
			try:
				return Pr_elo(home.elos[0] - away.elos[0])
			except:
				import pdb; pdb.set_trace()