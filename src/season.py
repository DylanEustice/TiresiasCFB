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

	def __init__(self, year=default.this_year, sim_type='elo_simple_lin_regress'):
		"""
		sim_type: elo_simple_lin_regress
		"""
		self._year = year
		self._teams = build_all_teams(years=range(2005, year+1))
		self._schedule = util.load_schedule(year=year)
		self._lines = util.load_json('lines.json', fdir=default.comp_team_dir)
		self._sim_type = sim_type

		all_data = util.load_all_dataFrame()
		self._games = all_data[all_data['Season'] == self._year]

		if self._sim_type == 'elo_simple_lin_regress':
			fname = os.path.join(default.elo_dir, 'Linear_Regression_Beta.txt')
			self._B = np.loadtxt(fname)

		if self._sim_type == 'pure_elo':
			self._scores = False
		else:
			self._scores = True

	@property
	def teams(self):
		return self._teams

	def save_week_predictions(self, week):
		"""
		"""
		results = self.sim_week(week)
		fname = 'Predictions_Week{}_{}.json'.format(week, self._sim_type)
		fdir = os.path.join('data', str(self._year))
		util.dump_json(results, fname, fdir=fdir)

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
				home.elos = home.set_elos_to_date(prev_date)
				away = self._teams[self._teams.index(int(away_tid))]
				away.elos = away.set_elos_to_date(prev_date)
			except ValueError:
				# Eiher team not in valid teams
				del results[gid]
				continue
			scores = self._sim_game(home, away, date)
			# Set data
			results[gid]['DateUtc'] = str(date)
			results[gid]['Home_TeamId'] = home_tid
			results[gid]['Away_TeamId'] = away_tid
			results[gid]['Home_TeamName'] = home.name
			results[gid]['Away_TeamName'] = away.name
			# Betting
			results[gid]['Spread'] = self._lines[gid]['Spread']
			results[gid]['OverUnder'] = self._lines[gid]['OverUnder']
			if self._scores:
				results[gid]['Home_Score'] = scores[0]
				results[gid]['Away_Score'] = scores[1]
				spread = 'Home' if scores[0]+results[gid]['Spread'] > scores[1] else 'Away'
				overunder = 'Over' if sum(scores) > results[gid]['OverUnder'] else 'Under'
				results[gid]['Bet'] = [spread, overunder]
			else:
				results[gid]['Home_Pr'] = scores
		return results

	def week_sim_acc(self, week, predictions=None):
		"""
		"""
		# Load predictions
		if predictions is None:
			fname = 'Predictions_Week{}_{}.json'.format(week, self._sim_type)
			fdir = os.path.join('data', str(self._year))
			predictions = util.load_json(fname, fdir=fdir)
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
			corr_game = pred == act
			# Spread
			pred = results['pred_diff_adj'] > 0
			act = results['act_diff_adj'] > 0
			corr_spread = pred == act
			# Overunder
			pred = results['pred_sum_adj'] > 0
			act = results['act_sum_adj'] > 0
			corr_overUnd = pred == act
			# Bias
			bias_home = np.mean(results['act_score'][:,0] - results['pred_score'][:,0])
			bias_away = np.mean(results['act_score'][:,1] - results['pred_score'][:,1])
			bias = np.mean(results['act_score'] - results['pred_score'])
			# Abs error
			abserr_home = np.mean(np.abs(results['act_score'][:,0] - results['pred_score'][:,0]))
			abserr_away = np.mean(np.abs(results['act_score'][:,1] - results['pred_score'][:,1]))
			abserr = np.mean(np.abs(results['act_score'] - results['pred_score']))
			# MSE
			mse_home = np.mean(np.power(results['act_score'][:,0] - results['pred_score'][:,0],2))
			mse_away = np.mean(np.power(results['act_score'][:,1] - results['pred_score'][:,1],2))
			mse = np.mean(np.power(results['act_score'] - results['pred_score'],2))
			# Print
			nGame = len(corr_game)
			print "Straight Up: {:2f} ({} of {})".format(100*np.mean(corr_game),
				sum(corr_game), nGame-sum(corr_game))
			print "     Spread: {:2f} ({} of {})".format(100*np.mean(corr_spread),
				sum(corr_spread), nGame-sum(corr_spread))
			print "  OverUnder: {:2f} ({} of {})".format(100*np.mean(corr_overUnd),
				sum(corr_overUnd), nGame-sum(corr_overUnd))
			print "       Bias: {:2f}, {:2f} (Home), {:2f} (Away))".format(bias, bias_home, bias_away)
			print "    Abs Err: {:2f}, {:2f} (Home), {:2f} (Away))".format(abserr, abserr_home, abserr_away)
			print "        MSE: {:2f}, {:2f} (Home), {:2f} (Away))".format(mse, mse_home, mse_away)
		else:
			corr_game = results['Home_Pr'] > 0.5 == results['act_diff'] > 0
			# Print
			nGame = len(corr_game)
			print "Straight Up: {:2f} ({} of {})".format(100*np.mean(corr_game),
				sum(corr_game), nGame-sum(corr_game))

	def _sim_game(self, home, away, date):
		"""
		"""
		if self._sim_type == 'elo_simple_lin_regress':
			home_elos = home.get_current_elos(next_game_date=date)
			away_elos = away.get_current_elos(next_game_date=date)
			return elo_lin_regress_sim(home_elos, away_elos, self._B)
		elif self._sim_type == 'pure_elo':
			home_elo = home.get_current_elos(next_game_date=date)[0]
			away_elo = away.get_current_elos(next_game_date=date)[0]
			return Pr_elo(home_elo - away_elo)


def elo_lin_regress_sim(team0_elos, team1_elos, B):
	elos = np.hstack([team0_elos, team1_elos])
	scores = np.array(np.matrix(elos) * B)[0]
	return scores