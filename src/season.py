import src.util as util
from src.team import build_all_teams
import src.default_parameters as default
import numpy as np
from dateutil import parser
import datetime
import pytz
import os


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

	@property
	def teams(self):
		return self._teams

	def save_week_predictions(self, week):
		"""
		"""
		results = self.sim_week(week, *args, **kwargs)
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
			results[gid]['Home_Score'] = scores[0]
			results[gid]['Away_Score'] = scores[1]
			# Betting
			results[gid]['Spread'] = self._lines[gid]['Spread']
			results[gid]['OverUnder'] = self._lines[gid]['OverUnder']
			spread = 'Home' if scores[0]+results[gid]['Spread'] > scores[1] else 'Away'
			overunder = 'Over' if sum(scores) > results[gid]['OverUnder'] else 'Under'
			results[gid]['Bet'] = [spread, overunder]
		return results

	def week_sim_acc(self, week):
		"""
		"""
		# Load predictions
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
		results['pred_score'] = np.zeros([len(gids), 2])
		results['act_score'] = np.zeros([len(gids), 2])
		results['pred_diff_adj'] = np.zeros(len(gids))
		results['act_diff_adj'] = np.zeros(len(gids))
		results['pred_sum_adj'] = np.zeros(len(gids))
		results['act_sum_adj'] = np.zeros(len(gids))
		results['Id'] = []
		results['Home_TeamName'] = []
		results['Away_TeamName'] = []
		results['Home_TeamId'] = []
		results['Away_TeamId'] = []
		for i, gid in enumerate(gids):
			# Get game
			this_pred = predictions[str(int(gid))]
			game = self._games[self._games['Id'].values == gid]
			ixUse = game['is_home'].values.astype(bool)
			# Get predictions
			pred_score = [this_pred['Home_Score'], this_pred['Away_Score']]
			pred_diff_adj = pred_score[0]+this_pred['Spread'] - pred_score[1]
			pred_sum_adj = sum(pred_score) - this_pred['OverUnder']
			# Get actual
			act_score = [game['this_Score'].values[ixUse], game['other_Score'].values[ixUse]]
			act_diff_adj = act_score[0]+this_pred['Spread'] - act_score[1]
			act_sum_adj = sum(act_score) - this_pred['OverUnder']
			# Log
			results['pred_score'][i,:] = pred_score
			results['act_score'][i,:] = act_score
			results['pred_diff_adj'][i] = pred_diff_adj
			results['act_diff_adj'][i] = act_diff_adj
			results['pred_sum_adj'][i] = pred_sum_adj
			results['act_sum_adj'][i] = act_sum_adj
			results['Id'].append(gid)
			results['Home_TeamName'].append(this_pred['Home_TeamName'])
			results['Away_TeamName'].append(this_pred['Away_TeamName'])
			results['Home_TeamId'].append(this_pred['Home_TeamId'])
			results['Away_TeamId'].append(this_pred['Away_TeamId'])
		return results

	def _sim_game(self, home, away, date):
		"""
		"""
		if self._sim_type == 'elo_simple_lin_regress':
			home_elos = home.get_current_elos(next_game_date=date)
			away_elos = away.get_current_elos(next_game_date=date)
			return elo_lin_regress_sim(home_elos, away_elos, self._B)


def elo_lin_regress_sim(team0_elos, team1_elos, B):
	elos = np.hstack([team0_elos, team1_elos])
	scores = np.array(np.matrix(elos) * B)[0]
	return scores