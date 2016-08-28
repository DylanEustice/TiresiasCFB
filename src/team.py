import src.util as util
import matplotlib.pyplot as plt
import os
import datetime
import src.default_parameters as default
import numpy as np
from dateutil import parser
import pytz


class Team:
	def __init__(self, tid, name, games, schedule):
		self.tid = tid
		self.name = name
		self.games = games.sort_values(by='DateUtc')
		self.seasons = np.unique(self.games['Season'])
		self.schedule = schedule.sort_values(by='DateUtc')
		self.info = {}
		for year in self.seasons:
			team_dir = os.path.join('data', str(int(year)), 'teams')
			try:
				self.info[year] = util.load_json(self.name + '.json', fdir=team_dir)
			except IOError:
				self.info[year] = util.load_json(default.team_alt_mapping[self.name] + '.json', fdir=team_dir)
		self._elo_params = {}
		self._elo_params['this_wl_elo'] = np.loadtxt(os.path.join(default.elo_dir, 'Optimal_Winloss_Params.txt'))
		self._elo_params['this_off_elo'] = np.loadtxt(os.path.join(default.elo_dir, 'Optimal_Offdef_Params.txt'))
		self._elo_params['this_def_elo'] = self._elo_params['this_off_elo']
		self._elo_params['this_cf_elo'] = np.loadtxt(os.path.join(default.elo_dir, 'Optimal_Conf_Params.txt'))
		self._elo_params['this_poff_elo'] = np.loadtxt(os.path.join(default.elo_dir, 'Optimal_PassYd_Params.txt'))
		self._elo_params['this_pdef_elo'] = self._elo_params['this_poff_elo']
		self._elo_params['this_roff_elo'] = np.loadtxt(os.path.join(default.elo_dir, 'Optimal_RushYd_Params.txt'))
		self._elo_params['this_rdef_elo'] = self._elo_params['this_roff_elo']
		self.elos = self.get_current_elos()

	def __eq__(self, id_):
		return id_ == self.tid or id_ == self.name

	def get_current_elos(self, next_game_date="schedule"):
		"""
		"""
		elo_fields = default.this_elo_fields

		if next_game_date == "schedule" and self.schedule.shape[0]:
			next_game_date = self.schedule['DateUtc'].values[0]
		elif next_game_date == "schedule":
			elos = [self.games[f].values[-1] for f in elo_fields]
			return elos
		else:
			assert isinstance(next_game_date, datetime.datetime), "Should be datetime"

		ix_last_game = self._get_prev_game_ix_before_date(next_game_date)
		last_game_date = self.games['DateUtc'].values[ix_last_game]
		# Make sure these are datetime objects (super annoying)
		next_game_date = parser.parse(str(next_game_date)).replace(tzinfo=pytz.UTC)
		last_game_date = parser.parse(str(last_game_date)).replace(tzinfo=pytz.UTC)
		date_diff = (next_game_date - last_game_date).total_seconds() / (24*60*60)
		elos = [self.games[f].values[ix_last_game] for f in elo_fields]
		if date_diff > default.season_day_sep:
			elos = self._regress_elos(elos)

		return elos

	def _get_prev_game_ix_before_date(self, date):
		prev_games = self.games['DateUtc'] < date
		return np.where(prev_games)[0][-1]

	def _regress_elos(self, elos):
		elo_fields = default.this_elo_fields
		for i, f in enumerate(elo_fields):
			season_regress = self._elo_params[f][4]
			init_elo = self._elo_params[f][5]
			elos[i] = elos[i] + season_regress*(init_elo - elos[i])
		return elos


	def get_game(self, gid):
		"""
		Returns stats for a game id
		"""
		if gid in self.gids:
			return self.scores[:,self.gids==gid]
		else:
			print "Game ID {} not found for {}".format(gid, self.name)
			return None

	def plot_stat(self, field, ax=None, **kwargs):
		"""
		Plots a stat given by field
		"""
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(111)
		c1 = self.info[self.seasons[-1]]['PrimaryColor']
		c2 = self.info[self.seasons[-1]]['SecondaryColor']
		stats = np.array(self.games[field])
		dates = np.array(self.games['DateUtc'])
		ax.plot(dates, stats, '-o', markerfacecolor=c1, markeredgecolor=c2, color=c2, **kwargs)
		ax.grid('on')
		return ax

	def get_training_data(self, prm, teams_dict, inp_fields, tar_fields, **kwargs):
		"""
		"""
		min_games_in_season = 8
		# Get games after minimum date
		games = []
		ixValid = np.array(self.games['DateUtc'] > prm.min_date)
		valid_games = self.games[ixValid]
		for _, g in valid_games.iterrows():
			if prm.min_games > 0:
				# Don't repeat games if selected
				if prm.home_only and not g['is_home']:
					continue
				# Get games within a certain previous range
				this_prev_games = get_games_in_range(self.games, g['DateUtc'], prm.date_diff)
				# From previous games, build data array
				if this_prev_games.shape[0] < prm.min_games:
					continue
				# Get data from other team
				other_tid = g['other_TeamId']
				other_team = teams_dict[other_tid]
				other_prev_games = get_games_in_range(other_team.games, g['DateUtc'], prm.date_diff)
				# Other team must also have enough games
				if other_prev_games.shape[0] < prm.min_games:
					continue
				# Build data
				this_inp_data_all = build_data_from_games(this_prev_games, inp_fields)
				this_inp_data = prm.inp_avg(this_inp_data_all, **kwargs)
				other_inp_data_all = build_data_from_games(other_prev_games, inp_fields)
				other_inp_data = prm.inp_avg(other_inp_data_all, **kwargs)
				inp_data = np.hstack([this_inp_data, other_inp_data])
				tar_data = build_data_from_games(g, tar_fields)
				if tar_data.shape[0] > 1:
					tar_data = tar_data.reshape(1,tar_data.shape[0])
				this_game = dict([('inp',inp_data), ('tar',tar_data), ('id',g['Id'])])
				games.append(this_game)
		return games


def build_data_from_games(games, fields):
	"""
	"""
	data = np.asarray(games[fields].replace('-', np.nan), float)
	if len(data.shape) == 1:
		data = data.reshape([data.shape[0], 1])
	data = data[~np.isnan(data).any(axis=1)]
	return data


def get_games_in_range(games, curr_date, max_diff):
	"""
	Get games within a certain previous range
	"""
	date_diff = curr_date - games['DateUtc']
	prior_games = date_diff > datetime.timedelta(0)
	recent_games = date_diff < max_diff
	valid_dates = np.logical_and(prior_games, recent_games)
	return games[valid_dates]


def build_all_teams(years=range(2005,2017), all_data=None):
	"""
	Builds a list of teams (entries are Team class) for all compiled data
	"""
	# Load data
	if all_data is None:
		all_data = util.load_all_dataFrame()
	schedule = util.load_schedule()
	# Load team info
	teamid_dict = util.load_json('team_names.json', fdir='data')
	teamids = sorted(set([tid for tid in all_data['this_TeamId']]))
	teams = []
	for tid in teamids:
		this_name = teamid_dict[str(int(tid))]
		this_games = all_data[all_data['this_TeamId'] == tid]
		this_schedule = schedule[schedule['this_TeamId'] == tid]
		has_games_years = any([any(this_games['Season'] == year) for year in years])
		if this_games.shape[0] > 0 and has_games_years:
			teams.append(Team(tid, this_name, this_games, this_schedule))
	return teams