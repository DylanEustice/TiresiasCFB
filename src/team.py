from src.util import *
import matplotlib.pyplot as plt
import os
import datetime

# Team name mapping (TEMPORARY HACK)
TEAM_ALT_MAPPING = {
	"Army": "Army West Point", 
	"Southern Mississippi": "Southern Miss", 
	"Central Florida": "UCF", 
	"Middle Tennessee State": "Middle Tennessee", 
	"Brigham Young": "BYU", 
	"Southern California": "USC", 
	"Mississippi": "Ole Miss", 
	"Southern Methodist": "SMU",
	"Texas Christian": "TCU",
	"Troy State": "Troy",
	"Florida International": "FIU",
	"Texas-San Antonio": "UTSA"
}
NO_TEAM = {
	"Massachusetts": range(2000,2012),
	"UAB": range(2015, 2017),
	"Western Kentucky": range(2000,2007),
	"Appalachian State": range(2000,2014),
	"Georgia Southern": range(2000,2014),
	"Texas State": range(2000,2012),
	"Old Dominion": range(2000,2014),
	"South Alabama": range(2000,2012),
	"Georgia State": range(2000,2013),
	"Texas-San Antonio": range(2000,2012),
	"Charlotte": range(2000,2015),
	"South Florida": [2000],
	"Troy State": [2000],
	"Florida International": range(2000,2005),
	"Florida Atlantic": range(2000,2005)
}

class Team:
	def __init__(self, tid, name, games, years, schedule):
		self.tid = tid
		self.name = name
		self.games = games.sort_values(by='DateUtc')
		self.seasons = years
		self.schedule = schedule.sort_values(by='DateUtc')
		self.info = {}
		for year in self.seasons:
			team_dir = os.path.join('data', str(int(year)), 'teams')
			try:
				self.info[year] = load_json(self.name + '.json', fdir=team_dir)
			except:
				if self.name in NO_TEAM and year in NO_TEAM[self.name]:
					continue
				self.info[year] = load_json(TEAM_ALT_MAPPING[self.name] + '.json', fdir=team_dir)
		self._elo_params = {}
		self._elo_params['this_wl_elo'] = np.loadtxt(os.path.join(ELO_DIR, 'Optimal_Winloss_Params.txt'))
		self._elo_params['this_off_elo'] = np.loadtxt(os.path.join(ELO_DIR, 'Optimal_Offdef_Params.txt'))
		self._elo_params['this_def_elo'] = self._elo_params['this_off_elo']
		self._elo_params['this_cf_elo'] = np.loadtxt(os.path.join(ELO_DIR, 'Optimal_Conf_Params.txt'))
		self.elos = self._get_current_elos()

	def __eq__(self, id_):
		return id_ == self.tid or id_ == self.name

	def _get_current_elos(self):
		"""
		"""
		if self.schedule.shape[0]:
			next_game_date = self.schedule['DateUtc'].values[0]
			last_game_date = self.games['DateUtc'].values[-1]
			date_diff = next_game_date - last_game_date
			date_diff = datetime.timedelta(seconds=date_diff.astype('O')/1e9)
			elo_fields = ['this_wl_elo', 'this_cf_elo', 'this_off_elo', 'this_def_elo']
			elos = [self.games[f].values[-1] for f in elo_fields]
			if date_diff > datetime.timedelta(days=100):
				for i, f in enumerate(elo_fields):
					season_regress = self._elo_params[f][4]
					init_elo = self._elo_params[f][5]
					elos[i] = elos[i] + season_regress*(init_elo - elos[i])
		elif self.games.shape[0]:
			elo_fields = ['this_wl_elo', 'this_cf_elo', 'this_off_elo', 'this_def_elo']
			elos = [self.games[f].values[-1] for f in elo_fields]
		else:
			assert False,"Team has no games at all?"
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
		ax.plot(dates, stats, '-o', markerfacecolor=c1, markeredgecolor=c2,
			color=c2, **kwargs)
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
				this_game = dict([('inp',inp_data), ('tar',tar_data)])
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
		all_data = load_all_dataFrame()
	schedule = load_schedule()
	# Load team info
	teamid_dict = load_json('team_names.json', fdir='data')
	teamids = sorted(set([tid for tid in all_data['this_TeamId']]))
	teams = []
	for tid in teamids:
		this_name = teamid_dict[str(int(tid))]
		this_games = all_data[all_data['this_TeamId'] == tid]
		this_schedule = schedule[schedule['this_TeamId'] == tid]
		has_games_years = any([any(this_games['Season'] == year) for year in years])
		if this_games.shape[0] > 0 and has_games_years:
			teams.append(Team(tid, this_name, this_games, years, this_schedule))
	return teams