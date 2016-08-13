from src.util import *
import matplotlib.pyplot as plt
import os

# Team name mapping (TEMPORARY HACK)
team_alt_mapping = {"Army": "Army West Point", 
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
					"Texas-San Antonio": "UTSA"}

class Team:
	def __init__(self, tid, name, games, year):
		self.tid = tid
		self.name = name
		self.games = games
		self.curr_year = year
		team_dir = os.path.join('data', str(int(year)), 'teams')
		try:
			self.info = load_json(self.name + '.json', fdir=team_dir)
		except:
			self.info = load_json(team_alt_mapping[self.name] + '.json', fdir=team_dir)

	def __eq__(self, id_):
		return id_ == self.tid or id_ == self.name

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
		c1 = self.info['PrimaryColor']
		c2 = self.info['SecondaryColor']
		stats = np.array(self.games[field])
		dates = np.array(self.games['DateUtc'])
		ax.plot(dates, stats, '-o', markerfacecolor=c1, markeredgecolor=c2,
			color=c2, **kwargs)
		ax.grid('on')
		return ax

	def get_training_data(self, prm, teams_dict, this_inp_fields, other_inp_fields, 
		tar_fields):
		"""
		"""
		# Get games after minimum date
		ixValid = np.array(self.games['DateUtc'] > prm.min_date)
		valid_games = self.games[ixValid]
		for _, g in valid_games.iterrows():
			if prm.min_games > 0:
				# Don't repeat games if selected
				if prm.home_only and g['is_home'].values[0] == 0:
					continue
				# Get games within a certain previous range
				this_prev_games = get_games_in_range(
					self.games, g['DateUtc'].values[0], prm.date_diff)
				# From previous games, build data array
				if this_prev_games.shape[0] < prm.min_games:
					continue
				# Get data from other team
				other_tid = g['other_TeamId'].values[0]
				other_team = teams_dict[other_tid]
				other_prev_games = get_games_in_range(
					other_team.games, g['DateUtc'].values[0], prm.date_diff)
				# Build data
				this_inp_data = build_data_from_games(this_prev_games, this_inp_fields)
				other_inp_data = build_data_from_games(other_prev_games, other_inp_fields)
				tar_data = build_data_from_games(g, tar_fields)

	def build_data_from_games(games, fields):
		"""
		"""
		games.replace('-', np.nan)
		data = np.asarray(games[fields], float)
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



def build_all_teams(year=2015, all_data=None):
	"""
	Builds a list of teams (entries are Team class) for all compiled data
	"""
	# Load data
	if all_data is None:
		all_data = load_all_dataFrame()
	# Load team info
	teamid_dict = load_json('team_names.json', fdir='data')
	teamids = sorted(set([tid for tid in all_data['this_TeamId']]))
	teams = []
	for tid in teamids:
		this_name = teamid_dict[str(int(tid))]
		this_games = all_data[all_data['this_TeamId'] == tid]
		if this_games.shape[0] > 0:
			shift_year = 0
			while np.all(this_games['Season'] != year-shift_year):
				shift_year += 1
			teams.append(Team(tid, this_name, this_games, year-shift_year))
	return teams