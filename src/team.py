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
		if gid in self.gids:
			return self.scores[:,self.gids==gid]
		else:
			print "Game ID {} not found for {}".format(gid, self.name)
			return None

	def plot_stat(self, field, ax=None, **kwargs):
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(111)
		c1 = self.info['PrimaryColor']
		c2 = self.info['SecondaryColor']
		stats = np.array(self.games[field])
		dates = np.array(self.games['DateUtc'])
		ax.plot(dates, stats, '-o', markerfacecolor=c1, markeredgecolor=c2, color=c2, **kwargs)
		ax.grid('on')
		return ax


def build_all_teams(year=2015):
	"""
	Builds a list of teams (entries are Team class) for all compiled data
	"""
	# Load data
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