from src.util import *
import matplotlib.pyplot as plt
import os

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
	"Massachusetts": range(2005,2012),
	"UAB": [2015],
	"Western Kentucky": range(2005,2007),
	"Appalachian State": range(2005,2014),
	"Georgia Southern": range(2005,2014),
	"Texas State": range(2005,2012),
	"Old Dominion": range(2005,2014),
	"South Alabama": range(2005,2012),
	"Georgia State": range(2005,2013),
	"Texas-San Antonio": range(2005,2012),
	"Charlotte": range(2005,2015),
}

class Team:
	def __init__(self, tid, name, games, years):
		self.tid = tid
		self.name = name
		self.games = games
		self.seasons = years
		self.info = {}
		for year in self.seasons:
			team_dir = os.path.join('data', str(int(year)), 'teams')
			try:
				self.info[year] = load_json(self.name + '.json', fdir=team_dir)
			except:
				if self.name in NO_TEAM and year in NO_TEAM[self.name]:
					continue
				self.info[year] = load_json(TEAM_ALT_MAPPING[self.name] + '.json', fdir=team_dir)

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


def build_all_teams(years=range(2005,2016)):
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
		has_games_years = any([any(this_games['Season'] == year) for year in years])
		if this_games.shape[0] > 0 and has_games_years:
			teams.append(Team(tid, this_name, this_games, years))
	return teams