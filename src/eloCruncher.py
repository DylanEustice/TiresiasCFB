import numpy as np
import pandas as pd
from src.util import *

# global paths
COMP_TEAM_DATA = os.path.join('data', 'compiled_team_data')


class Team:
	def __init__(self, tid, name, games):
		self.tid = tid
		self.name = teamid_dict[str(int(tid))]
		self.games = games
		self.scores = np.array([games['this_Score'], games['other_Score']])
		self.game_idx = np.array(games['Id'])

	def get_game(self, gid):
		if gid in self.game_idx:
			return self.scores[:,self.game_idx==gid]
		else:
			print "Game ID {} not found for {}".format(gid, self.name)
			return None


def build_all_teams():
	"""
	Builds a list of teams (entries are Team class) for all compiled data
	"""
	teamid_dict = load_json('team_names.json', fdir='data')
	all_data = pd.read_pickle(os.path.join(COMP_TEAM_DATA, 'all.df'))
	teamids = sorted(set([tid for tid in all_data['this_TeamId']]))
	teams = []
	for tid in teamids:
		this_name = teamid_dict[str(int(tid))]
		this_games = all_data[all_data['this_TeamId'] == tid]
		teams.append(Team(tid, this_name, this_games))
	return teams
