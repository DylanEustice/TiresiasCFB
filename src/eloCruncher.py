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
		self.tids = np.array([games['this_TeamId'], games['other_TeamId']])
		self.gids = np.array(games['Id'])
		self.dates = np.sort(games['DateUtc'])

	def __eq__(self, tid):
		return tid == self.tid

	def get_game(self, gid):
		if gid in self.gids:
			return self.scores[:,self.gids==gid]
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


def rating_adjuster(Ri, A, B, C, K, elo_diff, MoV):
	"""
	Adjust a team's Elo rating based on the outcome of the game
	A, B, C, K:	Parameters
	Ri:			Initial rating
	MoV:		Margin of victory (this_score - other_score)
	elo_diff:	Elo delta (elo_winner - elo_loser)
	"""
	MoV_mult = A / (B + C * elo_diff)
	MoV_adj = np.sign(MoV) * np.log(np.abs(MoV) + 1)
	return Ri + K * MoV_adj * MoV_mult


def run_all_elos(init_elo, teams, A, B, C, K):
	"""
	"""
	# First interleave all games, sorting by date
	shf_dates = np.concatenate([t.dates for t in teams])
	date_idx = np.argsort(shf_dates)
	scores = np.concatenate([t.scores for t in teams], axis=1)[:,date_idx]
	gids = np.concatenate([t.gids for t in teams], axis=1)[:,date_idx]
	tids = np.concatenate([t.tids for t in teams], axis=1)[:,date_idx]

