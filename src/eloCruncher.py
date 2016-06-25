import numpy as np
import pandas as pd
from src.util import *

# global paths
COMP_TEAM_DATA = os.path.join('data', 'compiled_team_data')


class Team:
	def __init__(self, tid, name, games):
		self.tid = tid
		self.name = name
		self.games = games
		self.scores = np.array([games['this_Score'], games['other_Score']])
		self.tids = games['other_TeamId']])
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
	# Load data then filter bad data and FCS games
	all_data = pd.read_pickle(os.path.join(COMP_TEAM_DATA, 'all.df'))
	all_data = all_data[all_data['this_Score'] != '-']
	all_data = all_data[all_data['other_conferenceId'] != '-1']
	# Load team info
	teamid_dict = load_json('team_names.json', fdir='data')
	teamids = sorted(set([tid for tid in all_data['this_TeamId']]))
	teams = []
	for tid in teamids:
		this_name = teamid_dict[str(int(tid))]
		this_games = all_data[all_data['this_TeamId'] == tid]
		if this_games.shape[0] > 0:
			teams.append(Team(tid, this_name, this_games))
	return teams


def rating_adjuster(Ri, A, B, C, K, elo_diff, MoV):
	"""
	Adjust a team's Elo rating based on the outcome of the game
	A, B, C, K:	Parameters
	Ri:			Initial elo rating
	MoV:		Margin of victory (this_score - other_score)
	elo_diff:	Elo delta (elo_winner - elo_loser)
	"""
	MoV_mult = A / (B + C * elo_diff)
	MoV_adj = np.sign(MoV) * np.log(np.abs(MoV) + 1)
	return Ri + K * MoV_adj * MoV_mult


def run_all_elos(teams, init_elo=1000, A=2.2, B=2.2, C=0.001, K=20):
	"""
	"""
	# First interleave all games, sorting by date
	shf_dates = np.concatenate([t.dates for t in teams])
	date_idx = np.argsort(shf_dates)
	scores = np.concatenate([t.scores for t in teams], axis=1)[:,date_idx]
	gids = np.concatenate([t.gids for t in teams])[date_idx]
	tids = np.concatenate([t.tids for t in teams], axis=1)[:,date_idx]
	# Set up elo dictionary
	elo_dict = {}
	for t in teams:
		elo_dict[t.tid] = np.zeros(t.gids.shape[0]+1)
		elo_dict[t.tid][0] = init_elo
	# Walk though games
	for i, gid in enumerate(gids):
		# Recalculate Elos with results of game
		MoV = scores[0,i] - scores[1,i]
		home_team = teams[teams.index(tids[0,i])]
		away_team = teams[teams.index(tids[1,i])]
		home_ixGame = np.where(home_team.gids == gid)[0][0]
		away_ixGame = np.where(away_team.gids == gid)[0][0]
		home_elo = elo_dict[home_team.tid][home_ixGame]
		away_elo = elo_dict[away_team.tid][away_ixGame]
		elo_diff = home_elo - away_elo if MoV > 0 else away_elo - home_elo
		new_home_elo = rating_adjuster(home_elo, A, B, C, K, elo_diff, MoV)
		new_away_elo = rating_adjuster(away_elo, A, B, C, K, elo_diff, -MoV)
		elo_dict[home_team.tid][home_ixGame+1] = new_home_elo
		elo_dict[away_team.tid][away_ixGame+1] = new_away_elo
