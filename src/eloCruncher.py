import numpy as np
import pandas as pd
import datetime
from src.util import *

# global paths
COMP_TEAM_DATA = os.path.join('data', 'compiled_team_data')


class Team:
	def __init__(self, tid, name, games):
		self.tid = tid
		self.name = name
		self.games = games
		self.scores = np.array([games['this_Score'], games['other_Score']])
		self.tids = np.array([games['this_TeamId'], games['other_TeamId']])
		self.gids = np.array(games['Id'])
		self.dates = np.sort(games['DateUtc'])
		self.elo = []

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


def run_all_elos(all_data, init_elo=1000, A=4.0, B=4.0, C=0.001, K=20):
	"""
	"""
	# Sort unique game ids by date
	gids, ix_gids = np.unique(all_data['Id'], return_index=True)
	ix_date = np.argsort(np.array(all_data['DateUtc'])[ix_gids])
	gids = gids[ix_date]
	data = np.array([all_data['Id'], all_data['this_TeamId'], all_data['other_TeamId'],
		all_data['this_Score'], all_data['other_Score']])
	# Find time between games
	dates = np.array([d for d in all_data['DateUtc']])[ix_gids][ix_date]
	dates_diff = np.array([(dates[i+1]-dates[i]).total_seconds() for i in range(len(dates)-1)])
	dates_diff /= (60*60*24)
	# Set up elo dictionary
	tids = np.unique(all_data['this_TeamId'])
	elo_dict = {}
	for tid in tids:
		elo_dict[tid] = []
		elo_dict[tid].append(init_elo)
	# Walk though games
	for i, gid in enumerate(gids):
		# Check for season gap (100 days)
		if dates_diff[i-1] > 100:
			for id_, elo in elo_dict.iteritems():
				elo_dict[id_][-1] += (init_elo - elo_dict[id_][-1]) / 2
		# Find game
		game = data[:,data[0,:]==gid]
		assert(game.shape[1] == 2)
		assert(game[1,:1] == game[2,1] and game[1,1] == game[2,0])
		# Recalculate Elos with results of game
		MoV = -np.diff(game[3:,:])
		assert(MoV[0,0] == -MoV[1,0])
		# Get team's and their information
		elos = [elo_dict[tid][-1] for tid in game[1,:]]
		# Calculate parameters based on results
		elo_diff = elos[0] - elos[1] if MoV[0,0] > 0 else elos[1] - elos[0]
		new_elos = [rating_adjuster(elos[i], A, B, C, K, elo_diff, MoV[i,0]) for i in range(2)]
		# Save
		elo_dict[game[1,0]].append(new_elos[0])
		elo_dict[game[1,1]].append(new_elos[1])
	return elo_dict
