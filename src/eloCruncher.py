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
			teams.append(Team(tid, this_name, this_games))
	return teams


def tally_elo_accuracy(elo_dict, teamgid_map, games, dates_diff, min_season=1):
	"""
	"""
	tallies = [[],[]] # 0: correct, 1: incorrect
	pr_result = [] # (Pr_win, correct)
	ixSeason = 0
	for i, game in enumerate(games):
		# Check for season gap (100 days)
		if i > 0 and dates_diff[i-1] > 100:
			ixSeason += 1
		# Don't do this for first min_season seasons
		if ixSeason >= min_season:
			# Get team's and their information
			gid = game[0,0]
			tids = [tid for tid in game[1,:]]
			ixGames = [teamgid_map[tid][ixSeason].index(gid) for tid in tids]
			elos = [elo_dict[tid][ixSeason][ix] for tid,ix in zip(tids, ixGames)]
			# Determine win probabilities and winner
			pr = elo_win_prob(elos)
			maxPr = max(pr)
			ixWin = np.argmax(game[3,:])
			correct = np.argmax(elos) == ixWin
			# Append probability of loss if correct, else probability of win
			ixTally = 0 if correct else 1
			add_pr = min(pr) if correct else max(pr)
			tallies[ixTally].append(add_pr)
			pr_result.append((maxPr, correct))
	return tallies, pr_result


def elo_win_prob(elos, div=400):
	"""
	Given elo ratings of teams A and B, calculate their probabilities of winning
	"""
	Pr_A = 1 / (10**( -1.*(elos[0]-elos[1])/div ) + 1)
	Pr_B = 1 / (10**( -1.*(elos[1]-elos[0])/div ) + 1)
	assert(round(Pr_A+Pr_B,6) == 1)
	return (Pr_A, Pr_B)


def rating_adjuster(Ri, A, B, C, K, elo_diff, MoV):
	"""
	Adjust a team's Elo rating based on the outcome of the game
	A, B, C, K:	Parameters
	Ri:			Initial elo rating
	MoV:		Margin of victory (this_score - other_score)
	elo_diff:	Elo delta (elo_winner - elo_loser)
	"""
	assert(B > C * elo_diff)
	MoV_mult = A / (B + C * elo_diff)
	MoV_adj = np.sign(MoV) * np.log(np.abs(MoV) + 1)
	return Ri + K * MoV_adj * MoV_mult


def run_all_elos(games=[], dates_diff=[], init_elo=1000, A=4.0, B=4.0, C=0.001, K=20):
	"""
	"""
	if len(games) == 0 or len(dates_diff) == 0:
		games, dates_diff = build_games()
	# Set up elo dictionary
	tids = np.unique(all_data['this_TeamId'])
	elo_dict = {}
	teamgid_map = {}
	for tid in tids:
		elo_dict[tid] = []
		elo_dict[tid].append([])
		elo_dict[tid][-1].append(init_elo)
		teamgid_map[tid] = []
		teamgid_map[tid].append([])
	# Walk though games
	ixSeason = 0
	for i, game in enumerate(games):
		# Check for season gap (100 days)
		if i > 0 and dates_diff[i-1] > 100:
			ixSeason += 1
			for id_, elo in elo_dict.iteritems():
				curr_elo = elo_dict[id_][-1][-1]
				elo_dict[id_].append([])
				elo_dict[id_][ixSeason].append(curr_elo + 0.5*(init_elo - curr_elo))
				teamgid_map[id_].append([])
		# Recalculate Elos with results of game
		MoV = -np.diff(game[3:,:])
		assert(MoV[0,0] == -MoV[1,0])
		# Get team's and their information
		elos = [elo_dict[tid][ixSeason][-1] for tid in game[1,:]]
		# Calculate parameters based on results
		elo_diff = elos[0] - elos[1] if MoV[0,0] > 0 else elos[1] - elos[0]
		new_elos = [rating_adjuster(elos[i], A, B, C, K, elo_diff, MoV[i,0]) for i in range(2)]
		assert(round(sum(elos),3) == round(sum(new_elos),3))
		# Save
		elo_dict[game[1,0]][ixSeason].append(new_elos[0])
		elo_dict[game[1,1]][ixSeason].append(new_elos[1])
		teamgid_map[game[1,0]][ixSeason].append(game[0,0])
		teamgid_map[game[1,1]][ixSeason].append(game[0,1])
	return elo_dict, teamgid_map, games


def build_games():
	"""
	Build list of unique games and the date differences
	"""
	# Load data
	all_data = load_all_dataFrame()
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
	# Walk though games
	games = []
	for i, gid in enumerate(gids):
		game = data[:,data[0,:]==gid]
		assert(game.shape[1] == 2)
		assert(game[1,:1] == game[2,1] and game[1,1] == game[2,0])
		games.append(game)
	return games, dates_diff