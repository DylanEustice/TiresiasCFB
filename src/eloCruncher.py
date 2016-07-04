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


def assess_elo_confidence(elo_dict, teamgid_map, games, dates_diff, min_season=1, do_print=True):
	"""
	"""
	pr_result = [] # (Pr_win, correct)
	elo_diff = []
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
			Pr_win = max(elo_win_prob(elos))
			correct = np.argmax(elos) == np.argmax(game[3,:])
			pr_result.append((Pr_win, correct))
			elo_diff.append(abs(elos[0]-elos[1]))
	if do_print:
		nExp = sum([pr[0] for pr in pr_result])
		nCorr = len([pr for pr in pr_result if pr[1]])
		pct_correct = 100. * nCorr / len(pr_result)
		pct_diff = 100. * (nExp - nCorr) / nExp
		print "Prediction Correct: {:3.2f} %".format(pct_correct)
		print "  Tally Difference: {:0.2f} %".format(pct_diff)
	return pr_result, elo_diff


def plt_pr_result_hist(pr_result, nBins=6, ax=None, **kwargs):
	"""
	"""
	bins = np.linspace(0.5,1.0,nBins)
	to_hist = []
	for i in range(nBins-1):
		ix = [j for j,pr in enumerate(pr_result) if bins[i] <= pr[0] < bins[i+1]]
		nCorrect = len([j for j in ix if pr_result[j][1]])
		pct_correct = 100. * nCorrect / len(ix)
		to_hist.append(pct_correct)
	if ax is None:
		fig = plt.figure()
		ax = fig.add_subplot(111)
	bin_width = bins[1] - bins[0]
	bins = bins[:-1] + bin_width / 2
	ax.bar(bins, to_hist, width=bin_width, **kwargs)
	ax.grid()
	return ax


def elo_win_prob(elos, div=400):
	"""
	Given elo ratings of teams A and B, calculate their probabilities of winning
	"""
	Pr_A = Pr_elo(elos[0]-elos[1], div=div)
	Pr_B = Pr_elo(elos[1]-elos[0], div=div)
	assert(round(Pr_A+Pr_B,6) == 1)
	return (Pr_A, Pr_B)


def Pr_elo(elo_diff, div=400):
	"""
	Return the probability of a win given elo difference
	"""
	return 1 / (10**( -1.*(elo_diff)/div ) + 1)


def t_test(pr_result, elo_diff):
	"""
	Given population of average elo advantage for winners who were favored,
	determine how well calculated percentages matchup with reality
	t = (X - mu) / (s / sqrt(N))
	X: sample mean
	mu: population mean
	s: sample standard deviation
	N: number of samples
	"""
	N = len(pr_result)
	X = sum([ed for pr,ed in zip(pr_result, elo_diff) if pr[1]]) / N
	mu = sum([x*Pr_elo(x) for x in elo_diff]) / N
	s = np.std([ed for pr,ed in zip(pr_result, elo_diff) if pr[1]])
	return (X-mu) / (s/N**0.5)


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


def run_all_elos(games=[], dates_diff=[], A=4.0, B=4.0, C=0.001, K=20,
	season_regress=0.5, init_elo=1000):
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
				start_elo = curr_elo + season_regress*(init_elo - curr_elo)
				elo_dict[id_][ixSeason].append(start_elo)
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