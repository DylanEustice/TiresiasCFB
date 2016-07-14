import numpy as np
import pandas as pd
import datetime
from src.util import *
from src.searchAlgorithms import evolutionary_search
import matplotlib.pyplot as plt
import os

# global paths
COMP_TEAM_DATA = os.path.join('data', 'compiled_team_data')

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

	def __eq__(self, tid):
		return tid == self.tid

	def get_game(self, gid):
		if gid in self.gids:
			return self.scores[:,self.gids==gid]
		else:
			print "Game ID {} not found for {}".format(gid, self.name)
			return None


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


def run_evolutionary_elo_search(nPop=10, iters=10, kill_rate=0.5, evolve_rng=0.5,
	min_season=1, elo_type="winloss"):
	"""
	"""
	all_data = load_all_dataFrame()
	games, dates_diff = build_games(all_data)
	if elo_type == "winloss":
		init_params = [4, 4, 1e-3, 9.5, 0.5, 1000]
	elif elo_type == "offdef":
		init_params = [4, 4, 1e-3, 7.5, 0.5, 1000]
	args = [all_data, games, dates_diff]
	kwargs = dict([('min_season',min_season), ('elo_type',elo_type)])
	return evolutionary_search(nPop, iters, kill_rate, evolve_rng,
		elo_obj_fun, init_params, *args, **kwargs)


def elo_obj_fun(params, all_data, games, dates_diff, min_season=1, elo_type="winloss"):
	"""
	params: A, B, C, K, season_regress, init_elo
	"""
	A, B, C, K, season_regress, init_elo = params
	if K <= 1e-6:
		return np.inf
	elo_dict, teamgid_map = run_all_elos(all_data, games=games,
		dates_diff=dates_diff, elo_type=elo_type, A=A, B=B, C=C, K=K,
		season_regress=season_regress, init_elo=init_elo)
	pr_result, elo_diff = assess_elo_confidence(elo_dict, teamgid_map, games, dates_diff,
		min_season=min_season, do_print=False, elo_type=elo_type)
	return abs(t_test(pr_result, elo_diff))


def assess_elo_confidence(elo_dict, teamgid_map, games, dates_diff, min_season=1,
	do_print=True, elo_type="winloss", avg_score=26.9):
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
			if elo_type == "winloss":
				Pr_win = max(elo_game_probs(elos))
				correct = np.argmax(elos) == np.argmax(game[3,:])
				pr_result.append((Pr_win, correct))
				elo_diff.append(abs(elos[0]-elos[1]))
			elif elo_type == "offdef":
				# Do for each offense-defense matchup seperately
				for j in range(2):
					offdef_elos = [elos[j][0], elos[1-j][1]]
					Pr_win = max(elo_game_probs(offdef_elos))
					ixMax_elo = np.argmax(offdef_elos)
					correct = ((ixMax_elo==1 and game[3,j] < avg_score) or
							   (ixMax_elo==0 and game[3,j] >= avg_score))
					pr_result.append((Pr_win, correct))
					elo_diff.append(abs(offdef_elos[0]-offdef_elos[1]))
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
		if len(ix) > 0:
			pct_correct = 100. * nCorrect / len(ix)
		else:
			pct_correct = -1
		to_hist.append(pct_correct)
	if ax is None:
		fig = plt.figure()
		ax = fig.add_subplot(111)
	bin_width = bins[1] - bins[0]
	bins = bins[:-1] + bin_width / 2
	ax.bar(bins, to_hist, width=bin_width, **kwargs)
	ax.grid()
	return ax


def elo_game_probs(elos, div=400):
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


def rating_adjuster(Ri, A, B, C, K, elo_diff, MoV, max_MoV_mult=1e3):
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


def run_all_elos(all_data, games=[], dates_diff=[], A=4.0, B=4.0, C=0.001, K=20,
	season_regress=0.5, init_elo=1000, elo_type="winloss", avg_score=26.9):
	"""
	elo_type: "winloss", "offdef"
	"""
	max_elo_diff = B / C - 1
	if len(games) == 0 or len(dates_diff) == 0:
		games, dates_diff = build_games()
	# Set up elo dictionary
	tids = np.unique(all_data['this_TeamId'])
	elo_dict = {}
	teamgid_map = {}
	for tid in tids:
		elo_dict[tid] = []
		elo_dict[tid].append([])
		if elo_type == "winloss":
			elo_dict[tid][-1].append(init_elo)
		elif elo_type == "offdef":
			elo_dict[tid][-1].append((init_elo, init_elo))
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
				if elo_type == "winloss":
					start_elo = curr_elo + season_regress*(init_elo - curr_elo)
				elif elo_type == "offdef":
					start_elo = tuple(e + season_regress*(init_elo - e) for e in curr_elo)
				elo_dict[id_][ixSeason].append(start_elo)
				teamgid_map[id_].append([])
		# Recalculate Elos with results of game
		if elo_type == "winloss":
			MoV = -np.diff(game[3:,:])
			assert(MoV[0,0] == -MoV[1,0])
		elif elo_type == "offdef":
			MoV = game[3,:] - avg_score
		# Get team's and their information
		elos = [elo_dict[tid][ixSeason][-1] for tid in game[1,:]]
		# Calculate parameters based on results
		if elo_type == "winloss":
			elo_diff = elos[0] - elos[1] if MoV[0,0] > 0 else elos[1] - elos[0]
			elo_diff = np.sign(elo_diff) * min(abs(elo_diff), max_elo_diff)
			new_elos = [rating_adjuster(elos[j], A, B, C, K, elo_diff, MoV[j,0]) for j in range(2)]
			try:
				assert(round(sum(elos),3) == round(sum(new_elos),3))
			except AssertionError:
				import pdb; pdb.set_trace()
		elif elo_type == "offdef":
			elo_diff = [elos[j][0] - elos[1-j][1] if MoV[j] > 0 else
						elos[1-j][1] - elos[j][0] for j in range(2)]
			elo_diff = [np.sign(ed) * min(abs(ed), max_elo_diff) for ed in elo_diff]
			new_elos = [(0,0),(0,0)]
			new_elos[0] = (rating_adjuster(elos[0][0], A, B, C, K, elo_diff[0], MoV[0]),
						   rating_adjuster(elos[0][1], A, B, C, K, elo_diff[1], -MoV[1]))
			new_elos[1] = (rating_adjuster(elos[1][0], A, B, C, K, elo_diff[1], MoV[1]),
						   rating_adjuster(elos[1][1], A, B, C, K, elo_diff[0], -MoV[0]))
		elo_dict[game[1,0]][ixSeason].append(new_elos[0])
		elo_dict[game[1,1]][ixSeason].append(new_elos[1])
		teamgid_map[game[1,0]][ixSeason].append(game[0,0])
		teamgid_map[game[1,1]][ixSeason].append(game[0,1])
	return elo_dict, teamgid_map


def build_games(all_data=None):
	"""
	Build list of unique games and the date differences
	"""
	# Load data
	if all_data is None:
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


def build_elo_mat(teamgid_map, games, dates_diff, elo_winloss, elo_offdef, min_season=1):
	"""
	"""
	# Remove seasons prior to min_season
	for tid,_ in elo_winloss.iteritems():
		teamgid_map[tid] = teamgid_map[tid][min_season:]
		elo_winloss[tid] = elo_winloss[tid][min_season:]
		elo_offdef[tid] = elo_offdef[tid][min_season:]
	# Build arrays
	X = []
	y = []
	ixSeason = 0
	for i, game in enumerate(games):
		# Check for season gap (100 days)
		if i > 0 and dates_diff[i-1] > 100:
			ixSeason += 1
		# Get team's and their information
		if ixSeason >= min_season:
			gid = game[0,0]
			tids = game[1,:]
			ixGames = [teamgid_map[tid][ixSeason-min_season].index(gid) for tid in tids]
			elos_wl = [elo_winloss[tid][ixSeason-min_season][ix] for ix,tid in zip(ixGames,tids)]
			elos_od = [elo_offdef[tid][ixSeason-min_season][ix] for ix,tid in zip(ixGames,tids)]
			X.append([elos_wl[0]] + list(elos_od[0]) + [elos_wl[1]] + list(elos_od[1]))
			X.append([elos_wl[1]] + list(elos_od[1]) + [elos_wl[0]] + list(elos_od[0]))
			y.append(game[3,0])
			y.append(game[4,0])
	X = np.array(X)
	y = np.array(y)
	if len(y.shape) < 2:
		y = y.reshape(y.shape[0], 1)
	return X, y


def run_conference_elos(teams, teamgid_map, games, dates_diff, team_elos, conferences):
	"""
	"""
	# build team to conference mapping
	teamConf_map = {}
	for team in teams:
		teamConf_map[team.tid] = str(int(team.info['ConferenceId']))
	# Init elo dict
	init_elo = team_elos[team_elos.keys()[0]][0][0]
	elo_dict = {}
	for cid in conferences.keys():
		elo_dict[cid] = []
		elo_dict[cid].append([init_elo])
	# Run elos over all games
	ixSeason = 0
	for i, game in enumerate(games):
		# Check for season gap (100 days)
		if i > 0 and dates_diff[i-1] > 100:
			ixSeason += 1
			for id_, elo in elo_dict.iteritems():
				curr_elo = elo_dict[id_][-1][-1]
				elo_dict[id_].append([curr_elo])
		# Find teams game index
		gid = game[0,0]
		tids = [tid for tid in game[1,:]]
		cids = [teamConf_map[tid] for tid in tids]
		if cids[0] == cids[1]:
			continue
		elos = [elo_dict[cid][ixSeason][-1] for cid in cids]
		ixGames = [teamgid_map[tid][ixSeason].index(gid) for tid in tids]
		elo_diff = [np.diff(team_elos[tid][ixSeason][ix:ix+2]) for tid,ix in zip(tids, ixGames)]
		elo_dict[cids[0]][ixSeason].append(elos[0] + elo_diff[0])
		elo_dict[cids[1]][ixSeason].append(elos[1] + elo_diff[1])