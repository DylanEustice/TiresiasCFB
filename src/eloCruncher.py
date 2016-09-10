import numpy as np
import pandas as pd
import datetime
import src.util as util
from src.team import build_all_teams
from src.searchAlgorithms import evolutionary_search
import matplotlib.pyplot as plt
import os
import src.default_parameters as default


def train_and_save_all_elos(nPop=10, iters=10, train_type='Optimal', **kwargs):
	"""
	"""
	assert train_type == 'Optimal' or train_type == 'Ranking'

	games, dates_diff = build_games()
	all_data = util.load_all_dataFrame()

	pop_and_fit = {}

	if train_type == 'Optimal':
		static_fields = None
	elif train_type == 'Ranking':
		static_fields = [False, False, False, False, True, False]

	do_ranking = train_type == 'Ranking'

	print "Winloss..."
	pop_wl, best_fit_wl = run_evolutionary_elo_search(obj_fun=elo_obj_fun_ranges, nPop=nPop,
		iters=iters, elo_type="winloss", static_fields=static_fields, do_ranking=do_ranking, 
		**kwargs)
	wl_elos, teamgid_map = run_elos(all_data, games=games, dates_diff=dates_diff, 
			elo_params=pop_wl[0,:], elo_type="winloss")
	np.savetxt(os.path.join(default.elo_dir, train_type+'_Winloss_Params.txt'), pop_wl[0,:])
	pop_and_fit['wl'] = [pop_wl, best_fit_wl]

	print "Offense/defense..."
	pop_od, best_fit_od = run_evolutionary_elo_search(obj_fun=elo_obj_fun_ranges, nPop=nPop,
		iters=iters, elo_type="offdef", static_fields=static_fields, do_ranking=do_ranking, 
		**kwargs)
	od_elos, teamgid_map = run_elos(all_data, games=games, dates_diff=dates_diff, 
			elo_params=pop_od[0,:], elo_type="offdef")
	np.savetxt(os.path.join(default.elo_dir, train_type+'_Offdef_Params.txt'), pop_od[0,:])
	pop_and_fit['od'] = [pop_od, best_fit_od]

	print "Conference..."
	pop_cf, best_fit_cf = run_evolutionary_elo_search(obj_fun=elo_obj_fun_ranges, nPop=nPop,
		iters=iters, elo_type="conf", teamgid_map=teamgid_map, team_elos=wl_elos,
		static_fields=static_fields, do_ranking=do_ranking, **kwargs)
	cf_elos, confgid_map = run_conference_elos(teamgid_map, wl_elos, games=games,
		dates_diff=dates_diff, elo_params=pop_cf[0,:])
	np.savetxt(os.path.join(default.elo_dir, train_type+'_Conf_Params.txt'), pop_cf[0,:])
	pop_and_fit['cf'] = [pop_cf, best_fit_cf]

	print "Passing..."
	pop_pod, best_fit_pod = run_evolutionary_elo_search(obj_fun=elo_obj_fun_ranges, nPop=nPop,
		iters=iters, elo_type="poffdef", static_fields=static_fields, do_ranking=do_ranking, 
		**kwargs)
	pod_elos, teamgid_map = run_elos(all_data, games=games, dates_diff=dates_diff, 
			elo_params=pop_pod[0,:], elo_type="poffdef")
	np.savetxt(os.path.join(default.elo_dir, train_type+'_PassYd_Params.txt'), pop_pod[0,:])
	pop_and_fit['pod'] = [pop_pod, best_fit_pod]

	print "Rushing..."
	pop_rod, best_fit_rod = run_evolutionary_elo_search(obj_fun=elo_obj_fun_ranges, nPop=nPop,
		iters=iters, elo_type="roffdef", static_fields=static_fields, do_ranking=do_ranking,
		**kwargs)
	rod_elos, teamgid_map = run_elos(all_data, games=games, dates_diff=dates_diff, 
			elo_params=pop_rod[0,:], elo_type="roffdef")
	np.savetxt(os.path.join(default.elo_dir, train_type+'_RushYd_Params.txt'), pop_rod[0,:])
	pop_and_fit['rod'] = [pop_rod, best_fit_rod]

	return pop_and_fit


def elo_obj_fun_ranges(params, all_data, games, dates_diff, elo_type="winloss",
	teamgid_map=None, team_elos=None, season_range=range(1,11)):
	"""
	"""
	if elo_type == "conf":
		elo_dict, gid_map = run_conference_elos(teamgid_map, team_elos,
			games=games, dates_diff=dates_diff, elo_params=params)
	else:
		elo_dict, gid_map = run_elos(all_data, games=games,
			dates_diff=dates_diff, elo_type=elo_type, elo_params=params)

	# By season
	obj_vals_seas = []
	for seas in season_range:
		pr_result, elo_diff = assess_elo_confidence(elo_dict, gid_map, games, dates_diff,
			season_range=[seas], do_print=False, elo_type=elo_type)
		obj_vals_seas.append(abs(t_test(pr_result, elo_diff)))
	seas_val = np.sqrt(np.mean(np.array(obj_vals_seas)**2))

	# By quartile
	obj_vals_qtr = []
	for qtr in range(4):
		pr_result, elo_diff = assess_elo_confidence(elo_dict, gid_map, games, dates_diff,
			season_quartile=[qtr], do_print=False, elo_type=elo_type, 
			season_range=season_range)
		obj_vals_qtr.append(abs(t_test(pr_result, elo_diff)))
	qtr_val = np.sqrt(np.mean(np.array(obj_vals_qtr)**2))

	# Full
	pr_result, elo_diff = assess_elo_confidence(elo_dict, gid_map, games, dates_diff,
		do_print=False, elo_type=elo_type, season_range=season_range)
	tot_val = abs(t_test(pr_result, elo_diff))

	return seas_val + qtr_val + tot_val


def elo_obj_fun(params, all_data, games, dates_diff, season_range=range(1,11),
	elo_type="winloss",	teamgid_map=None, team_elos=None, season_quartile=range(4)):
	"""
	params: A, B, C, K, season_regress, init_elo
	"""
	if params[3] <= 1e-6:
		return np.inf

	if elo_type == "conf":
		elo_dict, confgid_map = run_conference_elos(teamgid_map, team_elos,
			games=games, dates_diff=dates_diff, elo_params=params)
		pr_result, elo_diff = assess_elo_confidence(elo_dict, confgid_map, games,
			dates_diff,	season_range=season_range, do_print=False, elo_type=elo_type,
			season_quartile=season_quartile)

	else:
		elo_dict, teamgid_map = run_elos(all_data, games=games,
			dates_diff=dates_diff, elo_type=elo_type, elo_params=params)
		pr_result, elo_diff = assess_elo_confidence(elo_dict, teamgid_map, games,
			dates_diff,	season_range=season_range, do_print=False, elo_type=elo_type,
			season_quartile=season_quartile)

	return abs(t_test(pr_result, elo_diff))


def run_evolutionary_elo_search(obj_fun=elo_obj_fun_ranges, nPop=10, iters=10, kill_rate=0.5,
	evolve_rng=0.5,	season_range=range(1,11), elo_type="winloss", teamgid_map=None,
	team_elos=None, static_fields=None, init_params=None, do_ranking=True):
	"""
	"""
	all_data = util.load_all_dataFrame()
	games, dates_diff = build_games(all_data)

	if init_params is None:
		if elo_type == "winloss":
			init_params = [1., 1e-4, 1e-8, 9.5, 0.5, 1000]
		elif elo_type == "offdef":
			init_params = [1., 1e-4, 1e-8, 7.5, 0.5, 1000]
		elif elo_type == "conf":
			init_params = [1., 1e-3, 1e-6, 2., 0.5, 1000]
		elif elo_type == "poffdef" or elo_type == "roffdef":
			init_params = [1., 1e-4, 1e-8, 1.5, 0.5, 1000]

	if do_ranking:
		init_params[4] = 1.0

	if static_fields is None:
		static_fields = [False for _ in range(len(init_params))]
	static_fields = np.array(static_fields)

	args = [all_data, games, dates_diff]
	kwargs = dict([('season_range',season_range), ('elo_type',elo_type),
				   ('teamgid_map',teamgid_map), ('team_elos',team_elos)])

	return evolutionary_search(nPop, iters, kill_rate, evolve_rng,
		elo_obj_fun_ranges, init_params, static_fields, *args, **kwargs)


def assess_elo_confidence(elo_dict, gid_map, games, dates_diff, season_range=range(1,11), 
	do_print=True, elo_type="winloss", season_quartile=range(4)):
	"""
	"""
	# Get type
	is_winloss = elo_type == "winloss"
	is_offdef = elo_type == "offdef" or elo_type == "poffdef" or elo_type == "roffdef"
	is_conf = elo_type == "conf"
	assert is_winloss or is_offdef or is_conf, "elo_type must be  'winloss', '(p/r)offdef', or conf"
	# Get correct game field
	if elo_type == "offdef":
		compare_avg = default.avg_score
		MoV_field = 3
	elif elo_type == "poffdef":
		compare_avg = default.avg_passing
		MoV_field = 7
	elif elo_type == "roffdef":
		compare_avg = default.avg_rushing
		MoV_field = 9
	pr_result = [] # (Pr_win, correct)
	elo_diff = []
	ixSeason = 0
	for i, game in enumerate(games):
		# Check for season gap
		if i > 0 and dates_diff[i-1] > default.season_day_sep:
			ixSeason += 1
		# Don't do this for first min_season seasons
		if ixSeason in season_range:
			# Get team's and their information
			gid = game[0,0]
			tids = [tid for tid in game[1,:]]
			if is_winloss or is_offdef:
				ixGames = [gid_map[tid][ixSeason].index(gid) for tid in tids]
				game_quartials = [min(int(ix/3), 3) for ix in ixGames]
				elos = [elo_dict[tid][ixSeason][ix] for tid,ix in zip(tids, ixGames)]
			elif is_conf:
				cids = [cid for cid in game[5,:]]
				if cids[0] == cids[1]:
					continue
				ixGames = [gid_map[cid][ixSeason].index(gid) for cid in cids]
				game_quartials = [min(int(4*ix/len(gid_map[cid][ixSeason])), 3) for 
					ix,cid in zip(ixGames, cids)]
				elos = [elo_dict[cid][ixSeason][ix] for cid,ix in zip(cids,ixGames)]
			else:
				raise Exception('elo_type must be  "winloss", "offdef", "conf"')
			# Make sure this is in the correct season quartile
			if not any([q in season_quartile for q in game_quartials]):
				continue
			# Determine win probabilities and winner
			if is_winloss or is_conf:
				Pr_win = max(elo_game_probs(elos))
				correct = np.argmax(elos) == np.argmax(game[3,:])
				pr_result.append((Pr_win, correct))
				elo_diff.append(abs(elos[0]-elos[1]))
			elif is_offdef:
				# Do for each offense-defense matchup seperately
				for j in range(2):
					offdef_elos = [elos[j][0], elos[1-j][1]]
					Pr_win = max(elo_game_probs(offdef_elos))
					ixMax_elo = np.argmax(offdef_elos)
					correct = ((ixMax_elo==1 and game[MoV_field,j] < compare_avg) or
							   (ixMax_elo==0 and game[MoV_field,j] >= compare_avg))
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


def plt_pr_result_hist(pr_result, nBins=6, nFig=None, **kwargs):
	"""
	"""
	bins = np.linspace(0.5,1.0,nBins)
	to_hist = []
	n_occr = []
	for i in range(nBins-1):
		ix = [j for j,pr in enumerate(pr_result) if bins[i] <= pr[0] < bins[i+1]]
		nCorrect = len([j for j in ix if pr_result[j][1]])
		if len(ix) > 0:
			pct_correct = 100. * nCorrect / len(ix)
		else:
			pct_correct = -1
		to_hist.append(pct_correct)
		n_occr.append(len(ix))
	fig = plt.figure(nFig)
	ax1 = fig.add_subplot(211)
	ax2 = fig.add_subplot(212)
	bin_width = bins[1] - bins[0]
	bins = bins[:-1] + bin_width / 2
	ax1.bar(bins, to_hist, width=bin_width, **kwargs)
	ax1.grid('on')
	ax2.bar(bins, n_occr, width=bin_width, **kwargs)
	ax2.grid('on')


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


def rating_adjuster(Ri, params, elo_diff, MoV, max_MoV_mult=1e3):
	"""
	Adjust a team's Elo rating based on the outcome of the game
	A, B, C, K:	Parameters
	Ri:			Initial elo rating
	MoV:		Margin of victory (this_score - other_score)
	elo_diff:	Elo delta (elo_winner - elo_loser)
	"""
	MoV_mult = 1. / np.dot(params[:3], [1, elo_diff, np.sign(elo_diff)*elo_diff**2])
	MoV_adj = np.sign(MoV) * np.log(np.abs(MoV) + 1)
	new_Ri = Ri + params[3] * MoV_adj * MoV_mult
	return new_Ri


def append_elos_to_dataFrame(shift=True, save=True):
	"""
	"""
	elos = gen_elo_files(shift=shift, save=save)
	all_data = util.load_all_dataFrame()
	# Get game and team ids
	gids = [gid for gid in all_data['Id']]
	this_tids = [tid for tid in all_data['this_TeamId']]
	other_tids = [tid for tid in all_data['other_TeamId']]
	types = ['wl_elo', 'off_elo', 'def_elo', 'cf_elo', 'poff_elo', 'pdef_elo', 'roff_elo', 'rdef_elo']
	tid_elos = dict([(tid,[]) for tid in this_tids])
	for i, t in enumerate(types):
		this_elos = []
		other_elos = []
		for gid, this_tid, other_tid in zip(gids, this_tids, other_tids):
			# If game not here, just use teams' previous elo score
			if gid in elos:
				this_elo = elos[gid][this_tid][i]
				other_elo = elos[gid][other_tid][i]
			else:
				this_elo = tid_elos[this_tid][-1]
				other_elo = tid_elos[other_tid][-1]
			# Save elo
			tid_elos[this_tid].append(this_elo)
			tid_elos[other_tid].append(other_elo)
			this_elos.append(this_elo)
			other_elos.append(other_elo)
		all_data['this_' + t] = pd.Series(this_elos, index=all_data.index)
		all_data['other_' + t] = pd.Series(other_elos, index=all_data.index)
	# Save
	if save:
		all_data.to_pickle(os.path.join('data', 'compiled_team_data', 'all.df'))
	return all_data


def team_to_conf_map(teams):
	# Build team to conference mapping
	seasons = teams[0].seasons
	teamConf_map = []
	for season in seasons:
		teamConf_map.append({})
		for team in teams:
			try:
				teamConf_map[-1][team.tid] = str(int(team.info[season]['ConferenceId']))
			except KeyError:
				teamConf_map[-1][team.tid] = -1
	return teamConf_map


def gen_elo_files(shift=True, save=True):
	"""
	shift: shifts the elo of a game to be after the outcome
	"""
	teams = build_all_teams()
	wl_elos, od_elos, cf_elos, pod_elos, rod_elos, teamgid_map, confgid_map = run_best_elos()
	# Seperate offensive and defensive elos
	off_elos, def_elos = sep_off_def_elos(od_elos)
	poff_elos, pdef_elos = sep_off_def_elos(pod_elos)
	roff_elos, rdef_elos = sep_off_def_elos(rod_elos)
	# Put into dictionary with game/team ID keys
	teamConf_map = team_to_conf_map(teams)
	elos = {}
	s = 1 if shift else 0
	for tid in wl_elos:
		for ixSeason in range(len(wl_elos[tid])):
			cid = int(teamConf_map[ixSeason][tid])
			for ix in range(len(wl_elos[tid][ixSeason])-1):
				assert cid != -1
				gid = teamgid_map[tid][ixSeason][ix]
				if gid not in elos:
					elos[gid] = {}
				ixConf_elos = confgid_map[cid][ixSeason].index(gid)
				elos[gid][tid] = [wl_elos[tid][ixSeason][ix+s], 
								  off_elos[tid][ixSeason][ix+s], 
								  def_elos[tid][ixSeason][ix+s],
								  cf_elos[cid][ixSeason][ixConf_elos], 
								  poff_elos[tid][ixSeason][ix+s], 
								  pdef_elos[tid][ixSeason][ix+s],
								  roff_elos[tid][ixSeason][ix+s], 
								  rdef_elos[tid][ixSeason][ix+s]]
	# Save files to JSON
	if save:
		util.dump_json(elos, "elos.json", fdir=default.elo_dir)
	return elos

def sep_off_def_elos(elos):
	off_elos, def_elos = {}, {}
	for tid in elos:
		off_elos[tid] = []
		def_elos[tid] = []
		for ixSeason in range(len(elos[tid])):
			off_elos[tid].append([elo[0] for elo in elos[tid][ixSeason]])
			def_elos[tid].append([elo[1] for elo in elos[tid][ixSeason]])
	return off_elos, def_elos


def run_best_elos():
	"""
	"""
	# Load parameter files
	wl_params = np.loadtxt(os.path.join(default.elo_dir, 'Optimal_Winloss_Params.txt'))
	od_params = np.loadtxt(os.path.join(default.elo_dir, 'Optimal_Offdef_Params.txt'))
	cf_params = np.loadtxt(os.path.join(default.elo_dir, 'Optimal_Conf_Params.txt'))
	pod_params = np.loadtxt(os.path.join(default.elo_dir, 'Optimal_PassYd_Params.txt'))
	rod_params = np.loadtxt(os.path.join(default.elo_dir, 'Optimal_RushYd_Params.txt'))
	# Load data
	all_data = util.load_all_dataFrame()
	games, dates_diff = build_games()
	teams = build_all_teams()
	# Run elos
	wl_elos, teamgid_map = run_elos(all_data, games=games, dates_diff=dates_diff, 
		elo_params=wl_params, elo_type="winloss")
	od_elos,_ = run_elos(all_data, games=games, dates_diff=dates_diff, 
		elo_params=od_params, elo_type="offdef")
	cf_elos, confgid_map = run_conference_elos(teamgid_map, wl_elos, 
		games=games, dates_diff=dates_diff, elo_params=cf_params)
	pod_elos,_ = run_elos(all_data, games=games, dates_diff=dates_diff, 
		elo_params=pod_params, elo_type="poffdef")
	rod_elos,_ = run_elos(all_data, games=games, dates_diff=dates_diff, 
		elo_params=rod_params, elo_type="roffdef")
	return wl_elos, od_elos, cf_elos, pod_elos, rod_elos, teamgid_map, confgid_map


def run_elos(all_data, elo_params=[1., 1e-4, 1e-8, 10., 0.5, 1000], games=[],
	dates_diff=[], elo_type="winloss", start_season=0):
	"""
	elo_type: "winloss", "(p/r)offdef"
	"""
	if len(games) == 0 or len(dates_diff) == 0:
		games, dates_diff = build_games()
	# Set params
	params = elo_params[:4]
	season_regress = elo_params[4]
	init_elo = elo_params[5]
	max_elo_diff = ((4*params[0]*params[2] + params[1]**2)**0.5 - params[1]) / (2*params[2]) - 1
	# Get type
	is_winloss = elo_type == "winloss"
	is_offdef = elo_type == "offdef" or elo_type == "poffdef" or elo_type == "roffdef"
	assert is_winloss or is_offdef, "elo_type must be  'winloss' or '(p/r)offdef'"
	# Get correct game field
	if elo_type == "offdef":
		compare_avg = default.avg_score
		MoV_field = 3
	elif elo_type == "poffdef":
		compare_avg = default.avg_passing
		MoV_field = 7
	elif elo_type == "roffdef":
		compare_avg = default.avg_rushing
		MoV_field = 9
	# Set up elo dictionary
	tids = np.unique(all_data['this_TeamId'])
	elo_dict = {}
	teamgid_map = {}
	for tid in tids:
		elo_dict[tid] = []
		elo_dict[tid].append([])
		if is_winloss:
			elo_dict[tid][-1].append(init_elo)
		elif is_offdef:
			elo_dict[tid][-1].append((init_elo, init_elo))
		teamgid_map[tid] = []
		teamgid_map[tid].append([])
	# Walk though games
	ixSeason = 0
	for i, game in enumerate(games):
		# Check for season gap
		if i > 0 and dates_diff[i-1] > default.season_day_sep:
			ixSeason += 1
			for id_, elo in elo_dict.iteritems():
				curr_elo = elo_dict[id_][-1][-1]
				elo_dict[id_].append([])
				if is_winloss:
					start_elo = curr_elo + season_regress*(init_elo - curr_elo)
				elif is_offdef:
					start_elo = tuple(e + season_regress*(init_elo - e) for e in curr_elo)
				elo_dict[id_][ixSeason].append(start_elo)
				teamgid_map[id_].append([])
		if ixSeason < start_season:
			continue
		# Recalculate Elos with results of game
		if is_winloss:
			MoV = -np.diff(game[3:,:])
			assert(MoV[0,0] == -MoV[1,0])
		elif is_offdef:
			MoV = game[MoV_field,:] - compare_avg
		# Get team's and their information
		elos = [elo_dict[tid][ixSeason][-1] for tid in game[1,:]]
		# Calculate parameters based on results
		if is_winloss:
			elo_diff = elos[0] - elos[1] if MoV[0,0] > 0 else elos[1] - elos[0]
			elo_diff = np.sign(elo_diff) * min(abs(elo_diff), max_elo_diff)
			new_elos = [rating_adjuster(elos[j], params, elo_diff, MoV[j,0]) for j in range(2)]
			try:
				assert(round(sum(elos),3) == round(sum(new_elos),3))
			except AssertionError:
				import pdb; pdb.set_trace()
		elif is_offdef:
			elo_diff = [elos[j][0] - elos[1-j][1] if MoV[j] > 0 else
						elos[1-j][1] - elos[j][0] for j in range(2)]
			elo_diff = [np.sign(ed) * min(abs(ed), max_elo_diff) for ed in elo_diff]
			new_elos = [(0,0),(0,0)]
			new_elos[0] = (rating_adjuster(elos[0][0], params, elo_diff[0], MoV[0]),
						   rating_adjuster(elos[0][1], params, elo_diff[1], -MoV[1]))
			new_elos[1] = (rating_adjuster(elos[1][0], params, elo_diff[1], MoV[1]),
						   rating_adjuster(elos[1][1], params, elo_diff[0], -MoV[0]))
		elo_dict[game[1,0]][ixSeason].append(new_elos[0])
		elo_dict[game[1,1]][ixSeason].append(new_elos[1])
		teamgid_map[game[1,0]][ixSeason].append(game[0,0])
		teamgid_map[game[1,1]][ixSeason].append(game[0,1])
	return elo_dict, teamgid_map


def run_conference_elos(teamgid_map, team_elos, games=[],
	dates_diff=[], elo_params=[1., 1e-4, 1e-8, 10., 0.5, 1000]):
	"""
	"""
	if len(games) == 0 or len(dates_diff) == 0:
		games, dates_diff = build_games()
	# Set params
	params = elo_params[:4]
	season_regress = elo_params[4]
	init_elo = elo_params[5]
	max_elo_diff = ((4*params[0]*params[2] + params[1]**2)**0.5 - params[1]) / (2*params[2]) - 1
	# Init elo dict
	elo_dict = {}
	confgid_map = {}
	all_conf_ids = []
	for g in games:
		all_conf_ids.append(g[5,0])
		all_conf_ids.append(g[5,1])
	all_conf_ids = np.unique(all_conf_ids)
	for cid in all_conf_ids:
		elo_dict[cid] = []
		elo_dict[cid].append([init_elo])
		confgid_map[cid] = []
		confgid_map[cid].append([])
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
				confgid_map[id_].append([])
		# Find teams game index and conference ids
		gid = game[0,0]
		tids = [tid for tid in game[1,:]]
		cids = [cid for cid in game[5,:]]
		confgid_map[cids[0]][ixSeason].append(game[0,0])
		confgid_map[cids[1]][ixSeason].append(game[0,1])
		# Get MoV and conference elos at gametime
		MoV = -np.diff(game[3:,:])
		elos = [elo_dict[cid][ixSeason][-1] for cid in cids]
		if cids[0] == cids[1]:
			# skip if not OOC
			elo_dict[cids[0]][ixSeason].append(elos[0])
			elo_dict[cids[1]][ixSeason].append(elos[1])
			continue
		elo_diff = elos[0] - elos[1] if MoV[0,0] > 0 else elos[1] - elos[0]
		elo_diff = np.sign(elo_diff) * min(abs(elo_diff), max_elo_diff)
		new_elos = [rating_adjuster(elos[j], params, elo_diff, MoV[j,0]) for j in range(2)]
		# Get raw team probabilities
		ixGames = [teamgid_map[tid][ixSeason].index(gid) for tid in tids]
		elos_tms = [team_elos[tid][ixSeason][ix] for tid,ix in zip(tids, ixGames)]
		elo_diff_tms = elos_tms[0] - elos_tms[1] if MoV[0,0] > 0 else elos_tms[1] - elos_tms[0]
		Pr_outcome = Pr_elo(elo_diff_tms)
		# Normalize by number of conference OOC games and probability of outcome based
		# on team elos
		# conf_size_mult = (avg_conf_size/conf_sizes[cid])
		new_elos = [e + (1-Pr_outcome)*(ne-e) for e,ne,cid in zip(elos, new_elos, cids)]
		# Store
		elo_dict[cids[0]][ixSeason].append(new_elos[0])
		elo_dict[cids[1]][ixSeason].append(new_elos[1])
	return elo_dict, confgid_map


def build_games(all_data=None):
	"""
	Build list of unique games and the date differences
	"""
	# Load data
	if all_data is None:
		all_data = util.load_all_dataFrame()
	# Sort unique game ids by date
	gids, ix_gids = np.unique(all_data['Id'], return_index=True)
	ix_date = np.argsort(np.array(all_data['DateUtc'])[ix_gids])
	gids = gids[ix_date]
	data = np.array([all_data['Id'],
		all_data['this_TeamId'], all_data['other_TeamId'],
		all_data['this_Score'], all_data['other_Score'],
		all_data['this_conferenceId'], all_data['other_conferenceId'],
		all_data['this_Passing Yds'], all_data['other_Passing Yds'],
		all_data['this_Rushing Yds'], all_data['other_Rushing Yds']])
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
		if not np.any([np.isnan(val) for val in game.ravel()]):
			games.append(game)
	return games, dates_diff


def build_elo_mats(all_data=None, distinct_home=False, season_range=range(2006,2016), elo_fields=None):
	"""
	"""
	if all_data is None:
		all_data = append_elos_to_dataFrame(shift=False, save=False)
	if distinct_home:
		ixHome = all_data['is_home'] == True
		all_data = all_data[ixHome]

	ixInRange = np.array([season in season_range for season in all_data['Season'].values])

	X_fields = default.all_elo_fields if elo_fields is None else elo_fields
	X = np.matrix(np.vstack([all_data[f].values[ixInRange] for f in X_fields])).T

	y_fields = ['this_Score', 'other_Score']
	y = np.matrix(np.vstack([all_data[f].values[ixInRange] for f in y_fields])).T

	return X, y