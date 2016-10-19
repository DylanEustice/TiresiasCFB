from src.team import build_all_teams, setup_game_input
import src.default_parameters as default
import datetime
import numpy as np
import pandas as pd
import os


def rank_teams(dataset, date=datetime.datetime.now(), rank_algo='lin_regress'):
	"""
	"""
	season = date.year if date.month >= 9 else date.year-1
	teams = [t for t in build_all_teams() if season in t.seasons]
	if rank_algo == 'lin_regress':
		team_net_wins = dict([(team.name,0) for team in teams])
		team_pt_diff = dict([(team.name,0) for team in teams])
		for team0 in teams:
			for team1 in teams:
				if team0 is team1:
					continue
				inp = setup_game_input([team0, team1], date, dataset)
				if inp is None:
					continue
				inp = dataset._postprocess_arr(dataset.inp_post, inp[0,:])
				scores = np.array(inp * dataset.B_raw)
				win = 1 if scores[0,0] > scores[0,1] else -1
				pt_diff = scores[0,0] - scores[0,1]
				team_net_wins[team0.name] += win
				team_pt_diff[team0.name] += pt_diff
	by_wins = pd.Series(team_net_wins).sort_values(ascending=False)
	by_pts = pd.Series(team_pt_diff).sort_values(ascending=False)
	return by_wins, by_pts