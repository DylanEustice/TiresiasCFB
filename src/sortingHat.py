from src.team import build_all_teams
import src.default_parameters as default
import datetime
import numpy as np
import pandas as pd
import os


def rank_teams(date=datetime.datetime.now(), rank_algo='elo_lin_regress', elo_fields=None):
	"""
	"""
	season = date.year if date.month >= 9 else date.year-1
	teams = [t for t in build_all_teams() if season in t.seasons]
	if rank_algo == 'elo_lin_regress':
		B = np.loadtxt(os.path.join(default.elo_dir, 'Linear_Regression_Beta.txt'))
		for team in teams:
			team.elos = team.get_current_elos(next_game_date=date, elo_fields=elo_fields)
		team_net_wins = dict([(team.name,0) for team in teams])
		team_pt_diff = dict([(team.name,0) for team in teams])
		for team0 in teams:
			for team1 in teams:
				if team0 is team1:
					continue
				scores = elo_lin_regress_sim(team0, team1, B)
				win = 1 if scores[0] > scores[1] else -1
				pt_diff = scores[0] - scores[1]
				team_net_wins[team0.name] += win
				team_pt_diff[team0.name] += pt_diff
	by_wins = pd.Series(team_net_wins).sort_values(ascending=False)
	by_pts = pd.Series(team_pt_diff).sort_values(ascending=False)
	return by_wins, by_pts


def elo_lin_regress_sim(team0, team1, B):
	elos = np.hstack([team0.elos, team1.elos])
	scores = np.array(np.matrix(elos) * B)[0]
	return scores