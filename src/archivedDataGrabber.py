import pandas as pd
import numpy as np
import os
import datetime
from src.util import load_json

def load_all_archived_data(years=range(2005,2014)):
	"""
	"""
	dataframes = []
	drive_arr = []
	play_arr = []
	for year in years:
		dir_ = os.path.join('data', 'archived_data', str(year))
		fname = os.path.join(dir_, 'team-game-statistics.csv')
		tmp_df = pd.read_csv(fname)
		# Append season
		season = [year for _ in range(tmp_df.shape[0])]
		tmp_df['Season'] = pd.Series(season, index=tmp_df.index)
		dataframes.append(tmp_df)
		# Read in plays and drives
		drive_arr.append(pd.read_csv(os.path.join(dir_, 'drive.csv')))
		play_arr.append(pd.read_csv(os.path.join(dir_, 'play.csv')))
	all_data = pd.concat(dataframes)
	drives = pd.concat(drive_arr)
	plays = pd.concat(play_arr)
	# Add dates
	dates_raw = [d%1e8 for d in all_data['Game Code']]
	dates = [datetime.datetime(year=int(d/1e4), month=int((d/1e2)%1e2), day=int(d%1e2))
		for d in dates_raw]
	all_data['DateUtc'] = pd.Series(dates, index=all_data.index)
	# Add total 1st downs
	tot_first_down = (all_data['1st Down Pass'] + 
		all_data['1st Down Rush'] + all_data['1st Down Penalty'])
	all_data['1st Downs'] = tot_first_down
	# Add conversion pct
	third_down_conv = all_data['Third Down Conv'] / all_data['Third Down Att']
	all_data['3rd Down Conv'] = third_down_conv
	fourth_down_conv = all_data['Fourth Down Conv'] / all_data['Fourth Down Att']
	all_data['4th Down Conv'] = fourth_down_conv
	# Add special teams / defensive TDs
	all_data['DEF TDs'] = all_data['Fum Ret TD'] + all_data['Int Ret TD']
	all_data['Special Teams TDs'] = all_data['Kickoff Ret TD'] + all_data['Punt Ret TD']
	# Total yards
	all_data['Total Yards'] = all_data['Pass Yard'] + all_data['Rush Yard']
	# Total drives and plays
	nDrives = []
	nPlays = []
	for row, game in all_data.iterrows():
		# Get matching games then matching drives
		dr_games = drives[drives['Game Code'] == game['Game Code']]
		pl_games = plays[plays['Game Code'] == game['Game Code']]
		dr_match = dr_games[dr_games['Team Code'] == game['Team Code']]
		pl_match = pl_games[pl_games['Offense Team Code'] == game['Team Code']]
		nDrives.append(dr_match.shape[0])
		nPlays.append(pl_match.shape[0])
	all_data['Total Drives'] = pd.Series(nDrives, index=all_data.index)
	all_data['Total Plays'] = pd.Series(nPlays, index=all_data.index)
	# Yards per
	all_data['Yards Per Pass'] = all_data['Pass Yard'] / all_data['Pass Att']
	all_data['Yards Per Play'] = all_data['Total Yards'] / all_data['Total Plays']
	all_data['Yards Per Rush'] = all_data['Rush Yard'] / all_data['Rush Att']
	# Is home
	home_codes = (all_data['Game Code'].values / 1e12).astype(int)
	all_data['is_home'] = all_data['Team Code'] == home_codes
	# Total turnovers
	all_data['Turnovers'] = all_data['Pass Int'] + all_data['Fumble Lost']
	# Other (calc later)
	all_data['conferenceId'] = 0
	all_data['wl_elo'] = 0
	all_data['off_elo'] = 0
	all_data['def_elo'] = 0
	all_data['cf_elo'] = 0
	return all_data

def rename_fields(all_data):
	"""
	"""
	field_map = load_json('field_mapping.json', fdir=os.path.join('data', 'archived_data'))
	for old_field in all_data.keys():
		tmp_vals = pd.Series(all_data[old_field].values, index=all_data.index)
		all_data = all_data.drop(old_field, 1)
		if old_field in field_map:
			new_field = field_map[old_field]
			all_data[new_field] = tmp_vals
	return all_data
