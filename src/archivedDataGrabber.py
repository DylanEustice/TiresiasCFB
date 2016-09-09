import pandas as pd
import numpy as np
import os
import datetime
from src.util import load_json, dump_json

EXCLUDE_CONCAT_FIELDS = ['Id', 'DateUtc', 'Season', 'is_home']

def load_and_save_archived_data(years=range(2005,2013)):
	all_data = load_all_archived_data(years=years)
	save_dir = os.path.join('data', 'compiled_team_data')
	all_data.to_pickle(os.path.join(save_dir, 'archived.df'))
	return all_data

def load_all_archived_data(years=range(2005,2013)):
	"""
	Load data for past seasons. Comes from different source.
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
	all_data['3rd Down Conv'] = third_down_conv.replace(np.nan, 0.)
	fourth_down_conv = all_data['Fourth Down Conv'] / all_data['Fourth Down Att']
	all_data['4th Down Conv'] = fourth_down_conv.replace(np.nan, 0.)
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
	all_data['Yards Per Pass'] = (all_data['Pass Yard'] / all_data['Pass Att']).replace(np.nan, 0.)
	all_data['Yards Per Play'] = (all_data['Total Yards'] / all_data['Total Plays']).replace(np.nan, 0.)
	all_data['Yards per Rush'] = (all_data['Rush Yard'] / all_data['Rush Att']).replace(np.nan, 0.)
	# Is home
	home_codes = (all_data['Game Code'].values / 1e12).astype(int)
	all_data['is_home'] = np.array(all_data['Team Code'] == home_codes).astype(int)
	# Total turnovers
	all_data['Turnovers'] = all_data['Pass Int'] + all_data['Fumble Lost']
	# Other (calc later)
	all_data['conferenceId'] = 0
	all_data['wl_elo'] = 0
	all_data['off_elo'] = 0
	all_data['def_elo'] = 0
	all_data['cf_elo'] = 0
	# Rename fields and ids to match new data
	all_data = rename_fields(all_data)
	all_data = map_team_conf_fields(all_data)
	all_data = combine_games(all_data)
	all_data = remove_unknown_teams(all_data)
	return all_data

def rename_fields(all_data):
	"""
	Rename fields to match recent data
	"""
	field_map = load_json('field_mapping.json', fdir=os.path.join('data', 'archived_data'))
	for old_field in all_data.keys():
		tmp_vals = pd.Series(all_data[old_field].values, index=all_data.index)
		all_data = all_data.drop(old_field, 1)
		if old_field in field_map:
			new_field = field_map[old_field]
			all_data[new_field] = tmp_vals
	return all_data

def map_team_conf_fields(all_data):
	"""
	Map team and conference ids to match recent data. Also append conference to data.
	"""
	# Load all data
	try:
		team_id_map = load_json('team_id_mapping.json', fdir=os.path.join('data', 'archived_data'))
		conf_id_map = load_json('conf_id_mapping.json', fdir=os.path.join('data', 'archived_data'))
	except:
		print "Need to generate team or conference mapping first, returning None"
		return None
	years = np.unique(all_data['Season'])
	team_info_storage = []
	for year in years:
		dir_ = os.path.join('data', 'archived_data', str(year))
		team_info_storage.append(pd.read_csv(os.path.join(dir_, 'team.csv')))
	# Map team ids and add mapped conference ids
	new_team_ids = np.zeros(all_data['TeamId'].shape[0])
	new_conf_ids = np.zeros(all_data['TeamId'].shape[0])
	for i, (tid, year) in enumerate(zip(all_data['TeamId'].values, all_data['Season'].values)):
		try:
			new_team_id = team_id_map[str(tid)]
		except KeyError:
			new_team_id = str(-1)
		ixYear = np.where(years==year)[0]
		ixTeam = team_info_storage[ixYear]['Team Code'].values == tid
		old_conf_id = team_info_storage[ixYear]['Conference Code'].values[ixTeam][0]
		try:
			new_conf_id = conf_id_map[str(old_conf_id)]
		except KeyError:
			new_conf_id = str(-1)
		new_team_ids[i] = new_team_id
		new_conf_ids[i] = new_conf_id
	all_data['TeamId'] = pd.Series(new_team_ids, index=all_data.index)
	all_data['conferenceId'] = pd.Series(new_conf_ids, index=all_data.index)
	return all_data

def combine_games(all_data):
	"""
	Combine matching game id's to have 'this' and 'other' info
	"""
	combined_fields = concat_data_fields(all_data)
	combined_data = pd.DataFrame(columns=combined_fields)
	gids = np.unique(all_data['Id'])
	for gid in gids:
		games = all_data[all_data['Id'] == gid]
		assert games.shape[0] == 2, "Should have found 2 games here"
		concat_games = concat_game_rows(games, combined_fields)
		combined_data = combined_data.append(pd.Series(concat_games[0]), ignore_index=True)
		combined_data = combined_data.append(pd.Series(concat_games[1]), ignore_index=True)
	return combined_data

def concat_data_fields(all_data):
	"""
	"""
	fields = all_data.keys()
	new_fields = []
	for pre in ['this_','other_']:
		for field in fields:
			if field in EXCLUDE_CONCAT_FIELDS:
				if field not in new_fields:
					new_fields.append(field)
			else:
				new_fields.append(pre+field)
	return new_fields

def concat_game_rows(games, new_fields):
	"""
	"""
	new_games = []
	for i in range(2):
		this_game = dict([(k,games[k].values[i]) for k in games.keys()])
		other_game = dict([(k,games[k].values[i-1]) for k in games.keys()])
		new_game = {}
		for f in new_fields:
			if f[:4] == 'this':
				new_game[f] = this_game[f[5:]]
			elif f[:5] == 'other':
				new_game[f] = other_game[f[6:]]
			else:
				new_game[f] = this_game[f]
		new_games.append(new_game)
	return new_games

def remove_unknown_teams(all_data):
	"""
	Removes teams which aren't included in new data
	"""
	# Mappings
	try:
		team_id_map = load_json('team_id_mapping.json', fdir=os.path.join('data', 'archived_data'))
		conf_id_map = load_json('conf_id_mapping.json', fdir=os.path.join('data', 'archived_data'))
	except:
		print "Need to generate team or conference mapping first, returning None"
		return None
	# Reverse mappings
	team_id_map_rev = dict([(int(new),int(old)) for old,new in team_id_map.iteritems()])
	conf_id_map_rev = dict([(int(new),int(old)) for old,new in conf_id_map.iteritems()])
	# Determine games with unknown teams/conferences
	is_unknown_team = lambda tid0, tid1: tid0 not in team_id_map_rev or tid1 not in team_id_map_rev
	is_unknown_conf = lambda cid0, cid1: cid0 not in conf_id_map_rev or cid1 not in conf_id_map_rev
	zip_tids = zip(all_data['this_TeamId'].values, all_data['other_TeamId'].values)
	zip_cids = zip(all_data['this_conferenceId'].values, all_data['other_conferenceId'].values)
	ixUnknown_team = [is_unknown_team(ttid, otid) for ttid, otid in zip_tids]
	ixUnknown_conf = [is_unknown_conf(tcid, ocid) for tcid, ocid in zip_cids]
	ixKeep = np.logical_not(np.logical_or(ixUnknown_team, ixUnknown_conf))
	return all_data[ixKeep]

def map_team_ids(save=False):
	"""
	Generate (as well as possible) a mapping between team ids for the data sources.
	"""
	dir_ = os.path.join('data', 'archived_data', '2013')
	arch_teams = pd.read_csv(os.path.join(dir_, 'team.csv'))
	team_ids = load_json(os.path.join('data', 'team_ids.json'))
	mapping = dict([(str(tid),"") for _,tid in team_ids.iteritems()])
	for name, tid in team_ids.iteritems():
		ixName = arch_teams['Name'].values == name
		if any(ixName):
			mapping[tid] = str(arch_teams['Team Code'].values[ixName][0])
	mapping = dict([(old,new) for new,old in mapping.iteritems()])
	if save:
		dump_json(mapping, 'team_id_mapping.json', fdir=os.path.join('data', 'archived_data'))
	return mapping

def map_conf_ids(years=range(2005,2014), save=False):
	"""
	Generate (as well as possible) a mapping between conferece ids for the data sources.
	"""
	mapping = {}
	for year in years:
		dir_ = os.path.join('data', 'archived_data', str(year))
		arch_confs = pd.read_csv(os.path.join(dir_, 'conference.csv'))
		conferences = load_json(os.path.join('data', str(year), 'Conferences.json'))
		for cid, data in conferences.iteritems():
			if str(cid) not in mapping:
				mapping[str(cid)] = ""
				ixName0 = str(data['Name']) == arch_confs['Name'].values
				ixName1 = str(data['Name'])+" Conference" == arch_confs['Name'].values
				ixName = np.logical_or(ixName0, ixName1)
				if any(ixName):
					mapping[str(cid)] = str(arch_confs['Conference Code'].values[ixName][0])
	mapping = dict([(old,new) if old != "" else ("old"+new,new) for new,old in mapping.iteritems()])
	if save:
		dump_json(mapping, 'conf_id_mapping.json', fdir=os.path.join('data', 'archived_data'))
	return mapping
