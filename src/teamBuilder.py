import os
import re
from src.util import load_json, dump_json, grab_scraper_data, load_all_dataFrame
from src.team import build_all_teams
import pandas as pd
import csv
import datetime
from dateutil import parser
import src.default_parameters as default
from src.eloCruncher import append_elos_to_dataFrame


def build_all_from_scratch(refresh_data=True, years='all'):
	print "Compiling teams ..."
	compile_and_save_teams(years=years, refresh_data=refresh_data)
	compile_fields()
	print "Indexing games ..."
	index_game_folders()
	build_teamgame_index()
	print "Building team info and data ..."
	build_team_ids()
	build_team_names()
	build_team_DataFrames()
	save_all_df_cols()
	add_elo_conf_to_all()
	print "Adding archived data ..."
	add_archived_data()
	print "Building team schedules ..."
	build_team_schedules()
	extract_lines_from_schedule()
	print "Appending elos to all data ..."
	append_elos_to_dataFrame()


def index_game_folders():
	"""
	Create a dictionary with the gameid as the
	key and the path to the containing folder as
	the value.
	"""
	game_index = {}
	for root, dirs, files in os.walk('data'):
		for f in dirs:
			try:
				gameid = int(re.search(r'game_(?P<id>\d+)', f).group('id'))
				game_index[gameid] = os.path.join(root, f)
			except AttributeError:
				pass
	dump_json(game_index, 'game_index.json', fdir='data', indent=4)


def build_teamgame_index():
	teamgame_index = {}
	for root, dirs, files in os.walk('data'):
		for f in files:
			try:
				gameid = re.match(r'gameinfo_(?P<gameid>\d+).json', f).group('gameid')
				gameinfo = load_json(os.path.join(root, f))
				try:
					teamgame_index[gameinfo['HomeTeamId']].append(gameid)
				except KeyError:
					teamgame_index[gameinfo['HomeTeamId']] = [gameid]
				try:
					teamgame_index[gameinfo['AwayTeamId']].append(gameid)
				except KeyError:
					teamgame_index[gameinfo['AwayTeamId']] = [gameid]
			except AttributeError:
				pass
	dump_json(teamgame_index, 'teamgame_index.json', fdir='data', indent=4)


def add_archived_data(arch_years=range(2005,2013)):
	"""
	"""
	# Load data
	new_data = load_all_dataFrame()
	new_data.to_pickle(os.path.join(default.comp_team_dir, 'all_only_new.df'))
	arch_data = pd.read_pickle(os.path.join(default.comp_team_dir, 'archived.df'))
	# Only keep fields shared by both
	new_fields = list(new_data.keys())
	arch_fields = list(arch_data.keys())
	all_fields = sorted(list(set(new_fields + arch_fields)))
	fields = []
	for f in all_fields:
		if f in new_fields and f in arch_fields:
			fields.append(f)
	# Now build a data frame with both
	all_data = pd.DataFrame()
	ixNew = new_data['Season'] > arch_years[-1]
	ixArch = np.logical_or(arch_years[0] <= arch_data['Season'],
						   arch_years[-1] >= arch_data['Season'])
	use_new = new_data[ixNew]
	use_arch = arch_data[ixArch]
	ixSort = np.argsort(np.hstack([use_arch['DateUtc'].values, use_new['DateUtc'].values]))
	for f in fields:
		vals = np.hstack([use_arch[f].values, use_new[f].values])[ixSort]
		all_data[f] = pd.Series(vals)
	all_data.to_pickle(os.path.join(default.comp_team_dir, 'all.df'))


def compile_and_save_teams(years='all', refresh_data=False):
	"""
	Compile all teams into dictionaries and save as json
	"""
	teams = compile_teams(years=years, refresh_data=refresh_data)
	dump_json(teams, 'all.json', fdir=default.comp_team_dir, indent=4)
	for tid, team in teams.iteritems():
		dump_json(team, team['school'] + '.json', fdir=fdir, indent=4)


def build_team_schedules(years='all'):
	"""
	Make a data frame with only future games
	"""
	teams = {}
	today = datetime.datetime.now()
	game_index = load_json(os.path.join('data', 'game_index.json'))
	teamgame_index = load_json(os.path.join('data', 'teamgame_index.json'))
	all_schedule_gids = []
	if years == 'all':
		years = [2016]
	for year in years:
		try:
			yeardir = os.path.join('data', str(year))
		except IOError:
			continue
		# Add new teams' year
		for root, dirs, files in os.walk(os.path.join(yeardir, 'teams')):
			for f in files:
				newteam = load_json(os.path.join(root, f))
				try:
					setup_team_year(year, teams, newteam)
				except KeyError:
					init_team(teams, newteam)
					setup_team_year(year, teams, newteam)
		# Add all games to teams
		for teamid in teams:
			for gameid in teamgame_index[str(teamid)]:
				game_path = game_index[gameid]
				game_year = re.search(r'data\W+(?P<year>\d{4})\W+gameinfo', game_path).group('year')
				if game_year == str(year):
					teams[teamid]['games'][str(year)][gameid] = {}
					gameinfo = load_json('gameinfo_'+gameid+'.json', fdir=game_path)
					date = parser.parse(gameinfo['DateUtc'])
					if date >= today:
						teams[teamid]['games'][str(year)][gameid]['gameinfo'] = gameinfo
	schedule = pd.DataFrame(columns=default.schedule_columns)
	for tid, team in teams.iteritems():
		for year in team['games'].keys():
			for gid in team['games'][year].keys():
				info = team['games'][year][gid]['gameinfo']
				is_home = info['HomeTeamId'] == tid
				try:
					# set spread to negative if away
					spread = (1 - 2*is_home) * float(info['Spread'])
				except:
					spread = np.nan
				try:
					overunder = float(info['OverUnder'])
				except:
					overunder = np.nan
				# Assumes id, this_tid, other_tid, date, season, week, spread, overunder, is_home
				row = [
					gid, int(tid), int(info['AwayTeamId']) if home else int(info['HomeTeamId']),
					parser.parse(info['DateUtc']), int(info['Season']), int(info['Week']), 
					spread, overunder, is_home
				]
				schedule = schedule.append(pd.Series(row, index=default.schedule_columns), ignore_index=True)
	schedule.to_pickle(os.path.join(default.comp_team_dir, 'schedule.df'))
	return schedule


def extract_lines_from_schedule():
	schedule = load_schedule()
	schedule = schedule[schedule['is_home']]
	ixHasSpread = np.logical_not(np.isnan(schedule['Spread'].values))
	ixHasOverUnder = np.logical_not(np.isnan(schedule['OverUnder'].values))
	ixUse = np.logical_or(ixHasSpread, ixHasOverUnder)
	spreads = schedule['Spread'].values[ixUse]
	overUnder = schedule['OverUnder'].values[ixUse]
	gids = schedule['Id'].values[ixUse]
	try:
		lines = load_json('lines.json', fdir=default.comp_team_dir)
	except IOError:
		lines = {}
	for i, gid in enumerate(gids):
		if gid not in lines:
			lines[gid] = {}
			lines[gid]['Spread'] = spreads[i]
			lines[gid]['OverUnder'] = overUnder[i]
		else:
			# Overwrite if available
			if not np.isnan(spreads[i]):
				lines[gid]['Spread'] = spreads[i]
			if not np.isnan(overUnder[i]):
				lines[gid]['OverUnder'] = overUnder[i]
	dump_json(lines, 'lines.json', fdir=default.comp_team_dir)


def compile_teams(years='all', refresh_data=False):
	"""
	Compile teams from inputted year range into dictionaries.
	'refresh_data' will delete the current data and reload it
	from BarrelRollCFB.
	"""
	if refresh_data:
		grab_scraper_data()
	teams = {}
	game_index = load_json(os.path.join('data', 'game_index.json'))
	teamgame_index = load_json(os.path.join('data', 'teamgame_index.json'))
	if years == 'all':
		years = range(2000, default.this_year+1)
	for year in years:
		print year
		try:
			yeardir = os.path.join('data', str(year))
		except IOError:
			continue
		# Add new teams' year
		for root, dirs, files in os.walk(os.path.join(yeardir, 'teams')):
			for f in files:
				newteam = load_json(os.path.join(root, f))
				try:
					setup_team_year(year, teams, newteam)
				except KeyError:
					init_team(teams, newteam)
					setup_team_year(year, teams, newteam)
		# Add all games to teams
		for teamid in teams:
			for gameid in teamgame_index[str(teamid)]:
				game_path = game_index[gameid]
				game_year = re.search(r'data\W+(?P<year>\d{4})\W+gameinfo', game_path).group('year')
				if game_year == str(year):
					teams[teamid]['games'][str(year)][gameid] = {}
					gameinfo = load_json('gameinfo_'+gameid+'.json', fdir=game_path)
					try:
						boxscore = load_json('boxscore.json', fdir=game_path)
					except IOError:
						boxscore = None
					try:
						playerstats = load_json('playerstats.json', fdir=game_path)
					except IOError:
						playerstats = None
					teams[teamid]['games'][str(year)][gameid]['gameinfo'] = gameinfo
					teams[teamid]['games'][str(year)][gameid]['boxscore'] = boxscore
					teams[teamid]['games'][str(year)][gameid]['playerstats'] = playerstats
	return teams


def init_team(teams, newteam):
	teams[newteam['Id']] = {}
	teams[newteam['Id']]['school'] = newteam['School']
	teams[newteam['Id']]['name'] = newteam['Name']
	teams[newteam['Id']]['abbr'] = newteam['Abbr']
	teams[newteam['Id']]['primaryColor'] = newteam['PrimaryColor']
	teams[newteam['Id']]['secondaryColor'] = newteam['SecondaryColor']
	teams[newteam['Id']]['teamInfo'] = {}
	teams[newteam['Id']]['games'] = {}


def setup_team_year(year, teams, newteam):
	teams[newteam['Id']]['teamInfo'][str(year)] = newteam
	teams[newteam['Id']]['games'][str(year)] = {}


def compile_fields():
	"""
	Compile and save unique set of all boxscore field names
	"""
	box_keys = []
	game_keys = []
	teams = load_json('all.json', fdir=default.comp_team_dir)
	for tid, team in teams.iteritems():
		for year, games in team['games'].iteritems():
			for gid, game in games.iteritems():
				# boxscore
				try:
					box_keys.extend(game['boxscore']['awayTeam'].keys())
				except KeyError: pass
				try:
					game_keys.extend(game['boxscore']['hoemTeam'].keys())
				except KeyError: pass
				# gameinfo
				try:
					game_keys.extend(game['gameinfo'].keys())
				except KeyError: pass
	unique_box_keys = list(set(box_keys))
	unique_game_keys = list(set(game_keys))
	# Manual unwanted key removal
	unwanted_fields = ['Links', 'AwayTeamInfo', 'HomeTeamInfo', 'GameState']
	for field in unwanted_fields:
		try:
			unique_game_keys.remove(field)
		except ValueError:
			pass
	# write boxscore
	with open(os.path.join('data', 'boxscore_fields.csv'), 'wb') as f:
		writer = csv.writer(f)
		for key in unique_box_keys:
			writer.writerow([key])
	# write gameinfo
	with open(os.path.join('data', 'gameinfo_fields.csv'), 'wb') as f:
		writer = csv.writer(f)
		for key in unique_game_keys:
			writer.writerow([key])


def build_team_ids():
	"""
	Map school name to id #
	"""
	teams = load_json('all.json', fdir=default.comp_team_dir)
	teamid_dict = {}
	for tid, team in teams.iteritems():
		teamid_dict[team['school']] = tid
	dump_json(teamid_dict, 'team_ids.json', fdir='data', indent=4)


def build_team_names():
	"""
	Map school id # to name
	"""
	teams = load_json('all.json', fdir=default.comp_team_dir)
	teamid_dict = {}
	for tid, team in teams.iteritems():
		teamid_dict[tid] = team['school']
	dump_json(teamid_dict, 'team_names.json', fdir='data', indent=4)


def build_team_DataFrames():
	"""
	Build pandas DataFrame for each team and saves as pickle file
	"""
	teams = load_json('all.json', fdir=default.comp_team_dir)
	data_frames = []
	for tid, team in teams.iteritems():
		box_arr = team_boxscore_to_array(team, tid)
		fname = tid +'_DataFrame.df'
		box_arr.to_pickle(os.path.join(comp_team_dir, fname))
		data_frames.append(box_arr)
	all_df = concatenate_team_DataFrames(data_frames=data_frames)
	all_df.to_pickle(os.path.join(default.comp_team_dir, 'all.df'))


def add_elo_conf_to_all():
	"""
	"""
	all_df = load_all_dataFrame()
	# Add elo keys
	add_elo_keys = ['wl_elo', 'off_elo', 'def_elo', 'cf_elo']
	for pre in ['this_','other_']:
		for key in add_elo_keys:
			all_df[pre+key] = pd.Series(np.zeros(all_df.shape[0]), index=all_df.index)
	# Add conference id keys
	seasons = np.unique(all_df['Season'])
	teams = build_all_teams(years=seasons, all_data=all_data)
	# Build team to conference mapping
	teamConf_map_seasons = {}
	for season in seasons:
		teamConf_map_seasons[season] = {}
		for team in teams:
			if season in team.info:
				cid = float(team.info[season]['ConferenceId'])
				teamConf_map_seasons[season][team.tid] = cid
	this_confid = []
	other_confid = []
	for ttid,otid,season in zip(all_df['this_TeamId'], all_df['other_TeamId'], all_df['Season']):
		this_confid.append(teamConf_map_seasons[season][ttid])
		other_confid.append(teamConf_map_seasons[season][otid])
	all_df['this_ConfId'] = pd.Series(this_confid, index=all_df.index)
	all_df['other_ConfId'] = pd.Series(other_confid, index=all_df.index)
	all_df.to_pickle(os.path.join(default.comp_team_dir, 'all.df'))


def concatenate_team_DataFrames(data_frames=[]):
	"""
	"""
	if len(data_frames) == 0:
		teams = load_json('all.json', fdir=default.comp_team_dir)
		for tid, team in teams.iteritems():
			fname = str(tid) + '_DataFrame.df'
			data_frames.append(pd.read_pickle(os.path.join(default.comp_team_dir, fname)))
	all_df = pd.concat(data_frames)
	return all_df


def team_boxscore_to_array(team, team_id):
	"""
	Convert all games in json format to a pandas array
	"""
	box_keys_unq, game_keys_unq = get_fields()
	game_keys_unq.sort()
	box_keys_unq.sort()
	game_keys = []
	box_keys = []
	# append game information
	for key in game_keys_unq:
		# replace instances of 'home' and 'away' with 'this_' and 'other_'
		new_key = key
		new_key = re.sub("home", "this_", new_key, flags=re.I)
		new_key = re.sub("away", "other_", new_key, flags=re.I)
		game_keys.append(new_key)
	# append team stats keys
	for i in range(0,2):
		pos = 'this_' if i == 0 else 'other_'
		for key in box_keys_unq:
			box_keys.append(pos+key)
	# loop through all games
	box_array = pd.DataFrame(columns=game_keys + box_keys)
	for year, games in team['games'].iteritems():
		this_conf = team['teamInfo'][year]['ConferenceId']
		for gid, game in games.iteritems():
			box_dict = {}
			# decide how to access boxscore data
			is_home = team_is_home(team_id, game['gameinfo'])
			box_dict['is_home'] = 1 if is_home else 0
			this_box_field = 'homeTeam' if is_home else 'awayTeam'
			other_box_field = 'awayTeam' if is_home else 'homeTeam'
			this_game_field = 'Home' if is_home else 'Away'
			other_game_field = 'Away' if is_home else 'Home'
			# set conferences
			try:
				other_id_field = 'AwayTeamId' if is_home else 'HomeTeamId'
				other_id = game['gameinfo'][other_id_field]
				other_team = load_team_json(other_id)
				other_conf = other_team['teamInfo'][year]['ConferenceId']
				box_dict['other_conferenceId'] = other_conf
			except KeyError:
				box_dict['other_conferenceId'] = '-1'
			box_dict['this_conferenceId'] = this_conf
			# loop through all fields
			struct_fields = ['gameinfo', 'boxscore']
			key_fields = [game_keys, box_keys]
			for i, data_field in enumerate(struct_fields):
				is_ginfo = data_field == 'gameinfo'
				# load data
				try:
					data = game[data_field]
				except KeyError:
					UserWarning("Bad %s field found in game %s" % (data_field, gid))
					continue
				# add to array
				keys = key_fields[i]
				for key in keys:
					this_repl = this_game_field if is_ginfo else ''
					other_repl = other_game_field if is_ginfo else ''
					# determine if home/away field is this team
					if re.match("this_", key) is not None:
						data_key = re.sub("this_", this_repl, key)
						try:
							tmp_data = data if is_ginfo else data[this_box_field]
						except KeyError:
							UserWarning("Bad %s field found in game %s" % (data_field, gid))
							continue
					elif re.match("other_", key) is not None:
						data_key = re.sub("other_", other_repl, key)
						try:
							tmp_data = data if is_ginfo else data[other_box_field]
						except KeyError:
							UserWarning("Bad %s field found in game %s" % (data_field, gid))
							continue
					else:
						data_key = key
						tmp_data = data
					# set data
					try:
						box_dict[key] = tmp_data[data_key]
						if key == 'Date' or key == 'DateUtc':
							format = '%Y-%m-%d %H:%M'
							box_dict[key] = datetime.datetime.strptime(box_dict[key], format)
					except KeyError:
						box_dict[key] = '-'
			# Add game Series to DataFrame
			box_array = box_array.append(pd.Series(box_dict), ignore_index=True)
	sorted_box_array = box_array.sort_values(by='DateUtc', ascending=True)
	sorted_box_array = sorted_box_array.reset_index()
	return sorted_box_array


def get_fields():
	"""
	Load unique set of field names into boxscore and gameinfo key lists
	"""
	# boxscore
	box_keys = []
	with open(os.path.join('data', 'boxscore_fields.csv'), 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			box_keys.append(row[0])
	# gameinfo
	game_keys = []
	with open(os.path.join('data', 'gameinfo_fields.csv'), 'rb') as f:
		reader = csv.reader(f)
		for row in reader:
			game_keys.append(row[0])
	return box_keys, game_keys


def team_is_home(team_id, gameinfo):
	"""
	Check team id against home team in gameinfo structure
	"""
	assert (int(team_id) == int(gameinfo['HomeTeamId']) or
		    int(team_id) == int(gameinfo['AwayTeamId']))
	return int(team_id) == int(gameinfo['HomeTeamId'])


def load_team_json(team_id):
	if isinstance(team_id, str):
		name = team_id
	else:
		names = load_json('team_names.json', fdir='data')
		name = names[str(team_id)]
	team = load_json(name+'.json', fdir=default.comp_team_dir)
	return team


def save_all_df_cols():
	all_data = pd.read_pickle(os.path.join(default.comp_team_dir, 'all.df'))
	cols = all_data.columns
	inout = {}
	inout['inputs'] = [c for c in cols]
	inout['outputs'] = [c for c in cols]
	dump_json(inout, 'all_df_fields.json', fdir=default.io_dir)
