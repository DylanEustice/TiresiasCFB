import os
import re
from src.util import load_json, dump_json, grab_scraper_data, debug_assert
import pandas as pd
import csv


this_year = 2016


def build_all_from_scratch(refresh_data=True, years='all'):
	compile_and_save_teams(years=years, refresh_data=refresh_data)
	compile_fields()
	build_team_ids()
	build_team_names()
	build_team_DataFrames()


def compile_and_save_teams(years='all', refresh_data=False):
	"""
	Compile all teams into dictionaries and save as json
	"""
	teams = compile_teams(years=years, refresh_data=refresh_data)
	fdir = os.path.join('data', 'compiled_team_data')
	dump_json(teams, 'all.json', fdir=fdir, indent=4)
	for tid, team in teams.iteritems():
		dump_json(team, team['school'] + '.json', fdir=fdir, indent=4)


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
		years = range(2000, this_year)
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
			print teamid
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
	teams = load_json('all.json', fdir=os.path.join('data', 'compiled_team_data'))
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
	teams = load_json('all.json', fdir=os.path.join('data', 'compiled_team_data'))
	teamid_dict = {}
	for tid, team in teams.iteritems():
		teamid_dict[team['school']] = tid
	dump_json(teamid_dict, 'team_ids.json', fdir='data', indent=4)


def build_team_names():
	"""
	Map school id # to name
	"""
	teams = load_json('all.json', fdir=os.path.join('data', 'compiled_team_data'))
	teamid_dict = {}
	for tid, team in teams.iteritems():
		teamid_dict[tid] = team['school']
	dump_json(teamid_dict, 'team_names.json', fdir='data', indent=4)


def build_team_DataFrames():
	"""
	Build pandas DataFrame for each team and saves as pickle file
	"""
	comp_team_dir = os.path.join('data', 'compiled_team_data')
	teams = load_json('all.json', fdir=comp_team_dir)
	data_frames = []
	for tid, team in teams.iteritems():
		box_arr = team_boxscore_to_array(team, tid)
		fname = tid +'_DataFrame.df'
		box_arr.to_pickle(os.path.join(comp_team_dir, fname))
		data_frames.append(box_arr)
	all_df = pd.concat(data_frames)
	all_df.to_pickle(os.path.join(comp_team_dir, 'all.df'))


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
	for year, games in team['games'].iteritems():
		box_array = pd.DataFrame(columns=game_keys + box_keys)
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
					except KeyError:
						box_dict[key] = '-'
			# Add game Series to DataFrame
			box_array = box_array.append(pd.Series(box_dict), ignore_index=True)
	return box_array.sort_values(by='Date', ascending=True)


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
	debug_assert(int(team_id) == int(gameinfo['HomeTeamId']) or
				 int(team_id) == int(gameinfo['AwayTeamId']))
	return int(team_id) == int(gameinfo['HomeTeamId'])


def load_team_json(team_id):
	names = load_json('team_names.json', fdir='data')
	name = names[str(team_id)]
	team = load_json(name+'.json', fdir=os.path.join('data', 'compiled_team_data'))
	return team