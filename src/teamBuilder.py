import os
import re
from src.IOutils import load_json, dump_json, grab_scraper_data
import pandas as pd
import csv


this_year = 2016


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
	try:
		unique_game_keys.remove('Links')
	except ValueError: pass
	try:
		unique_game_keys.remove('AwayTeamInfo')
	except ValueError: pass
	try:
		unique_game_keys.remove('HomeTeamInfo')
	except ValueError: pass
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
	return int(team_id) == int(gameinfo['HomeTeamId'])


def team_boxscore_to_array(team):
	"""
	Converts all games in json format to a pandas array
	"""
	box_keys_unq, game_keys_unq = get_fields()
	box_keys_unq.sort()
	game_keys_unq.sort()
	box_keys = []
	game_keys = []
	# append game information
	for key in game_keys_unq:
		# replace instances of 'home' and 'away' with 'off' and 'def'
		new_key = key
		new_key = re.sub("home", "off", new_key, flags=re.I)
		new_key = re.sub("away", "def", new_key, flags=re.I)
		game_keys.append(new_key)
	# append team stats keys
	for i in range(0,2):
		pos = 'off' if i == 0 else 'def'
		for key in box_keys_unq:
			box_keys.append(pos+' '+key)
	# loop through all games
	for year, games in team['games'].iteritems():
		box_array = pd.DataFrame(columns=game_keys + box_keys)
		for gid, game in games.iteritems():
			box_dict = {}
			is_home = team_is_home(team_id, game['gameinfo'])
			# Game info data
			home_repl = 'off' if is_home else 'def'
			away_repl = 'def' if is_home else 'off'
			# load gameinfo
			try:
				ginfo = game['gameinfo']
			except KeyError:
				UserWarning("Bad gameinfo field found in game %s" % gid)
				continue
			# add game information
			for key, val in ginfo.iteritems():
				if re.search("^links|(Home|Away)teaminfo", key, flags=re.I):
					continue
				new_key = key
				new_key = re.sub("home", home_repl, new_key, flags=re.I)
				new_key = re.sub("away", away_repl, new_key, flags=re.I)
				assert(new_key in game_keys)
				box_dict[new_key] = val
			# Boxscore data
			off_field = 'homeTeam' if is_home else 'awayTeam'
			def_field = 'awayTeam' if is_home else 'homeTeam'
			# load boxscore
			try:
				box = game['boxscore']
			except KeyError:
				UserWarning("Bad boxscore field found in game %s" % gid)
				continue
			# add boxscore stats
			for i in range(0,2):
				pos = 'off' if i == 0 else 'def'
				field = off_field if i == 0 else def_field
				for j, key in enumerate(box_keys_unq):
					bk_idx = i * len(box_keys_unq) + j
					try:
						box_dict[box_keys[bk_idx]] = box[field][key]
					except KeyError:
						box_dict[box_keys[bk_idx]] = "-"
			# Add game Series to DataFrame
			box_array = box_array.append(pd.Series(box_dict), ignore_index=True)
	return box_array.sort_values(by='Date', ascending=True)


def build_team_ids():
	"""
	Maps school name to id #
	"""
	teams = load_json('all.json', fdir=os.path.join('data', 'compiled_team_data'))
	teamid_dict = {}
	for tid, team in teams.iteritems():
		teamid_dict[team['school']] = tid
	dump_json(teamid_dict, 'team_ids.json', fdir='data', indent=4)


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
	Compiles teams from inputted year range into dictionaries.
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