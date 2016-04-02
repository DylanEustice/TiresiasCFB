import os
import re
from src.IOutils import load_json, dump_json, grab_scraper_data


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
		years = range(2000, 2016)
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
					teams[teamid][str(year)]['games'][gameid] = {}
					gameinfo = load_json('gameinfo_'+gameid+'.json', fdir=game_path)
					try:
						boxscore = load_json('boxscore.json', fdir=game_path)
					except IOError:
						boxscore = None
					try:
						playerstats = load_json('playerstats.json', fdir=game_path)
					except IOError:
						playerstats = None
					teams[teamid][str(year)]['games'][gameid]['gameinfo'] = gameinfo
					teams[teamid][str(year)]['games'][gameid]['boxscore'] = boxscore
					teams[teamid][str(year)]['games'][gameid]['playerstats'] = playerstats
	return teams


def init_team(teams, newteam):
	teams[newteam['Id']] = {}
	teams[newteam['Id']]['school'] = newteam['School']
	teams[newteam['Id']]['name'] = newteam['Name']
	teams[newteam['Id']]['abbr'] = newteam['Abbr']
	teams[newteam['Id']]['primaryColor'] = newteam['PrimaryColor']
	teams[newteam['Id']]['secondaryColor'] = newteam['SecondaryColor']


def setup_team_year(year, teams, newteam):
	teams[newteam['Id']][str(year)] = {}
	teams[newteam['Id']][str(year)]['teamInfo'] = newteam
	teams[newteam['Id']][str(year)]['games'] = {}