import os

schedule_columns = ['Id', 'this_TeamId', 'other_TeamId', 'DateUtc', 'Season',
					'Week', 'Spread', 'OverUnder', 'is_home']

this_elo_fields = ['this_wl_elo', 'this_cf_elo', 'this_off_elo', 'this_def_elo', 
				   'this_poff_elo', 'this_roff_elo', 'this_pdef_elo', 'this_rdef_elo']
other_elo_fields = ['other_wl_elo', 'other_cf_elo', 'other_off_elo', 'other_def_elo',
					'other_poff_elo', 'other_roff_elo', 'other_pdef_elo', 'other_rdef_elo']
all_elo_fields = this_elo_fields + other_elo_fields

season_day_sep = 100
this_year = 2016

avg_score = 26.9
avg_rushing = 160.3
avg_passing = 225.7

# Directories
comp_team_dir = os.path.join('data', 'compiled_team_data')
io_dir = os.path.join('data', 'inout_fields')
prm_dir = os.path.join('data', 'network_params')
data_dir = os.path.join('data', 'data_sets')
elo_dir = os.path.join('data', 'elo')

# Team name mapping
team_alt_mapping = {
	"Army": "Army West Point", 
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
	"Texas-San Antonio": "UTSA"
}
no_team = {
	"Massachusetts": range(2000,2012),
	"UAB": range(2015, 2017),
	"Western Kentucky": range(2000,2007),
	"Appalachian State": range(2000,2014),
	"Georgia Southern": range(2000,2014),
	"Texas State": range(2000,2012),
	"Old Dominion": range(2000,2014),
	"South Alabama": range(2000,2012),
	"Georgia State": range(2000,2013),
	"Texas-San Antonio": range(2000,2012),
	"Charlotte": range(2000,2015),
	"South Florida": [2000],
	"Troy State": [2000],
	"Florida International": range(2000,2005),
	"Florida Atlantic": range(2000,2005)
}