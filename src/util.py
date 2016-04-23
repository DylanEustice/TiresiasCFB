import os
import json
import shutil
import errno


def debug_assert(condition):
	try:
		assert(condition)
	except AssertionError:
		import pdb
		pdb.set_trace()
		print "Debugger set. Enter 'u' to go up in stack frame"


def ensure_path(path):
	"""
	Make sure os path exists, create it if not
	"""
	try:
		os.makedirs(path)
	except OSError as exception:
		if exception.errno != errno.EEXIST:
			raise


def dump_json(data, fname, fdir='.', indent=None):
	"""
	Save data to file. 
	NOTE: Writes as text file, not binary.
	"""
	ensure_path(fdir)
	with open(os.path.join(fdir, fname), 'w') as f:
		json.dump(data, f, indent=indent, sort_keys=True)


def load_json(fname, fdir='.'):
	"""
	Reads data from file. 
	NOTE: Reads from text file, not binary.
	"""
	with open(os.path.join(fdir, fname), 'r') as f:
		return json.load(f)


def copy_dir(src, dst):
	"""
	Attempt to copy directory, on failure copy file. Will overwrite
	any files in dst.
	"""
	# Remove destination directory if already exists
	if os.path.exists(dst):
		shutil.rmtree(dst)
	# Copy directory over
	try:
		shutil.copytree(src, dst)
	except OSError as exc:
		if exc.errno == errno.ENOTDIR:
			shutil.copy(src, dst)
		else:
			raise Exception()


def grab_scraper_data(src=os.path.join('..','BarrelRollCFBData','data'),
					  dst=os.path.join('data')):
	"""
	Copy in data directory from BarrelRollCFBData
	"""
	copy_dir(src, dst)


def load_team_DataFrame(team_id, path_to_data='.'):
	fname = str(team_id) + '_DataFrame.df'
	return pd.read_pickle(os.path.join(path_to_data, 'data', 'compiled_team_data', fname))