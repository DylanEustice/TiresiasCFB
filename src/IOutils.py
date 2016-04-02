import os
import json


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
		json.dump(data, f, indent=indent)


def load_json(fname, fdir='.'):
	"""
	Reads data from file. 
	NOTE: Reads from text file, not binary.
	"""
	with open(os.path.join(fdir, fname), 'r') as f:
		return json.load(f)	