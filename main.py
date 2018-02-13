#Copyright (C) 2018  Seyed Mehran Kazemi, Licensed under the GPL V3; see: <https://www.gnu.org/licenses/gpl-3.0.en.html>
from sys import argv
from trainer_tester import TrainerTester
from params import Params

def getopts(arguments):
	opts = {}  # Empty dictionary to store key-value pairs.
	while arguments:  # While there are arguments left to parse...
		if arguments[0][0] == '-':  # Found a "-name value" pair.
			opts[arguments[0]] = arguments[1]  # Add key and value to the dictionary.
		arguments = arguments[1:]  # Reduce the argument list by copying it starting from index 1.
	return opts

current_models = ["SimplE_ignr", "SimplE_avg", "ComplEx", "TransE"]
current_datasets = ["wn18", "fb15k"]

opts = getopts(argv)
if not "-m" in opts:
	print("Please specify the model name using -m.")
	exit()
if not opts["-m"] in current_models:
	print("Model name not recognized.")
	exit()

if not "-d" in opts:
	print("Please specify the dataset using -d.")
	exit()

if not opts["-d"] in current_datasets:
	print("Dataset not recognized.")
	exit()

model_name = opts["-m"]
dataset = opts["-d"]
params = Params()
params.use_default(dataset=dataset, model=model_name) 
tt = TrainerTester(model_name=model_name, params=params, dataset=dataset)
tt.train_earlystop_test()


