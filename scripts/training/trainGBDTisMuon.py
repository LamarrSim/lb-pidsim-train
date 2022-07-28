#from __future__ import annotations

import yaml
import numpy as np

from lb_pidsim_train.utils    import argparser
from lb_pidsim_train.trainers import ScikitTrainer
from sklearn.ensemble         import GradientBoostingClassifier


# +---------------------+
# |    Initial setup    |
# +---------------------+

print ( "\n\t\t\t\t\t+--------------------------------------+"   )
print (   "\t\t\t\t\t|                                      |"   )
print (   "\t\t\t\t\t|        Gradient BDT - training       |"   )
print (   "\t\t\t\t\t|                                      |"   )
print (   "\t\t\t\t\t+--------------------------------------+\n" )

parser = argparser ("GBDT training", avoid_arguments = "model")
parser . add_argument ( "-w", "--weights", default = "no", choices = ["yes", "no"] )
args = parser . parse_args()

slot = "-" . join ( args.sample . split("-") [:-1] )
calib_sample = ( "data" in args.sample )

if calib_sample : print ( "[INFO] Calibration samples selected for training" )
else            : print ( "[INFO] Monte Carlo samples selected for training" )

sw_avail = ( args.weights == "yes" )
if sw_avail: print ( "[INFO] sWeighted GBDT training selected" )

# +---------------------------+
# |    Configuration files    |
# +---------------------------+

with open ("config/config.yml") as file:
  config = yaml.full_load (file)

with open ("config/datasets.yml") as file:
  datasets = yaml.full_load (file)

with open ("config/variables.yml") as file:
  variables = yaml.full_load (file)

with open ("config/selections.yml") as file:
  selections = yaml.full_load (file)

with open ("config/hyperparams/ismuon-gbdt.yml") as file:
  hyperparams = yaml.full_load (file)

# +----------------------------+
# |    Trainer construction    | 
# +----------------------------+

model_name = f"isMuon_{args.particle}_{args.sample}_{args.version}"

trainer = ScikitTrainer ( name = model_name ,
                          export_dir  = config["class_dir"] ,
                          export_name = model_name ,
                          report_dir  = config["report_dir"] ,
                          report_name = model_name )

# +-------------------------+
# |    Data for training    |
# +-------------------------+

if calib_sample:
  data_dir = config["data_dir"]["data"]
else:
  data_dir = config["data_dir"]["simu"]

file_list = datasets["isMuon"][args.particle][args.sample]
file_list = [ f"{data_dir}/{file_name}" for file_name in file_list ]

hp = hyperparams[args.particle][args.sample]

trainer . feed_from_root_files ( root_files = file_list , 
                                 X_vars = variables["isMuon"]["X_vars"][slot] , 
                                 Y_vars = variables["isMuon"]["Y_vars"][slot] , 
                                 w_var  = variables["isMuon"]["w_vars"][slot] if sw_avail else None , 
                                 selections = selections["isMuon"][slot] , 
                                 tree_names = None if calib_sample else "make_tuple" ,
                                 chunk_size = hp["chunk_size"] , 
                                 verbose = 1 )

# +--------------------------+
# |    Data preprocessing    |
# +--------------------------+

X_preprocessing = variables["isMuon"]["X_preprocessing"][args.sample]

trainer . prepare_dataset ( X_preprocessing = X_preprocessing , 
                            X_vars_to_preprocess = trainer.X_vars ,
                            verbose = 2 )

# +------------------------+
# |    Prescale weights    |
# +------------------------+

pT = trainer.X[:,0] / np.cosh ( trainer.X[:,1] ) / 1e3
pT = np.c_ [pT]

w_prescale = np.ones_like ( pT )
if args.particle == "Proton":
  w_prescale *= 0.003
  w_prescale [ pT < 3 ] = 1.0
  w_prescale [ (pT >= 3) & (pT < 6) ] = 0.03

trainer._w *= w_prescale

# +--------------------------+
# |    Model construction    |
# +--------------------------+

trainer.params.get ("model", "Gradient BDT")

model = GradientBoostingClassifier ( loss             = trainer.params.get ( "loss"             , hp["loss"]             ) ,
                                     learning_rate    = trainer.params.get ( "learning_rate"    , hp["learning_rate"]    ) ,
                                     n_estimators     = trainer.params.get ( "n_estimators"     , hp["n_estimators"]     ) ,
                                     criterion        = trainer.params.get ( "criterion"        , hp["criterion"]        ) ,
                                     min_samples_leaf = trainer.params.get ( "min_samples_leaf" , hp["min_samples_leaf"] ) ,
                                     max_depth        = trainer.params.get ( "max_depth"        , hp["max_depth"]        ) )

# +--------------------+
# |    Run training    |
# +--------------------+

trainer . train_model ( model = model ,
                        validation_split = hp["validation_split"] ,
                        verbose = 1 )
