#from __future__ import annotations

import yaml

from lb_pidsim_train.utils    import argparser
from lb_pidsim_train.trainers import ScikitTrainer
from sklearn.ensemble         import GradientBoostingClassifier


# +---------------------------+
# |    Configuration files    |
# +---------------------------+

with open ("config/config.yaml") as file:
  config = yaml.full_load (file)

with open ("config/datasets.yaml") as file:
  datasets = yaml.full_load (file)

with open ("config/variables.yaml") as file:
  variables = yaml.full_load (file)

with open ("config/selections.yaml") as file:
  selections = yaml.full_load (file)

with open ("config/hyperparams/gbdt.yaml") as file:
  hyperparams = yaml.full_load (file)

# +----------------------------+
# |    Trainer construction    | 
# +----------------------------+

parser = argparser ("Model training", avoid_arguments = "model")
args = parser . parse_args()

model_name = f"GBDT_isMuon_{args.particle}_{args.sample}_{args.version}"

trainer = ScikitTrainer ( name = model_name ,
                          export_dir  = config["class_dir"] ,
                          export_name = model_name ,
                          report_dir  = config["report_dir"] ,
                          report_name = model_name )

# +-------------------------+
# |    Optimization step    |
# +-------------------------+

hp = hyperparams["isMuon"][args.particle][args.sample]
# TODO add OptunAPI update

# +-------------------------+
# |    Data for training    |
# +-------------------------+

data_dir  = config["data_dir"]
file_list = datasets["isMuon"][args.particle][args.sample]
file_list = [ f"{data_dir}/{file_name}" for file_name in file_list ]

trainer . feed_from_root_files ( root_files = file_list , 
                                 X_vars = variables["isMuon"]["X_vars"][args.sample] , 
                                 Y_vars = variables["isMuon"]["Y_vars"][args.sample] , 
                                 w_var  = variables["isMuon"]["w_vars"][args.sample] , 
                                 selections = selections["isMuon"][args.sample] , 
                                 tree_names = None , 
                                 chunk_size = hp["chunk_size"] , 
                                 verbose = 1 )

# +--------------------------+
# |    Data preprocessing    |
# +--------------------------+

X_vars_to_preprocess = variables["isMuon"]["X_vars_to_preprocess"][args.sample]

trainer . prepare_dataset ( X_preprocessing = hp["X_preprocessing"] , 
                            X_vars_to_preprocess = X_vars_to_preprocess ,
                            verbose = 1 )

# +--------------------------+
# |    Model construction    |
# +--------------------------+

model = GradientBoostingClassifier ( loss = hp["loss"] ,
                                     learning_rate = hp["learning_rate"] ,
                                     n_estimators = hp["n_estimators"] ,
                                     criterion = hp["criterion"] ,
                                     min_samples_leaf = hp["min_samples_leaf"] ,
                                     max_depth = hp["max_depth"] )

# +--------------------+
# |    Run training    |
# +--------------------+

trainer . train_model ( model = model ,
                        validation_split = hp["validation_split"] ,
                        verbose = 1 )
