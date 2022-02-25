#from __future__ import annotations

import yaml

from lb_pidsim_train.utils    import argparser
from lb_pidsim_train.trainers import ScikitClassifier
from sklearn.ensemble         import GradientBoostingClassifier


# +---------------------------+
# |    Configuration files    |
# +---------------------------+

with open ("../training/config/config.yaml") as file:
  config = yaml.full_load (file)

with open ("../training/config/datasets.yaml") as file:
  datasets = yaml.full_load (file)

with open ("../training/config/variables.yaml") as file:
  variables = yaml.full_load (file)

with open ("../training/config/selections.yaml") as file:
  selections = yaml.full_load (file)

with open ("../training/config/hyperparams/rating-gbdt.yaml") as file:
  hyperparams = yaml.full_load (file)

# +----------------------------+
# |    Trainer construction    | 
# +----------------------------+

parser = argparser ("Model rating")
parser . add_argument ( "-a", "--algo", required = True )   # TODO add choices
args = parser . parse_args()

model_name = f"{args.algo}_{args.model}_{args.particle}_{args.sample}_{args.version}"
class_name = f"rtg_{model_name}"

trainer = ScikitClassifier ( name = class_name ,
                             model_dir   = config["model_dir"] ,
                             model_name  = model_name ,
                             export_dir  = config["class_dir"] ,
                             export_name = class_name ,
                             report_dir  = config["report_dir"] ,
                             report_name = class_name )

# +-------------------------+
# |    Data for training    |
# +-------------------------+

data_dir  = config["data_dir"]
file_list = datasets[args.model][args.particle][args.sample]
file_list = [ f"{data_dir}/{file_name}" for file_name in file_list ]

hp = hyperparams[args.model][args.particle][args.sample]

trainer . feed_from_root_files ( root_files = file_list , 
                                 X_vars = variables[args.model]["X_vars"][args.sample] , 
                                 Y_vars = variables[args.model]["Y_vars"][args.sample] , 
                                 w_var  = variables[args.model]["w_vars"][args.sample] , 
                                 selections = selections[args.model][args.sample] , 
                                 tree_names = None , 
                                 chunk_size = hp["chunk_size"] , 
                                 verbose = 1 )

# +--------------------------+
# |    Data preprocessing    |
# +--------------------------+

trainer . prepare_dataset ( verbose = 1 )

# +--------------------------+
# |    Model construction    |
# +--------------------------+

model = GradientBoostingClassifier ( loss = hp["loss"] ,
                                     learning_rate = hp["learning_rate"] ,
                                     n_estimators = hp["n_estimators"] ,
                                     criterion = hp["criterion"] ,
                                     max_depth = hp["max_depth"] ,
                                     max_features = hp["max_features"] )

# +--------------------+
# |    Run training    |
# +--------------------+

trainer . train_model ( model = model ,
                        validation_split = hp["validation_split"] ,
                        inverse_transform = hp["inverse_transform"] ,
                        performance_metric = hp["performance_metric"] ,
                        verbose = 1 )
