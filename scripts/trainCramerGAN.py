import numpy as np
import pandas as pd
import tensorflow as tf


# +---------------------------+
# |    Configuration files    |
# +---------------------------+

import yaml

with open ("config/config.yaml") as file:
  config = yaml.full_load (file)

with open ("config/datasets.yaml") as file:
  datasets = yaml.full_load (file)

with open ("config/variables.yaml") as file:
  variables = yaml.full_load (file)

with open ("config/selections.yaml") as file:
  selections = yaml.full_load (file)

with open ("config/hyperparams/cramergan.yaml") as file:
  hyperparams = yaml.full_load (file)

# +----------------------------+
# |    Trainer construction    | 
# +----------------------------+

from lb_pidsim_train.utils import argparser
from lb_pidsim_train.trainers import GanTrainer

parser = argparser ("Model training")
args = parser . parse_args()

model_name = f"CramerGAN_{args.model}_{args.particle}_{args.sample}_{args.version}"

trainer = GanTrainer ( name = model_name ,
                       export_dir  = config["model_dir"] ,
                       export_name = model_name ,
                       report_dir  = config["report_dir"] ,
                       report_name = model_name )

# +-------------------------+
# |    Data for training    |
# +-------------------------+

data_dir  = config["data_dir"]
file_list = datasets[args.model][args.particle][args.sample]
file_list = [ f"{data_dir}/{file_name}" for file_name in file_list ]

hp = hyperparams[args.model][args.particle][args.sample]
print (hp)
# TODO add OptunAPI update

trainer . feed_from_root_files ( root_files = file_list , 
                                 X_vars = variables[args.model]["X_vars"][args.sample] , 
                                 Y_vars = variables[args.model]["Y_vars"][args.sample] , 
                                 w_var  = variables[args.model]["w_vars"][args.sample] , 
                                 selections = selections[args.model][args.sample] , 
                                 tree_names = None, 
                                 chunk_size = hp["chunk_size"], 
                                 verbose = 1)

# +--------------------------+
# |    Data preprocessing    |
# +--------------------------+

trainer . prepare_dataset ( X_preprocessing = hp["X_preprocessing"] , 
                            Y_preprocessing = hp["Y_preprocessing"] , 
                            X_vars_to_preprocess = variables[args.model]["X_vars_to_preprocess"][args.sample] ,
                            Y_vars_to_preprocess = variables[args.model]["Y_vars_to_preprocess"][args.sample] ,
                            verbose = 1)










files_list = datasets [args.sample] [args.model] [args.particle]
data_files = [ datasets['data_dir'] + file_name for file_name in files_list ]

selection = selections [args.model] [args.sample]
vars_dict = variables  [args.model] [args.sample]

with open ('config/hyperparams/db/{}.yaml' . format (model_name)) as file:
  hp = yaml.full_load (file)


###################################
##  Define generator architecture
###################################

from core.utils import getBaseLayer
from tensorflow.keras.layers import Dropout

gen_layers = list()

num_gen_layers = int ( hp['num_gen_layers'] )
num_gen_nodes  = int ( hp['num_gen_nodes' ] )
gen_activation = str ( hp['gen_activation'] )
gen_dropout  = float ( hp['gen_dropout'] )

for layer in range (num_gen_layers):
  gen_layers += getBaseLayer ( num_gen_nodes, gen_activation )
  gen_layers += [ Dropout (gen_dropout) ]


#######################################
##  Define discriminator architecture
#######################################

disc_layers = list()

num_disc_layers = int ( hp['num_disc_layers'] )
num_disc_nodes  = int ( hp['num_disc_nodes' ] )
disc_activation = str ( hp['disc_activation'] )
disc_dropout  = float ( hp['disc_dropout'] )

for layer in range (num_disc_layers):
  disc_layers += getBaseLayer ( num_disc_nodes, disc_activation )
  disc_layers += [ Dropout (disc_dropout) ]


##############################
##  Setup training procedure
##############################

from core import PidGanTrainer as Trainer

trainer = Trainer ( 
                    name        = model_name               ,
                    data_files  = data_files               ,
                    selection   = selection                ,
                    X_vars      = vars_dict['X_vars']      ,
                    Y_vars      = vars_dict['Y_vars']      ,
                    w_var       = vars_dict['w_vars']      , 
                    chunk_size  = int ( hp['chunk_size'] ) ,
                    export_dir  = args.exp_dir             ,
                    export_name = model_name               , 
                    report_dir  = args.rep_dir             ,
                  )

trainer . build (
                  all_hparams          = hp                          ,
                  preprocessing        = str ( hp['preprocessing'] ) ,
                  gan_algorithm        = str ( hp['gan_algorithm'] ) ,
                  generator_layers     = gen_layers                  ,
                  discriminator_layers = disc_layers                 ,
                )

trainer . learn (
                  batch_size    = int   ( hp['batch_size']    ) ,
                  num_epochs    = int   ( hp['num_epochs']    ) ,
                  num_checks    = int   ( hp['num_checks']    ) ,
                  disc_updates  = int   ( hp['disc_updates' ] ) ,
                  learning_rate = float ( hp['learning_rate'] ) ,
                  scheduling_lr = bool  ( hp['scheduling_lr'] ) ,
                  optimizer     = str   ( hp['optimizer']     ) ,
                )