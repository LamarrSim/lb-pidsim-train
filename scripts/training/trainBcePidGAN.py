#from __future__ import annotations

import yaml
import tensorflow as tf

from lb_pidsim_train.utils      import argparser
from lb_pidsim_train.trainers   import GanTrainer
from lb_pidsim_train.algorithms import BceGAN
from lb_pidsim_train.callbacks  import GanModelSaver, GanExpLrScheduler
from tensorflow.keras.layers    import Dense, LeakyReLU, Dropout


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

with open ("config/hyperparams/bcegan.yaml") as file:
  hyperparams = yaml.full_load (file)

# +----------------------------+
# |    Trainer construction    | 
# +----------------------------+

parser = argparser ("Model training")
parser . add_argument ( "-r", "--reweighting", default = "no", choices = ["yes", "no"] )
args = parser . parse_args()

model_name = f"{args.model}_{args.particle}_{args.sample}_{args.version}"

rw_enabled = (args.reweighting == "yes")

if rw_enabled: model_name += ".r"
model_name += ".sb"   # standard BceGAN

trainer = GanTrainer ( name = model_name ,
                       export_dir  = config["model_dir"] ,
                       export_name = model_name ,
                       report_dir  = config["report_dir"] ,
                       report_name = model_name )

# +-------------------------+
# |    Optimization step    |
# +-------------------------+

hp = hyperparams[args.model][args.particle][args.sample]
# TODO add OptunAPI update

# +-------------------------+
# |    Data for training    |
# +-------------------------+

data_dir  = config["data_dir"]
file_list = datasets[args.model][args.particle][args.sample]
file_list = [ f"{data_dir}/{file_name}" for file_name in file_list ]

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

X_preprocessing = variables[args.model]["X_preprocessing"][args.sample]
Y_preprocessing = variables[args.model]["Y_preprocessing"][args.sample]

trainer . prepare_dataset ( X_preprocessing = X_preprocessing , 
                            Y_preprocessing = Y_preprocessing , 
                            X_vars_to_preprocess = trainer.X_vars ,
                            Y_vars_to_preprocess = trainer.Y_vars ,
                            enable_reweights = rw_enabled ,
                            verbose = 1 )

# +--------------------------+
# |    Model construction    |
# +--------------------------+

d_num_layers  = hp["d_num_layers"]
d_num_nodes   = hp["d_num_nodes"]
d_alpha_leaky = hp["d_alpha_leaky"]

discriminator = list()
for layer in range (d_num_layers):
  discriminator . append ( Dense (d_num_nodes, kernel_initializer = "glorot_uniform") )
  discriminator . append ( LeakyReLU (alpha = d_alpha_leaky) )

g_num_layers  = hp["g_num_layers"]
g_num_nodes   = hp["g_num_nodes"]
g_alpha_leaky = hp["g_alpha_leaky"]

generator = list()
for layer in range (g_num_layers):
  generator . append ( Dense (g_num_nodes, kernel_initializer = "glorot_uniform") )
  generator . append ( LeakyReLU (alpha = g_alpha_leaky) )
  generator . append ( Dropout (rate = 0.1) )

model = BceGAN ( X_shape = len(trainer.X_vars) , 
                 Y_shape = len(trainer.Y_vars) , 
                 discriminator = discriminator , 
                 generator = generator , 
                 latent_dim = hp["latent_dim"] )

# +---------------------------+
# |    Model configuration    |
# +---------------------------+

d_opt = tf.optimizers.RMSprop ( learning_rate = hp["d_lr"] )
g_opt = tf.optimizers.RMSprop ( learning_rate = hp["g_lr"] )

model . compile ( d_optimizer = d_opt , 
                  g_optimizer = g_opt , 
                  d_updt_per_batch = hp["d_updt_per_batch"] , 
                  g_updt_per_batch = hp["g_updt_per_batch"] )

model . summary()

# +-----------------+
# |    Callbacks    |
# +-----------------+

model_saver  = GanModelSaver ( name = model_name , 
                               dirname = config["model_dir"] , 
                               model_to_save = "all" ,
                               verbose = 1 )

lr_scheduler = GanExpLrScheduler ( factor = hp["lr_sched_factor"] , 
                                   step = hp["lr_sched_step"] )

# +--------------------+
# |    Run training    |
# +--------------------+

trainer . train_model ( model = model ,
                        batch_size = hp["batch_size"] ,
                        num_epochs = hp["num_epochs"] ,
                        validation_split = hp["validation_split"] ,
                        scheduler = [model_saver, lr_scheduler] ,
                        verbose = 1 )
