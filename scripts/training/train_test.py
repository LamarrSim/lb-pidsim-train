#from __future__ import annotations

import yaml
import tensorflow as tf

from lb_pidsim_train.utils      import argparser
from lb_pidsim_train.trainers   import GanTrainer
from lb_pidsim_train.algorithms import CramerGAN
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

hyperparams = dict()

with open ("config/hyperparams/cramergan.yaml") as file:
  hyperparams["std"] = yaml.full_load (file)

with open ("config/hyperparams/base-cramergan.yaml") as file:
  hyperparams["base"] = yaml.full_load (file)

# +----------------------------+
# |    Trainer construction    | 
# +----------------------------+

parser = argparser ("Model training")
args = parser . parse_args()

slot = "-" . join ( args.sample . split("-") [:-1] )

model_name = f"{args.model}_{args.particle}_{args.sample}_{args.version}"
model_name += ".sc"   # standard CramerGAN

trainer = GanTrainer ( name = model_name ,
                       export_dir  = config["model_dir"] ,
                       export_name = model_name ,
                       report_dir  = config["report_dir"] ,
                       report_name = model_name )

# +-------------------------+
# |    Optimization step    |
# +-------------------------+

hyperparams = hyperparams["std"]

hp = hyperparams[args.model][args.particle][slot]
# TODO add OptunAPI update

# +-------------------------+
# |    Data for training    |
# +-------------------------+

data_dir  = "/home/mabarbet/PythonFastSim/data"
file_list = [ f"PIDsimu2016_U_{args.particle.lower()}.root" ]
file_list = [ f"{data_dir}/{file_name}" for file_name in file_list ]

trainer . feed_from_root_files ( root_files = file_list , 
                                 X_vars = variables[args.model]["X_vars"][slot] , 
                                 Y_vars = variables[args.model]["Y_vars"][slot] , 
                                 w_var  = None, 
                                 selections = selections[args.model][slot] , 
                                 tree_names = f"make_tuple/tuple_{args.particle.lower()}" , 
                                 chunk_size = hp["chunk_size"] , 
                                 verbose = 1 )

if args.model == "Muon":   # Compute MuonLL to replace MuonMuLL
  trainer._datachunk["MuonLL"] = trainer._datachunk["probe_Brunel_MuonMuLL"] - \
                                 trainer._datachunk["probe_Brunel_MuonBgLL"]
  trainer._Y_vars[-1] = "MuonLL"
  columns = trainer.X_vars + trainer.Y_vars + trainer.w_var if trainer.w_var else trainer.X_vars + trainer.Y_vars
  trainer._datachunk = trainer._datachunk[columns]

# +--------------------------+
# |    Data preprocessing    |
# +--------------------------+

X_preprocessing = variables[args.model]["X_preprocessing"][slot]
Y_preprocessing = variables[args.model]["Y_preprocessing"][slot]

trainer . prepare_dataset ( X_preprocessing = X_preprocessing , 
                            Y_preprocessing = Y_preprocessing , 
                            X_vars_to_preprocess = trainer.X_vars ,
                            Y_vars_to_preprocess = trainer.Y_vars ,
                            enable_reweights = False ,
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


model = CramerGAN ( X_shape = len(trainer.X_vars) , 
                    Y_shape = len(trainer.Y_vars) , 
                    discriminator = discriminator , 
                    generator = generator , 
                    latent_dim = hp["latent_dim"] , 
                    critic_dim = hp["critic_dim"] )

# +---------------------------+
# |    Model configuration    |
# +---------------------------+

d_opt = tf.optimizers.RMSprop ( learning_rate = hp["d_lr"] )
g_opt = tf.optimizers.RMSprop ( learning_rate = hp["g_lr"] )

model . compile ( d_optimizer = d_opt , 
                  g_optimizer = g_opt , 
                  d_updt_per_batch = hp["d_updt_per_batch"] , 
                  g_updt_per_batch = hp["g_updt_per_batch"] ,
                  grad_penalty = hp["grad_penalty"] )

model . summary()

# +-----------------+
# |    Callbacks    |
# +-----------------+

model_saver  = GanModelSaver ( name = model_name , 
                               dirname = config["model_dir"] , 
                               model_to_save = "gen",
                               verbose = 1 )

lr_scheduler = GanExpLrScheduler ( factor = hp["lr_sched_factor"] , 
                                   step = hp["lr_sched_step"] )

# +--------------------+
# |    Run training    |
# +--------------------+

trainer . train_model ( model = model ,
                        batch_size = hp["batch_size"] ,
                        num_epochs = 10,#hp["num_epochs"] ,
                        validation_split = hp["validation_split"] ,
                        scheduler = [model_saver, lr_scheduler] ,
                        verbose = 1 )
