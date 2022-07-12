#from __future__ import annotations

import yaml
import tensorflow as tf

from lb_pidsim_train.utils      import argparser
from lb_pidsim_train.trainers   import GanTrainer
from lb_pidsim_train.algorithms import WGAN_ALP
from lb_pidsim_train.callbacks  import GanModelSaver, GanExpLrScheduler
from tensorflow.keras.layers    import Dense, LeakyReLU, Dropout


# +---------------------+
# |    Initial setup    |
# +---------------------+

print ( "\n\t\t\t\t\t+------------------------------------------+"   )
print (   "\t\t\t\t\t|                                          |"   )
print (   "\t\t\t\t\t|        Wasserstein GAN - training        |"   )
print (   "\t\t\t\t\t|                                          |"   )
print (   "\t\t\t\t\t+------------------------------------------+\n" )

parser = argparser ("WGAN training")
parser . add_argument ( "-w", "--weights", default = "no", choices = ["yes", "no"] )
parser . add_argument ( "-r", "--reweighting", default = "no", choices = ["yes", "no"] )
args = parser . parse_args()

slot = "-" . join ( args.sample . split("-") [:-1] )
calib_sample = ( "data" in args.sample )

if calib_sample : print ( "[INFO] Calibration samples selected for training" )
else            : print ( "[INFO] Monte Carlo samples selected for training" )

sw_avail = ( args.weights == "yes" )
if sw_avail: print ( "[INFO] sWeighted GAN training selected" )

rw_enabled = ( args.reweighting == "yes" )
if rw_enabled: print ( "[INFO] Reweighting strategy enabled for training" )

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

with open (f"config/hyperparams/{args.model.lower()}-wgan.yml") as file:
  hyperparams = yaml.full_load (file)
  hyperparams = hyperparams["standard"] if sw_avail else hyperparams["base"]

# +----------------------------+
# |    Trainer construction    | 
# +----------------------------+

model_name = f"{args.model}_{args.particle}_{args.sample}_{args.version}"

if rw_enabled : model_name += ".r"    # reweighting enabled
if sw_avail   : model_name += ".ww"   # WGAN with weights
else          : model_name += ".bw"   # base WGAN

trainer = GanTrainer ( name = model_name ,
                       export_dir  = config["model_dir"] ,
                       export_name = model_name ,
                       report_dir  = config["report_dir"] ,
                       report_name = model_name )

# +-------------------------+
# |    Optimization step    |
# +-------------------------+

hp = hyperparams[args.particle][args.sample]
# TODO add OptunAPI update

# +-------------------------+
# |    Data for training    |
# +-------------------------+

if calib_sample:
  data_dir = config["data_dir"]["data"]
else:
  data_dir = config["data_dir"]["simu"]

file_list = datasets[args.model][args.particle][args.sample]
file_list = [ f"{data_dir}/{file_name}" for file_name in file_list ]

trainer . feed_from_root_files ( root_files = file_list , 
                                 X_vars = variables[args.model]["X_vars"][slot] , 
                                 Y_vars = variables[args.model]["Y_vars"][slot] , 
                                 w_var  = variables[args.model]["w_vars"][slot] if sw_avail else None , 
                                 selections = selections[args.model][slot] , 
                                 tree_names = None if calib_sample else "make_tuple" , 
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

trainer.params.get ("model", "Wasserstein GAN")

d_num_layers  = trainer.params.get ( "d_num_layers"  , hp["d_num_layers"]  ) 
d_num_nodes   = trainer.params.get ( "d_num_nodes"   , hp["d_num_nodes"]   )
d_alpha_leaky = trainer.params.get ( "d_alpha_leaky" , hp["d_alpha_leaky"] )

discriminator = list()
for layer in range (d_num_layers):
  discriminator . append ( Dense (d_num_nodes, kernel_initializer = "glorot_uniform") )
  discriminator . append ( LeakyReLU (alpha = d_alpha_leaky) )

g_num_layers   = trainer.params.get ( "g_num_layers"   , hp["g_num_layers"]  )
g_num_nodes    = trainer.params.get ( "g_num_nodes"    , hp["g_num_nodes"]   )
g_alpha_leaky  = trainer.params.get ( "g_alpha_leaky"  , hp["g_alpha_leaky"] )
g_dropout_rate = trainer.params.get ( "g_dropout_rate" , 0.0 )

generator = list()
for layer in range (g_num_layers):
  generator . append ( Dense (g_num_nodes, kernel_initializer = "glorot_uniform") )
  generator . append ( LeakyReLU (alpha = g_alpha_leaky) )
  generator . append ( Dropout (rate = g_dropout_rate) )

model = WGAN_ALP ( X_shape = len(trainer.X_vars) , 
                   Y_shape = len(trainer.Y_vars) , 
                   discriminator = discriminator , 
                   generator = generator , 
                   latent_dim = trainer.params.get ( "latent_dim" , hp["latent_dim"] ) )

# +---------------------------+
# |    Model configuration    |
# +---------------------------+

trainer.params.get ("d_optimizer", "RMSprop")
trainer.params.get ("g_optimizer", "RMSprop")

d_opt = tf.optimizers.RMSprop ( learning_rate = trainer.params.get ( "d_lr0" , hp["d_lr"] ) )
g_opt = tf.optimizers.RMSprop ( learning_rate = trainer.params.get ( "g_lr0" , hp["g_lr"] ) )

model . compile ( d_optimizer = d_opt , 
                  g_optimizer = g_opt , 
                  d_updt_per_batch = trainer.params.get ( "d_updt_per_batch" , hp["d_updt_per_batch"] ) , 
                  g_updt_per_batch = trainer.params.get ( "g_updt_per_batch" , hp["g_updt_per_batch"] ) ,
                  v_adv_dir_updt = trainer.params.get ( "v_adv_dir_updt" , hp["v_adv_dir_updt"] ) ,
                  adv_lp_penalty = trainer.params.get ( "adv_lp_penalty" , hp["adv_lp_penalty"] ) )

model . summary()

# +-----------------+
# |    Callbacks    |
# +-----------------+

model_saver  = GanModelSaver ( name = model_name , 
                               dirname = config["model_dir"] , 
                               model_to_save = "all" ,
                               verbose = 1 )

lr_scheduler = GanExpLrScheduler ( factor = trainer.params.get ( "lr_sched_factor" , hp["lr_sched_factor"] ) , 
                                   step   = trainer.params.get ( "lr_sched_step"   , hp["lr_sched_step"]   ) )

# +--------------------+
# |    Run training    |
# +--------------------+

trainer . train_model ( model = model ,
                        batch_size = hp["batch_size"] ,
                        num_epochs = hp["num_epochs"] ,
                        validation_split = hp["validation_split"] ,
                        scheduler = [model_saver, lr_scheduler] ,
                        verbose = 1 )
