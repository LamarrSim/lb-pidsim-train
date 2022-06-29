#from __future__ import annotations

import yaml
import tensorflow as tf

from lb_pidsim_train.utils      import argparser
from lb_pidsim_train.trainers   import GanTrainer
from lb_pidsim_train.algorithms import BceGAN
from lb_pidsim_train.callbacks  import GanModelSaver, GanExpLrScheduler
from tensorflow.keras.layers    import Dense, LeakyReLU, Dropout


# +---------------------+
# |    Initial setup    |
# +---------------------+

print ( "\n\t\t\t\t\t+-------------------------------+"   )
print (   "\t\t\t\t\t|                               |"   )
print (   "\t\t\t\t\t|        BCE GAN - tuning       |"   )
print (   "\t\t\t\t\t|                               |"   )
print (   "\t\t\t\t\t+-------------------------------+\n" )

parser = argparser ("BceGAN tuning")
parser . add_argument ( "-t", "--template", required = True )
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

with open (f"config/hyperparams/{args.model.lower()}-bcegan.yml") as file:
  hyperparams = yaml.full_load (file) ["tuning"]

# +----------------------------+
# |    Trainer construction    | 
# +----------------------------+

template_name = f"{args.template}"
template_vrs  = template_name.split("_v")[1][4:]

model_name = f"{args.model}_{args.particle}_{args.sample}_{args.version}.{template_vrs}"

if rw_enabled : model_name += ".r"     # reweighting enabled
if sw_avail   : model_name += ".twb"   # tuned BceGAN with weights
else          : model_name += ".tbb"   # tuned base BceGAN

trainer = GanTrainer ( name = model_name ,
                       export_dir  = config["model_dir"] ,
                       export_name = model_name ,
                       report_dir  = config["report_dir"] ,
                       report_name = model_name )

# +-------------------------+
# |    Optimization step    |
# +-------------------------+

hp = hyperparams[args.model][args.particle]
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

# +---------------------+
# |    Model loading    |
# +---------------------+

trainer . load_model ( filepath = "{}/{}" . format ( config["model_dir"], template_name ) , 
                       model_to_load = "all" , 
                       enable_reweights = rw_enabled ,
                       verbose = 1 )

# +--------------------------+
# |    Model construction    |
# +--------------------------+

trainer.params.get ("model", "Binary cross-entropy GAN")

fine_tuned_d_layers = trainer.params.get ( "fine_tuned_d_layers" , hp["fine_tuned_d_layers"] )
add_d_num_layers    = trainer.params.get ( "add_d_num_layers"    , hp["add_d_num_layers"]    )
add_d_num_nodes     = trainer.params.get ( "add_d_num_nodes"     , hp["add_d_num_nodes"]     )
add_d_alpha_leaky   = trainer.params.get ( "add_d_alpha_leaky"   , hp["add_d_alpha_leaky"]   )

discriminator = trainer . extract_model ( player = "disc" , 
                                          fine_tuned_layers = fine_tuned_d_layers )

for layer in range (add_d_num_layers):
  discriminator . append ( Dense (add_d_num_nodes, kernel_initializer = "glorot_uniform") )
  discriminator . append ( LeakyReLU (alpha = add_d_alpha_leaky) )

fine_tuned_g_layers = trainer.params.get ( "fine_tuned_g_layers" , hp["fine_tuned_g_layers"] )
add_g_num_layers    = trainer.params.get ( "add_g_num_layers"    , hp["add_g_num_layers"]    )
add_g_num_nodes     = trainer.params.get ( "add_g_num_nodes"     , hp["add_g_num_nodes"]     )
add_g_alpha_leaky   = trainer.params.get ( "add_g_alpha_leaky"   , hp["add_g_alpha_leaky"]   )

generator = trainer . extract_model ( player = "gen" ,
                                      fine_tuned_layers = fine_tuned_g_layers )

for layer in range (add_g_num_layers):
  generator . append ( Dense (add_g_num_nodes, kernel_initializer = "glorot_uniform") )
  generator . append ( LeakyReLU (alpha = add_g_alpha_leaky) )

model = BceGAN ( X_shape = len(trainer.X_vars) , 
                 Y_shape = len(trainer.Y_vars) , 
                 discriminator = discriminator , 
                 generator  = generator , 
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
                  g_updt_per_batch = trainer.params.get ( "g_updt_per_batch" , hp["g_updt_per_batch"] ) )

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
