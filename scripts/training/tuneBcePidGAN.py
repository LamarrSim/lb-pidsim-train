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

hyperparams = dict()

with open ("config/hyperparams/tuned-bcegan.yaml") as file:
  hyperparams["tune"] = yaml.full_load (file)

with open ("config/hyperparams/fine-tuned-bcegan.yaml") as file:
  hyperparams["fine"] = yaml.full_load (file)

# +----------------------------+
# |    Trainer construction    | 
# +----------------------------+

parser = argparser ("Model fine-tuning")
parser . add_argument ( "-t", "--template", required = True )
parser . add_argument ( "-f", "--finetuning", default = "no", choices = ["yes", "no"] )
args = parser . parse_args()

ft_enabled = (args.finetuning == "yes")
template_name = f"{args.template}"

if ft_enabled:
  model_name = f"{template_name}_bcegan-finetuning"
else:
  model_name = f"{args.model}_{args.particle}_{args.sample}_bcegan-{args.version}"

trainer = GanTrainer ( name = model_name ,
                       export_dir  = config["model_dir"] ,
                       export_name = model_name ,
                       report_dir  = config["report_dir"] ,
                       report_name = model_name )

# +-------------------------+
# |    Optimization step    |
# +-------------------------+

hyperparams = hyperparams["fine"] if ft_enabled else hyperparams["tune"]

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

# +---------------------+
# |    Model loading    |
# +---------------------+

trainer . load_model ( filepath = "{}/{}" . format ( config["model_dir"], template_name ) , 
                       model_to_load = "all" , 
                       enable_reweights = False ,
                       verbose = 1 )

# +--------------------------+
# |    Model construction    |
# +--------------------------+

discriminator = trainer . extract_model ( player = "disc" , 
                                          fine_tuned_layers = None if ft_enabled else hp["fine_tuned_d_layers"] )

add_d_num_layers  = None if ft_enabled else hp["add_d_num_layers"]
add_d_num_nodes   = None if ft_enabled else hp["add_d_num_nodes"]
add_d_alpha_leaky = None if ft_enabled else hp["add_d_alpha_leaky"]

if add_d_num_layers:
  for layer in range (add_d_num_layers):
    discriminator . append ( Dense (add_d_num_nodes, kernel_initializer = "glorot_uniform") )
    discriminator . append ( LeakyReLU (alpha = add_d_alpha_leaky) )

generator = trainer . extract_model ( player = "gen" ,
                                      fine_tuned_layers = None if ft_enabled else hp["fine_tuned_g_layers"] )

add_g_num_layers  = None if ft_enabled else hp["add_g_num_layers"]
add_g_num_nodes   = None if ft_enabled else hp["add_g_num_nodes"]
add_g_alpha_leaky = None if ft_enabled else hp["add_g_alpha_leaky"]

if add_g_num_layers:
  for layer in range (add_g_num_layers):
    generator . append ( Dense (add_g_num_nodes, kernel_initializer = "glorot_uniform") )
    generator . append ( LeakyReLU (alpha = add_g_alpha_leaky) )
    # generator . append ( Dropout (rate = 0.2) )

model = BceGAN ( X_shape = len(trainer.X_vars) , 
                 Y_shape = len(trainer.Y_vars) , 
                 discriminator = discriminator , 
                 generator  = generator , 
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
