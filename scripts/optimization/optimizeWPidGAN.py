import yaml
import socket
import tensorflow as tf
import hopaas_client as hpc

from lb_pidsim_train.utils import argparser
from lb_pidsim_train.trainers import GanTrainer
from lb_pidsim_train.algorithms.gan import WGAN_ALP
from lb_pidsim_train.callbacks.gan  import ExpLrScheduler, HopaasModelSaver
from lb_pidsim_train.callbacks import HopaasReporter
from tensorflow.keras.layers import Dense, LeakyReLU, Dropout


# +---------------------+
# |    Initial setup    |
# +---------------------+

print ( "\n\t\t\t\t\t+----------------------------------------------+"   )
print (   "\t\t\t\t\t|                                              |"   )
print (   "\t\t\t\t\t|        Wasserstein GAN - optimization        |"   )
print (   "\t\t\t\t\t|                                              |"   )
print (   "\t\t\t\t\t+----------------------------------------------+\n" )

parser = argparser ("WGAN optimization")
parser . add_argument ( "-w", "--weights", default = "no", choices = ["yes", "no"] )
parser . add_argument ( "-r", "--reweighting", default = "no", choices = ["yes", "no"] )
parser . add_argument ( "-n", "--node_name", required = True )
parser . add_argument ( "-j", "--num_jobs", required = True )
args = parser . parse_args()

slot = "-" . join ( args.sample . split("-") [:-1] )
calib_sample = ( "data" in args.sample )

if calib_sample : print ( "[INFO] Calibration samples selected for training" )
else            : print ( "[INFO] Monte Carlo samples selected for training" )

sw_avail = ( args.weights == "yes" )
if sw_avail: print ( "[INFO] sWeighted GAN training selected" )

rw_enabled = ( args.reweighting == "yes" )
if rw_enabled: print ( "[INFO] Reweighting strategy enabled for training" )

num_jobs = int ( args.num_jobs )

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
  hyperparams = hyperparams["with-weights"] if sw_avail else hyperparams["no-weights"]

# +-----------------------------+
# |    Client initialization    | 
# +-----------------------------+

server_address = config["hopaas"]["address"]
server_port    = config["hopaas"]["port"]

client  = hpc.Client ( server = f"{server_address}:{server_port}" ,
                       token  = config["hopaas"]["token"] )

# +--------------------+
# |    Model naming    | 
# +--------------------+

base_model_name = f"{args.model}_{args.particle}_{args.sample}_{args.version}"

if rw_enabled : base_model_name += "-r"    # reweighting enabled
if sw_avail   : base_model_name += "-ww"   # WGAN with weights
else          : base_model_name += "-nw"   # WGAN without weights

base_model_name += "-opt"

# +----------------------+
# |    Study creation    | 
# +----------------------+

hp = hyperparams[args.particle][args.sample]
properties = hp.copy()
properties . update ( dict ( d_lr = hpc.suggestions.LogUniform (1e-5, 1e-3) ,
                             g_lr = hpc.suggestions.LogUniform (1e-5, 1e-3) ,
                             duxb = hpc.suggestions.Int (1,5) ,
                             adv_lp = hpc.suggestions.LogUniform (1e1, 1e3) ,
                             bs_factor = hpc.suggestions.Int (1,4) ) )

my_address = socket.gethostbyname(socket.gethostname())
my_node_name = f"{args.node_name}"

study = hpc.Study ( name = base_model_name ,
                    properties = properties ,
                    special_properties = dict ( address = my_address ,
                                                node_name = my_node_name ) ,
                    direction = "maximize" ,   # corresponds to difficulties in gen/ref separation
                    pruner  = hpc.pruners.MedianPruner ( n_startup_trials = 20 ,
                                                         n_warmup_steps = 20 ,
                                                         interval_steps = 1 ,
                                                         n_min_trials = 10 ) ,
                    sampler = hpc.samplers.TPESampler ( n_startup_trials = 20 ) ,
                    client  = client )

for iTrial in range(num_jobs):

  with study.trial() as trial:

    print(f"\n{'< ' * 30} Trial n. {trial.id} {' >' * 30}\n")

    # +----------------------------+
    # |    Trainer construction    | 
    # +----------------------------+

    model_name = f"{base_model_name}_suid{study.study_id[:4]}-trial{trial.id:0>4}"

    trainer = GanTrainer ( name = model_name ,
                           export_dir  = "{}/optimization_studies/{}" . format (config["model_dir"], args.model) ,
                           export_name = model_name ,
                           report_dir  = "{}/optimization_studies/{}" . format (config["report_dir"], args.model) ,
                           report_name = model_name )

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
    g_dropout_rate = trainer.params.get ( "g_dropout_rate" , 0.1 )

    generator = list()
    for layer in range (g_num_layers):
      generator . append ( Dense (g_num_nodes, kernel_initializer = "glorot_uniform") )
      generator . append ( LeakyReLU (alpha = g_alpha_leaky) )
      generator . append ( Dropout (rate = g_dropout_rate) )

    classifier = list()
    for layer in range(5):
      classifier . append ( Dense (64, kernel_initializer = "glorot_uniform") )
      classifier . append ( LeakyReLU (alpha = 0.1) )
      classifier . append ( Dropout (rate = 0.1) )

    model = WGAN_ALP ( X_shape = len(trainer.X_vars) , 
                       Y_shape = len(trainer.Y_vars) , 
                       discriminator = discriminator , 
                       generator = generator ,
                       classifier = classifier ,
                       latent_dim = trainer.params.get ( "latent_dim" , hp["latent_dim"] ) )

    # +---------------------------+
    # |    Model configuration    |
    # +---------------------------+

    trainer.params.get ("d_optimizer", "RMSprop")
    trainer.params.get ("g_optimizer", "RMSprop")

    d_opt = tf.optimizers.RMSprop ( learning_rate = trainer.params.get ( "d_lr0" , trial.d_lr ) )
    g_opt = tf.optimizers.RMSprop ( learning_rate = trainer.params.get ( "g_lr0" , trial.g_lr ) )
    c_opt = tf.optimizers.Adam ( learning_rate = 0.0001 )

    model . compile ( d_optimizer = d_opt , 
                      g_optimizer = g_opt , 
                      c_optimizer = c_opt ,
                      d_updt_per_batch = trial.duxb , 
                      g_updt_per_batch = trainer.params.get ( "g_updt_per_batch" , hp["g_updt_per_batch"] ) ,
                      v_adv_dir_updt = trainer.params.get ( "v_adv_dir_updt" , hp["v_adv_dir_updt"] ) ,
                      adv_lp_penalty = trial.adv_lp )

    # +-----------------+
    # |    Callbacks    |
    # +-----------------+

    lr_scheduler = ExpLrScheduler ( factor = trainer.params.get ( "lr_sched_factor" , hp["lr_sched_factor"] ) , 
                                    step   = trainer.params.get ( "lr_sched_step"   , hp["lr_sched_step"]   ) )

    pruner = HopaasReporter ( trial = trial ,
                              loss = "val_c_loss" ,
                              pruning = True ,
                              step = 1 ,
                              timeout = 20000 )

    model_saver = HopaasModelSaver ( trial = trial ,
                                     name = model_name , 
                                     dirname = trainer.export_dir , 
                                     model_to_save = "gen" ,
                                     min_trials_to_save = 20 ,
                                     verbose = 1 )

    # +--------------------+
    # |    Run training    |
    # +--------------------+

    trainer . train_model ( model = model ,
                            batch_size = 256 * trial.bs_factor ,
                            num_epochs = hp["num_epochs"] ,
                            validation_split = hp["validation_split"] ,
                            callbacks = [lr_scheduler, pruner, model_saver] ,
                            produce_report = False ,
                            verbose = 1 )
