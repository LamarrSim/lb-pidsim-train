#from __future__ import annotations

import os
import yaml
import pickle
import datetime
import numpy as np
import tensorflow as tf

import matplotlib.font_manager
import matplotlib.pyplot as plt
import mplhep as hep

from lb_pidsim_train.utils import argparser
from lb_pidsim_train.trainers import DataHandler
from lb_pidsim_train.utils import PidsimColTransformer


NP_FLOAT = np.float32
"""Default data-type for arrays."""

TF_FLOAT = tf.float32
"""Default data-type for tensors."""


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

with open ("../training/config/validation.yaml") as file:
  validation = yaml.full_load (file)

# +--------------------+
# |    Data loading    | 
# +--------------------+

parser = argparser ("Model validation")
parser . add_argument ( "-r", "--reweighting", default = "no", choices = ["yes", "no"] )
args = parser . parse_args()

dh = DataHandler()

data_dir  = config["data_dir"]
file_list = datasets[args.model][args.particle][args.sample]
file_list = [ f"{data_dir}/{file_name}" for file_name in file_list ]

chunk_size = validation[args.model][args.particle][args.sample]["chunk_size"]

dh.feed_from_root_files ( root_files = file_list , 
                          X_vars = variables[args.model]["X_vars"][args.sample] , 
                          Y_vars = variables[args.model]["Y_vars"][args.sample] , 
                          w_var  = variables[args.model]["w_vars"][args.sample] , 
                          selections = selections[args.model][args.sample] , 
                          tree_names = None , 
                          chunk_size = chunk_size , 
                          verbose = 1 )

dh.prepare_dataset()

# +--------------------------+
# |    Preprocessing step    |
# +--------------------------+

model_dir  = config["model_dir"]
model_name = f"{args.model}_{args.particle}_{args.sample}_{args.version}"

file_X = f"{model_dir}/{model_name}/transform_X.pkl"
scaler_X = PidsimColTransformer ( pickle.load ( open (file_X, "rb") ) )

file_Y = f"{model_dir}/{model_name}/transform_Y.pkl"
scaler_Y = PidsimColTransformer ( pickle.load ( open (file_Y, "rb") ) )

rw_enabled = (args.reweighting == "yes")

if rw_enabled:
  file_w = f"{model_dir}/{model_name}/saved_reweighter"
  reweighter = tf.keras.models.load_model ( file_w )

# +---------------------+
# |    Generate data    |
# +---------------------+

generator  = tf.keras.models.load_model ( f"{model_dir}/{model_name}/saved_generator" )

batch_size = len(dh.X)
latent_dim = generator.input_shape[1] - dh.X.shape[1]

X_gen = scaler_X . transform ( dh.X )
X_gen = tf.convert_to_tensor ( X_gen, dtype = TF_FLOAT )
latent_tensor = tf.random.normal ( shape = (batch_size, latent_dim), dtype = TF_FLOAT )
input_tensor  = tf.concat ( [X_gen, latent_tensor], axis = 1 )

Y_gen = generator ( input_tensor ) . numpy() . astype (NP_FLOAT)
Y_gen = scaler_Y . inverse_transform ( Y_gen )

if rw_enabled:
  w_gen = reweighter(X_gen) . numpy() . astype (NP_FLOAT)
else:
  w_gen = np.copy (dh.w) . astype (NP_FLOAT)

# +-----------------------+
# |    Histograms info    |
# +-----------------------+

var_info = {
    "probe_Brunel_RichDLLe"            : [ np.linspace (-150, 150, 100) , r"Rich Differential Log-Likelihood ($e-\pi$)"   , "RichDLLe"  ] ,
    "probe_Brunel_RichDLLmu"           : [ np.linspace (-150, 150, 100) , r"Rich Differential Log-Likelihood ($\mu-\pi$)" , "RichDLLmu" ] ,
    "probe_Brunel_RichDLLk"            : [ np.linspace (-150, 150, 100) , r"Rich Differential Log-Likelihood ($K-\pi$)"   , "RichDLLk"  ] ,
    "probe_Brunel_RichDLLp"            : [ np.linspace (-150, 150, 100) , r"Rich Differential Log-Likelihood ($p-\pi$)"   , "RichDLLp"  ] ,
    "probe_Brunel_MuonMuLL"            : [ np.linspace (-9, 1, 100) , "Muon-detector Signal Likelihood"     , "MuonMuLL" ] ,
    "probe_Brunel_MuonBgLL"            : [ np.linspace (-9, 1, 100) , "Muon-detector Background Likelihood" , "MuonBgLL" ] ,
    "probe_Brunel_PIDe"                : [ np.linspace (-25 , 15 , 100) , r"Combined Differential Log-Likelihood ($e-\pi$)"   , "PIDe"  ] ,
    "probe_Brunel_PIDmu"               : [ np.linspace (-20 , 20 , 100) , r"Combined Differential Log-Likelihood ($\mu-\pi$)" , "PIDmu" ] ,
    "probe_Brunel_PIDK"                : [ np.linspace (-150, 150, 100) , r"Combined Differential Log-Likelihood ($K-\pi$)"   , "PIDK"  ] ,
    "probe_Brunel_PIDp"                : [ np.linspace (-150, 150, 100) , r"Combined Differential Log-Likelihood ($p-\pi$)"   , "PIDp"  ] ,
    "probe_Brunel_MC15TuneV1_ProbNNe"  : [ np.linspace (0, 1, 100) , "ProbNNe (Tune MC15V1)"  , "ProbNNe"  ] ,
    "probe_Brunel_MC15TuneV1_ProbNNmu" : [ np.linspace (0, 1, 100) , "ProbNNmu (Tune MC15V1)" , "ProbNNmu" ] ,
    "probe_Brunel_MC15TuneV1_ProbNNk"  : [ np.linspace (0, 1, 100) , "ProbNNk (Tune MC15V1)"  , "ProbNNk"  ] ,
    "probe_Brunel_MC15TuneV1_ProbNNp"  : [ np.linspace (0, 1, 100) , "ProbNNp (Tune MC15V1)"  , "ProbNNp"  ] ,
    "probe_Brunel_MC15TuneV1_ProbNNpi" : [ np.linspace (0, 1, 100) , "ProbNNpi (Tune MC15V1)" , "ProbNNpi" ] ,
  }

sample_info = {
    "2016MagUp"   : r"$2016~(1.67~\rm{fb}^{-1})~\it{MagUp}$"   ,
    "2016MagDown" : r"$2016~(1.67~\rm{fb}^{-1})~\it{MagDown}$" ,
  }

# +-------------------------+
# |    Plotting function    |
# +-------------------------+

def plot_histogram ( x_ref , 
                     x_gen , 
                     bins  = 100 ,
                     w_ref = None ,
                     w_gen = None ,
                     density = False ,
                     label_x = None  ,
                     label_y_ref = None ,
                     label_y_gen = None ,
                     sub_label = None  ,
                     var_name  = None  ,
                     particle  = None  ,
                     sample    = None  ,
                     log_scale = False ,
                     verbose = 0 ) -> None:
  h_ref, bins_ref = np.histogram ( x_ref , bins = bins     , weights = w_ref, density = density )
  h_gen, bins_gen = np.histogram ( x_gen , bins = bins_ref , weights = w_gen, density = density )

  ## y-axis boundaries
  y_min = min ( h_ref[h_ref>0].min() , h_gen[h_gen>0].min() )
  y_max = max ( h_ref[h_ref>0].max() , h_gen[h_gen>0].max() )
  if log_scale:
    y_min  = min ( 1 , y_min )
    y_max *= 15
  else:
    y_min  = min ( 0 , y_min )
    y_max += 0.25 * y_max

  ## Matplotlib set-up
  plt.style.use ( hep.style.LHCb2 )
  plt.rcParams["axes.linewidth"]    = 1.3
  plt.rcParams["xtick.major.width"] = 1
  plt.rcParams["ytick.major.width"] = 1
  plt.rcParams["xtick.minor.width"] = 1
  plt.rcParams["ytick.minor.width"] = 1
  plt.rcParams["xtick.major.size"]  = 10
  plt.rcParams["ytick.major.size"]  = 10
  plt.rcParams["xtick.minor.size"]  = 5
  plt.rcParams["ytick.minor.size"]  = 5

  ## Figure size and labels
  plt.figure ( figsize = (7,9), dpi = 100 )
  fig, ax = plt.subplots()
  ax.set_xlabel ( f"{label_x}"          , fontname = "serif" , size = 32 )
  ax.set_ylabel ( "Weighted candidates" , fontname = "serif" , size = 32 )

  ## Histogram plots
  hep.histplot  ( h_ref, bins_ref, histtype = "fill", color = "#0571b0", label = label_y_ref )
  hep.histplot  ( h_gen, bins_gen, histtype = "step", color = "#ca0020", lw = 2.5, label = label_y_gen )

  ## Text annotations
  hep.lhcb.text ( "preliminary", loc = 0, fontname = "serif", fontsize = 28, italic = [False, False] )
  if (particle is not None):
    ax.annotate ( f"{particle} track candidates" , 
                  ha = "right" , va = "bottom" ,
                  xy = (0.999, 1.005), xycoords = "axes fraction" ,
                  fontname = "serif" , fontsize = 28 )
  if (sample is not None):
    ax.annotate ( f"{sub_label}" , 
                  ha = "left" , va = "top" ,
                  xy = (0.05,0.95) , xycoords = "axes fraction" ,
                  fontname = "serif" , fontsize = 28 )

  ## Saving options
  img_dir = "./images"
  if not os.path.exists (img_dir): os.makedirs (img_dir)
  if (var_name is not None) and (particle is not None) and (sample is not None):
    fig_name = f"{var_name}_{particle}_{sample}"
  else:
    timestamp = str (datetime.now()) . split (".") [0]
    timestamp = timestamp . replace (" ","_")
    fig_name = "output_"
    for time, unit in zip ( timestamp.split(":"), ["h","m","s"] ):
      fig_name += time + unit   # YYYY-MM-DD_HHhMMmSSs

  ## Legend and y-axis scale
  ax.legend (loc = "upper right", prop = {"family" : "serif", "size" : 25} )
  if log_scale: 
    ax.set_yscale ("log")
    fig_name += "_logscale"

  ax.set_xlim (bins_ref[0], bins_ref[-1])
  ax.set_ylim (y_min, y_max)

  plt.savefig (f"{img_dir}/{fig_name}.png")
  if (verbose > 0): print (f"Figure correctly exported to {img_dir}/{fig_name}.png") 
  plt.show()
  plt.close()

# +-----------------------+
# |    Plot histograms    |
# +-----------------------+

for i in range (len(dh.Y_vars)):
  plot_histogram ( x_ref = dh.Y[:,i]  ,
                   x_gen = Y_gen[:,i] ,
                   bins  = var_info[dh.Y_vars[i]][0] ,
                   w_ref = dh.w.flatten()  ,
                   w_gen = w_gen.flatten() ,
                   density = False ,
                   label_x = var_info[dh.Y_vars[i]][1] ,
                   label_y_ref = "Calibration (sWeighted)" ,
                   label_y_gen = "Generated (reweighted)" if rw_enabled else "Generated (sWeighted)" ,
                   sub_label = sample_info[args.sample]  ,
                   var_name  = var_info[dh.Y_vars[i]][2] ,
                   particle  = args.particle ,
                   sample    = args.sample   ,
                   log_scale = False ,
                   verbose = 1 )
  plot_histogram ( x_ref = dh.Y[:,i]  ,
                   x_gen = Y_gen[:,i] ,
                   bins  = var_info[dh.Y_vars[i]][0] ,
                   w_ref = dh.w.flatten()  ,
                   w_gen = w_gen.flatten() ,
                   density = False ,
                   label_x = var_info[dh.Y_vars[i]][1] ,
                   label_y_ref = "Calibration (sWeighted)" ,
                   label_y_gen = "Generated (reweighted)" if rw_enabled else "Generated (sWeighted)"  ,
                   sub_label = sample_info[args.sample]  ,
                   var_name  = var_info[dh.Y_vars[i]][2] ,
                   particle  = args.particle ,
                   sample    = args.sample  ,
                   log_scale = True ,
                   verbose = 1 )
