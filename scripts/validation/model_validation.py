#from __future__ import annotations

import yaml
import pickle
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import matplotlib.pyplot as plt

from tqdm import tqdm
from itertools import product
from html_reports import Report
from matplotlib.patches import Patch
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

# +---------------------------------+
# |    Data Handler construction    | 
# +---------------------------------+

parser = argparser ("Model validation")
parser . add_argument ( "-a", "--algo", required = True )   # TODO add choices
parser . add_argument ( "-w", "--weights", default = "yes", choices = ["yes", "no"] )
args = parser . parse_args()

model_name = f"{args.algo}_{args.model}_{args.particle}_{args.sample}_{args.version}"

data_handler = DataHandler()

# +---------------------------+
# |    Data for validation    |
# +---------------------------+

sw = args.weights == "yes"

data_dir  = config["data_dir"]
file_list = datasets[args.model][args.particle][args.sample]
file_list = [ f"{data_dir}/{file_name}" for file_name in file_list ]

chunk_size = validation[args.model][args.particle][args.sample]["chunk_size"]

data_handler . feed_from_root_files ( root_files = file_list , 
                                      X_vars = variables[args.model]["X_vars"][args.sample] , 
                                      Y_vars = variables[args.model]["Y_vars"][args.sample] , 
                                      w_var  = variables[args.model]["w_vars"][args.sample] if sw else None, 
                                      selections = selections[args.model][args.sample] , 
                                      tree_names = None , 
                                      chunk_size = chunk_size , 
                                      verbose = 1 )

data_handler . prepare_dataset()

# +------------------------+
# |    Generator loading   |
# +------------------------+

model_dir = config["model_dir"]
generator = tf.keras.models.load_model ( f"{model_dir}/{model_name}/saved_model" )

# +------------------------------------+
# |    Preprocessing transformation    |
# +------------------------------------+

file_X = f"{model_dir}/{model_name}/transform_X.pkl"
scaler_X = PidsimColTransformer ( pickle.load ( open (file_X, "rb") ) )

file_Y = f"{model_dir}/{model_name}/transform_Y.pkl"
scaler_Y = PidsimColTransformer ( pickle.load ( open (file_Y, "rb") ) )

# +---------------------+
# |    Generate data    |
# +---------------------+

batch_size = len(data_handler.X)
latent_dim = generator.input_shape[1] - data_handler.X.shape[1]

X_gen = scaler_X . transform ( data_handler.X )
X_gen = tf.convert_to_tensor ( X_gen, dtype = TF_FLOAT )
latent_tensor = tf.random.normal ( shape = (batch_size, latent_dim), dtype = TF_FLOAT )
input_tensor  = tf.concat ( [X_gen, latent_tensor], axis = 1 )

Y_gen = generator ( input_tensor ) . numpy() . astype (NP_FLOAT)
Y_gen = scaler_Y . inverse_transform ( Y_gen )

# +------------------------+
# |    Prepare datasets    |
# +------------------------+

num_jobs = validation[args.model][args.particle][args.sample]["num_jobs"]

row_to_drop = len(Y_gen) % num_jobs

X = np.vsplit ( data_handler.X[:-row_to_drop,:] , num_jobs )
Y_gen = np.vsplit ( Y_gen[:-row_to_drop,:] , num_jobs )
Y_ref = np.vsplit ( data_handler.Y[:-row_to_drop,:] , num_jobs )
w = np.vsplit ( data_handler.w[:-row_to_drop] , num_jobs )

datasets = list()
for i in range (num_jobs):
  datasets . append ( [ X[i], Y_gen[i], Y_ref[i], w[i] ] )

# +-----------------------+
# |    Histograms info    |
# +-----------------------+

binning_vars = [ "probe_Brunel_P", "probe_Brunel_ETA", "nTracks_Brunel" ]   # equal to X_vars[:3]

boundaries = {
    "probe_Brunel_P"                   : [ 3e3, 4e3, 5e3, 6e3, 9e3, 15e3, 25e3, 50e3, 100e3, 200e3 ],
    "probe_Brunel_ETA"                 : [ 1.5, 2.5, 3.0, 3.5, 4.0, 5.5 ],
    "nTracks_Brunel"                   : [ 0, 50, 100, 200, 1000 ],
    "probe_Brunel_trackcharge"         : [ -2, 0, 2 ],
    "probe_Brunel_RichDLLe"            : np.linspace ( -150, 150, 100 ) ,
    "probe_Brunel_RichDLLmu"           : np.linspace ( -150, 150, 100 ),
    "probe_Brunel_RichDLLk"            : np.linspace ( -150, 150, 100 ),
    "probe_Brunel_RichDLLp"            : np.linspace ( -150, 150, 100 ),
    "probe_Brunel_isMuon"              : [ -0.5, 0.5, 1.5 ],
    "probe_Brunel_MuonMuLL"            : np.linspace ( -9, 1, 100 ),
    "probe_Brunel_MuonBgLL"            : np.linspace ( -9, 1, 100 ),
    "probe_Brunel_PIDe"                : np.linspace ( -25, 15, 100 ),
    "probe_Brunel_PIDK"                : np.linspace ( -150, 150, 100 ),
    "probe_Brunel_PIDp"                : np.linspace ( -150, 150, 100 ),
    "probe_Brunel_MC15TuneV1_ProbNNe"  : np.linspace ( 0, 1, 100 ),
    "probe_Brunel_MC15TuneV1_ProbNNpi" : np.linspace ( 0, 1, 100 ),
    "probe_Brunel_MC15TuneV1_ProbNNk"  : np.linspace ( 0, 1, 100 ),
    "probe_Brunel_MC15TuneV1_ProbNNp"  : np.linspace ( 0, 1, 100 ),
    "probe_Brunel_PIDmu"               : np.linspace ( -20, 20, 100 ),
    "probe_Brunel_MC15TuneV1_ProbNNmu" : np.linspace ( 0, 1, 100 ),
  }

# +-----------------------+
# |    Fill histograms    |
# +-----------------------+

def fill_histos (data):
  X, Y_gen, Y_ref, w = data
  histos_gen = list()
  histos_ref = list()

  for idx, y_var in enumerate (data_handler.Y_vars):
    histos_gen . append (
      np.histogramdd ( np.c_ [ X[:,:3], Y_gen[:,idx] ] ,
                       bins = [ boundaries[var] for var in binning_vars + [y_var] ] ,
                       weights = w . flatten(),
                     ) [0] )
    histos_ref . append (
      np.histogramdd ( np.c_ [ X[:,:3], Y_ref[:,idx] ] ,
                       bins = [ boundaries[var] for var in binning_vars + [y_var] ] ,
                       weights = w . flatten(),
                     ) [0] )

  return histos_gen, histos_ref

# +-------------------------------+
# |    Multiprocessing filling    |
# +-------------------------------+

scheduler = mp.Pool (processes = num_jobs)

hgen = None
href = None
for histos_gen, histos_ref in scheduler.imap_unordered (fill_histos, datasets):
  if hgen is None: hgen = histos_gen
  else:            hgen = [ h1 + h2 for h1, h2 in zip (hgen, histos_gen) ]
  if href is None: href = histos_ref
  else:            href = [ h1 + h2 for h1, h2 in zip (href, histos_ref) ]

# +-----------------------------+
# |    Validation plots info    |
# +-----------------------------+

var_titles = {
    "probe_Brunel_P"                   : f"p ({args.particle}) / MeV" ,
    "probe_Brunel_ETA"                 : r"$\eta$ ({args.particle})"  ,
    "nTracks_Brunel"                   : "nTracks" ,
    "probe_Brunel_trackcharge"         : "Charge"  ,
    "probe_Brunel_RichDLLe"            : "Rich-only DLLe"  ,
    "probe_Brunel_RichDLLmu"           : "Rich-only DLLmu" ,
    "probe_Brunel_RichDLLk"            : "Rich-only DLLk"  ,
    "probe_Brunel_RichDLLp"            : "Rich-only DLLp"  ,
    "probe_Brunel_isMuon"              : "IsMuon" ,
    "probe_Brunel_MuonMuLL"            : "Muon-only MuLL"  ,
    "probe_Brunel_MuonBgLL"            : "Muon-only BkgLL" ,
    "probe_Brunel_PIDe"                : "Combined DLLe" ,
    "probe_Brunel_PIDK"                : "Combined DLLK" ,
    "probe_Brunel_PIDp"                : "Combined DLLp" ,
    "probe_Brunel_MC15TuneV1_ProbNNe"  : "ProbNNe (Tune MC15V1)"  ,
    "probe_Brunel_MC15TuneV1_ProbNNpi" : "ProbNNpi (Tune MC15V1)" ,
    "probe_Brunel_MC15TuneV1_ProbNNk"  : "ProbNNk (Tune MC15V1)"  ,
    "probe_Brunel_MC15TuneV1_ProbNNp"  : "ProbNNp (Tune MC15V1)"  ,
    "probe_Brunel_PIDmu"               : "Combined DLLmu" ,
    "probe_Brunel_MC15TuneV1_ProbNNmu" : "ProbNNmu (Tune MC15V1)" ,
  }

format_bin = {
    "probe_Brunel_P"   : lambda m, M : rf"{m/1e3:.1f} < $p$ < {M/1e3:.1f} GeV/$c$" ,
    "probe_Brunel_ETA" : lambda m, M : rf"{m:.2f} < $\eta$ < {M:.2f}"  ,
    "nTracks_Brunel"   : lambda m, M :  f"{m:.0f} < nTracks < {M:.0f}" ,
  }

X_bins = [ boundaries[b] for b in binning_vars ]
Y_bins = [ boundaries[b] for b in data_handler.Y_vars ]

# +-----------------------+
# |    Plot histograms    |
# +-----------------------+

report = Report()

def plot_histos (bin_id):
  cut_val, cut_set = list(), list()
  for x_var, i_bin, bndr in zip (data_handler.X_vars, bin_id, X_bins):
    cut_val . append ( np.array ( [bndr[i_bin], bndr[i_bin+1]] ) )          # for report
    cut_set . append ( format_bin[x_var] ( bndr[i_bin], bndr[i_bin+1] ) )   # for plot

  # Print the studied intervals for p, eta and nTracks on the report
  report.add_markdown ( f"**Range of the momentum:** {tuple (cut_val[0]/1e3)} GeV/c" )
  report.add_markdown ( f"**Range of the pseudorapidity:** {tuple (cut_val[1])}"     )
  report.add_markdown ( f"**Range of the event multiplicity:** {tuple (cut_val[2])}" )

  # Draw the validation histograms for the studied intervals
  for y_id, (y_var, y_bin) in enumerate (zip (data_handler.Y_vars, Y_bins)):
    entries_gen = np.array (hgen) [y_id] [bin_id]
    entries_ref = np.array (href) [y_id] [bin_id]

    max_entries = max ( entries_ref.max(), entries_gen.max() )
    max_entries += 0.25 * max_entries

    if np.sum (entries_ref) >= 50:   # avoid plots with low statistic
      plt.figure (figsize = (8,5), dpi = 100)
      plt.xlabel (var_titles[y_var], fontsize = 12)
      plt.ylabel ("Candidates", fontsize = 12)

      left_edge, right_edge = y_bin[:-1], y_bin[1:]
      values = np.array ( [left_edge, right_edge] ) . T . flatten()
      entries_gen = np.array ( [entries_gen, entries_gen] ) . T . flatten()   # for horizontal line
      entries_ref = np.array ( [entries_ref, entries_ref] ) . T . flatten()   # for horizontal line
      plt.axis ([values.min(), values.max(), 0, int (max_entries)])

      plt.plot (values, entries_ref, color = "dodgerblue", linewidth = 1.5)
      plt.fill_between (values, 0, entries_ref, color = "lightskyblue")
      plt.plot (values, entries_gen, color = "deeppink", linewidth = 1.5)

      custom_legend = [
        Patch (fc = "lightskyblue", ec = "dodgerblue", lw = 1.5, label = "Training data"),
        Patch (fc = "white", ec = "deeppink", lw = 1.5, label = "Generated data")
      ]
      plt.legend (handles = custom_legend, loc = "upper left", fontsize = 10)
      plt.annotate (f"{cut_set[0]}\n{cut_set[1]}\n{cut_set[2]}", 
                    ha = "center", va = "center", size = 10, 
                    xy = (0.84, 0.90), xycoords = "axes fraction",
                    bbox = dict (boxstyle = "round", fc = "w", alpha = 0.8, ec = ".8"))
      if not sw:
        plt.annotate ("no sWeights", color = "w", weight = "bold",
                      ha = "center", va = "center", size = 10,
                      xy = (0.84, 0.78), xycoords = "axes fraction", 
                      bbox = dict (boxstyle = "round", fc = "r", alpha = 1.0, ec = ".8"))

      report.add_figure(); plt.close()

# +-------------------------+
# |    Produce all plots    |
# +-------------------------+

combinations = product (*[np.arange (len (b) - 1) for b in X_bins])
total_combs  = np.prod ( [len (b) - 1 for b in X_bins] )

for bin_id in tqdm (combinations, total = total_combs, desc = "Formatting", unit = "plot"):
  plot_histos (bin_id)
  report.add_markdown ("***")

report_dir = config["report_dir"]
filename = f"{report_dir}/val_{model_name}.html"

report.write_report ( filename = filename )
print ( f"Report correctly exported to {filename}" )
