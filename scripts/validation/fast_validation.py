import yaml
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from lb_pidsim_train.utils import argparser
from lb_pidsim_train.trainers import DataHandler
from lb_pidsim_train.preprocessing import LbColTransformer

from tqdm import tqdm
from datetime import datetime
from html_reports import Report


NP_FLOAT = np.float32
TF_FLOAT = tf.float32


# +---------------------+
# |    Initial setup    |
# +---------------------+

print ( "\n\t\t\t\t\t+------------------------------------------+"   )
print (   "\t\t\t\t\t|                                          |"   )
print (   "\t\t\t\t\t|        GAN-based model validation        |"   )
print (   "\t\t\t\t\t|                                          |"   )
print (   "\t\t\t\t\t+------------------------------------------+\n" )

parser = argparser ("GAN-based model validation")
args = parser . parse_args()

slot = "-" . join ( args.sample . split("-") [:-1] )
calib_sample = ( "data" in args.sample )

if calib_sample : print ( "[INFO] GAN model trained on Calibration samples selected" )
else            : print ( "[INFO] GAN model trained on Monte Carlo samples selected" )

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

with open ("config/validation.yml") as file:
  validation = yaml.full_load (file)

# +---------------------------------+
# |    Data Handler construction    | 
# +---------------------------------+

model_name = f"{args.model}_{args.particle}_{args.sample}_{args.version}"

train_info = args.version.split("-")[1]

if train_info[0] == "w":
  sw_avail = True
  print ("[INFO] GAN model trained with sWeights selected")
else:
  sw_avail = False

if train_info[1] == "b":
  print ("[INFO] Model trained with the BceGAN algorithm selected")
elif train_info[1] == "c":
  print ("[INFO] Model trained with the CramerGAN algorithm selected")
elif train_info[1] == "w":
  print ("[INFO] Model trained with the WGAN-ALP algorithm selected")

if len(train_info) == 3:
  rw_enabled = True
  print ("[INFO] GAN model trained after a reweighting strategy selected")
else:
  rw_enabled = False

dh = DataHandler()

# +---------------------------+
# |    Data for validation    |
# +---------------------------+

if calib_sample:
  data_dir = config["data_dir"]["data"]
else:
  data_dir = config["data_dir"]["simu"]

file_list = datasets[args.model][args.particle][args.sample]
file_list = [ f"{data_dir}/{file_name}" for file_name in file_list ]

dh . feed_from_root_files ( root_files = file_list , 
                            X_vars = variables[args.model]["X_vars"][slot] , 
                            Y_vars = variables[args.model]["Y_vars"][slot] , 
                            w_var  = variables[args.model]["w_vars"][slot] if sw_avail else None , 
                            selections = selections[args.model][slot] , 
                            tree_names = None if calib_sample else "make_tuple" , 
                            chunk_size = validation[args.model][args.particle][args.sample]["chunk_size"] , 
                            verbose = 1 )

if args.model == "Muon":   # Compute MuonLL to replace MuonMuLL
  dh._datachunk["MuonLL"] = dh._datachunk["probe_Brunel_MuonMuLL"] - \
                                 dh._datachunk["probe_Brunel_MuonBgLL"]
  dh._Y_vars[-1] = "MuonLL"
  columns = dh.X_vars + dh.Y_vars + dh.w_var if dh.w_var else dh.X_vars + dh.Y_vars
  dh._datachunk = dh._datachunk[columns]

dh . prepare_dataset()

# +------------------------+
# |    Generator loading   |
# +------------------------+

opt_study = ("suid" in f"{args.version}") and ("trial" in f"{args.version}")

if opt_study:
  model_dir  = "{}/opt_studies/{}" . format ( config["model_dir"]  , args.model )
  report_dir = "{}/opt_studies/{}" . format ( config["report_dir"] , args.model )
else:
  model_dir  = config["model_dir"]
  report_dir = config["report_dir"]

generator = tf.keras.models.load_model ( f"{model_dir}/{model_name}/saved_generator" )
print (f"[INFO] Model correctly loaded from {model_dir}/{model_name}/saved_generator")

# +------------------------------------+
# |    Preprocessing transformation    |
# +------------------------------------+

file_X = f"{model_dir}/{model_name}/transform_X.pkl"
scaler_X = LbColTransformer ( pickle.load ( open (file_X, "rb") ) )

file_Y = f"{model_dir}/{model_name}/transform_Y.pkl"
scaler_Y = LbColTransformer ( pickle.load ( open (file_Y, "rb") ) )

# +--------------------------+
# |    Generate fake data    |
# +--------------------------+

batch_size = len(dh.X)
latent_dim = generator.input_shape[1] - dh.X.shape[1]

X_gen = scaler_X . transform ( dh.X )
X_gen = tf.convert_to_tensor ( X_gen, dtype = TF_FLOAT )
latent_tensor = tf.random.normal ( shape = (batch_size, latent_dim), dtype = TF_FLOAT )
input_tensor  = tf.concat ( [X_gen, latent_tensor], axis = 1 )

Y_gen = generator ( input_tensor ) . numpy() . astype (NP_FLOAT)
Y_ref = scaler_Y . transform ( dh.Y )
Y_gen = scaler_Y . inverse_transform ( Y_gen )

# +----------------------+
# |    Plot functions    |
# +----------------------+

def validation_plots (report) -> None:
  for i, y_var in tqdm ( enumerate (dh.Y_vars), total = len(dh.Y_vars), desc = "Plotting", unit = "var" ):
    report.add_markdown (f'<h2 align="center">{y_var}</h2>')
    _int_val_plots ( report, dh.Y[:,i], Y_gen[:,i], x_label = y_var, log_scale = False )
    _int_val_plots ( report, dh.Y[:,i], Y_gen[:,i], x_label = y_var, log_scale = True  )
    _int_cut_eff_plots ( report, dh.Y[:,i], Y_gen[:,i], corr_var = "p", cut_name = y_var )
    _int_cut_eff_plots ( report, dh.Y[:,i], Y_gen[:,i], corr_var = "eta", cut_name = y_var )
    _bin_1d_corr_plots ( report, dh.Y[:,i], Y_gen[:,i], corr_var = "p", x_label = y_var, log_scale = False )
    _bin_1d_corr_plots ( report, dh.Y[:,i], Y_gen[:,i], corr_var = "eta", x_label = y_var, log_scale = False )
    _bin_1d_corr_plots ( report, dh.Y[:,i], Y_gen[:,i], corr_var = "p", x_label = y_var, log_scale = True )
    _bin_1d_corr_plots ( report, dh.Y[:,i], Y_gen[:,i], corr_var = "eta", x_label = y_var, log_scale = True )
    report.add_markdown ("---")

# - - -

def _int_2d_corr_plots ( figure  ,
                         gs_list , 
                         x_ref , x_gen , y , 
                         bins = 10 , 
                         density = False , 
                         w_ref   = None  ,
                         w_gen   = None  ,
                         xlabel  = None  ,
                         ylabel  = None  ) -> None:
  """
  Internal.
  2-D plots reporting the correlations between PID variables
  and the kinematic ones in the whole space.
  """
  if len(gs_list) != 2: raise ValueError ("It should be passed only 2 GridSpec positions.")

  x_min = min ( x_ref.mean() - 3 * x_ref.std() , x_gen.mean() - 3 * x_gen.std() )
  x_max = max ( x_ref.mean() + 3 * x_ref.std() , x_gen.mean() + 3 * x_gen.std() )
  y_min = y.min() - 0.1 * ( y.max() - y.min() )
  y_max = y.max() + 0.1 * ( y.max() - y.min() )
  binning = [ np.linspace ( x_min, x_max, bins + 1 ) ,
              np.linspace ( y_min, y_max, bins + 1 ) ]

  ax0 = figure.add_subplot ( gs_list[0] )
  if xlabel: ax0 . set_xlabel ( xlabel, fontsize = 10 )
  if ylabel: ax0 . set_ylabel ( ylabel, fontsize = 10 )
  hist2d = np.histogram2d ( x_ref, y, weights = w_ref, density = density, bins = binning )
  ax0 . pcolormesh ( binning[0], binning[1], hist2d[0].T, cmap = plt.get_cmap ("viridis"), vmin = 0 )
  ax0 . annotate ( "original", color = "w", weight = "bold",
                   ha = "center", va = "center", size = 10,
                   xy = (0.8, 0.9), xycoords = "axes fraction", 
                   bbox = dict (boxstyle = "round", fc = "dodgerblue", alpha = 1.0, ec = "1.0") )

  ax1 = figure.add_subplot ( gs_list[1] )
  if xlabel: ax1 . set_xlabel ( xlabel, fontsize = 10 )
  if ylabel: ax1 . set_ylabel ( ylabel, fontsize = 10 )
  hist2d = np.histogram2d ( x_gen, y, weights = w_gen, density = density, bins = binning )
  ax1 . pcolormesh ( binning[0], binning[1], hist2d[0].T, cmap = plt.get_cmap ("viridis"), vmin = 0 )
  ax1 . annotate ( "generated", color = "w", weight = "bold",
                   ha = "center", va = "center", size = 10,
                   xy = (0.8, 0.9), xycoords = "axes fraction", 
                   bbox = dict (boxstyle = "round", fc = "deeppink", alpha = 1.0, ec = "1.0") )

# - - -

def _int_val_plots ( report,
                     x_ref  , 
                     x_gen  ,
                     x_label = None ,
                     log_scale = False ) -> None:
  """
  Internal.
  Validation plots of the distribution of the PID variables
  obtained intregating over the whole kinematic space.
  """
  fig = plt.figure ( figsize = (28, 6), dpi = 100 )
  gs = gridspec.GridSpec ( nrows = 2 , 
                           ncols = 5 ,
                           wspace = 0.25 ,
                           hspace = 0.25 ,
                           width_ratios  = [2, 2, 1, 1, 1] , 
                           height_ratios = [1, 1] )

  ax0 = fig.add_subplot ( gs[0:,0] )
  ax0 . set_xlabel (x_label, fontsize = 12)
  ax0 . set_ylabel ("Candidates", fontsize = 12)
  if dh.w_var is not None:
    ref_label = "Original (sWeighted)"
    gen_label = "Generated (reweighted)" if rw_enabled else "Generated (sWeighted)"
  else:
    ref_label = "Original (no sWeights)"
    gen_label = "Generated (no sWeights)"
  bin_min = min ( x_ref.mean() - 3 * x_ref.std() , x_gen.mean() - 3 * x_gen.std() )
  bin_max = max ( x_ref.mean() + 3 * x_ref.std() , x_gen.mean() + 3 * x_gen.std() )
  h_ref, bins, _ = ax0 . hist ( x_ref, bins = np.linspace (bin_min, bin_max, 75), 
                                weights = dh.w, color = "dodgerblue", label = ref_label )
  h_gen, _ , _ = ax0 . hist ( x_gen, bins = bins, weights = dh.w, histtype = "step", lw = 1.5, 
                              color = "deeppink", label = gen_label )
  ax0 . legend (loc = "upper left", fontsize = 10)
  y_max = max ( h_ref.max(), h_gen.max() )
  if log_scale:
    y_min = min ( h_ref[h_ref>0].min(), h_gen[h_gen>0].min() )
    y_max *= 20
    ax0 . set_yscale ("log")
  else:
    y_min = 0.0
    y_max += 0.2 * y_max
  ax0 . set_ylim (bottom = y_min, top = y_max)
  ax0 . set_xlim (left = bin_min, right = bin_max)

  ax1 = fig.add_subplot ( gs[0:,1] )
  ax1 . set_xlabel (x_label, fontsize = 12)
  ax1 . set_ylabel ("Candidates", fontsize = 12)
  ref_label = "Original (sWeighted)" if dh.w_var else "Original (no sWeights)"
  gen_label = "Generated"
  h_ref, bins, _ = ax1 . hist ( x_ref, bins = np.linspace (bin_min, bin_max, 75), 
                                weights = dh.w, color = "dodgerblue", label = ref_label )
  h_gen, _ , _ = ax1 . hist ( x_gen, bins = bins, histtype = "step", lw = 1.5, 
                              color = "deeppink", label = gen_label )
  ax1 . legend (loc = "upper left", fontsize = 10)
  y_max = max ( h_ref.max(), h_gen.max() )
  if log_scale:
    y_min = min ( h_ref[h_ref>0].min(), h_gen[h_gen>0].min() )
    y_max *= 20
    ax1 . set_yscale ("log")
  else:
    y_min = 0.0
    y_max += 0.2 * y_max
  ax1 . set_ylim (bottom = y_min, top = y_max)
  ax1 . set_xlim (left = bin_min, right = bin_max)

  _int_2d_corr_plots ( figure  = fig ,
                       gs_list = [ gs[0,2], gs[1,2] ] ,
                       x_ref = x_ref , 
                       x_gen = x_gen , 
                       y = dh.X[:,0]/1e3 ,
                       bins = 25 , 
                       density = False , 
                       w_ref = dh.w.flatten() ,
                       w_gen = dh.w.flatten() ,
                       xlabel = x_label ,
                       ylabel = "Momentum [Gev/$c$]" )

  _int_2d_corr_plots ( figure  = fig ,
                       gs_list = [ gs[0,3], gs[1,3] ] ,
                       x_ref = x_ref , 
                       x_gen = x_gen , 
                       y = dh.X[:,1] ,
                       bins = 25 , 
                       density = False , 
                       w_ref = dh.w.flatten() ,
                       w_gen = dh.w.flatten() ,
                       xlabel = x_label ,
                       ylabel = "Pseudorapidity" )

  _int_2d_corr_plots ( figure  = fig ,
                       gs_list = [ gs[0,4], gs[1,4] ] ,
                       x_ref = x_ref , 
                       x_gen = x_gen , 
                       y = dh.X[:,2] ,
                       bins = 25 , 
                       density = False , 
                       w_ref = dh.w.flatten() ,
                       w_gen = dh.w.flatten() ,
                       xlabel = x_label ,
                       ylabel = "$\mathtt{nTracks}$" )

  report.add_figure(options = "width=100%"); plt.clf(); plt.close()

# - - -

def _int_cut_eff_plots ( report ,
                         x_ref  , x_gen ,
                         corr_var = "p" ,
                         cut_name = None ) -> None:
  """
  Internal.
  Validation plots reporting the efficiencies obtained with various cuts
  of the PID variables for the reference and fake samples (whole kinematic space).
  """
  if corr_var not in ["p", "eta"]:
    raise ValueError ("The correlation variable should be chosen within ['p', 'eta']")

  fig, ax = plt.subplots ( nrows = 1, ncols = 3, figsize = (21,4), dpi = 100 )

  if corr_var == "p":
    var  = dh.X[:,0]/1e3
    bins = np.linspace (0, 100, num = 25)
    x_label = "Momentum [GeV/$c$]"
  else:
    var  = dh.X[:,1]
    bins = np.linspace (1.8, 5.5, num = 25)
    x_label = "Pseudorapidity"

  for i, (pctl, sel) in enumerate ( zip ( [25, 50, 75], ["Loose", "Mild", "Tight"] ) ):
    ax[i].set_title  (f"{cut_name} > $Q_{i+1}$", fontsize = 14)
    ax[i].set_xlabel (x_label, fontsize = 12)
    ax[i].set_ylabel (f"{sel} selection efficiency", fontsize = 12)

    cut_ref = np.percentile (x_ref, pctl, axis = None)
    query_ref = ( x_ref > cut_ref )
    query_gen = ( x_gen > cut_ref )

    h_all, bin_edges = np.histogram ( var            , bins = bins , weights = dh.w            . flatten() )
    h_ref, _         = np.histogram ( var[query_ref] , bins = bins , weights = dh.w[query_ref] . flatten() )
    h_gen, _         = np.histogram ( var[query_gen] , bins = bins , weights = dh.w[query_gen] . flatten() )

    h_all = np.where ( h_all > 0.0 , h_all , 1e-12 )
    h_ref = np.where ( h_ref > 0.0 , h_ref , 1e-12 )
    h_gen = np.where ( h_gen > 0.0 , h_gen , 1e-12 )

    bin_centers = ( bin_edges[1:] + bin_edges[:-1] ) / 2.0
    eff_ref = np.clip ( h_ref / h_all , 0.0 , 1.0 )
    eff_gen = np.clip ( h_gen / h_all , 0.0 , 1.0 )

    h_all_err = np.sqrt(h_all) / h_all
    h_ref_err = np.sqrt(h_ref) / h_ref
    h_gen_err = np.sqrt(h_gen) / h_gen

    eff_ref_err = eff_ref * np.sqrt ( h_all_err**2 + h_ref_err**2 )
    eff_gen_err = eff_gen * np.sqrt ( h_all_err**2 + h_gen_err**2 )

    ax[i].errorbar ( bin_centers, eff_ref, yerr = eff_ref_err, marker = "o", markersize = 5, capsize = 3, elinewidth = 2, 
                     mec = "dodgerblue", mfc = "w", color = "dodgerblue", label = "Data sample", zorder = 0 )
    ax[i].errorbar ( bin_centers, eff_gen, yerr = eff_gen_err, marker = "o", markersize = 5, capsize = 3, elinewidth = 1, 
                     mec = "deeppink", mfc = "w", color = "deeppink", label = "Trained model", zorder = 1 )

    ax[i].legend (fontsize = 10)
    ax[i].set_ylim (-0.1, 1.1)

  report.add_figure(options = "width=100%"); plt.clf(); plt.close()

# - - -

def _bin_1d_corr_plots ( report ,
                         x_ref  , 
                         x_gen  ,
                         corr_var = "p" ,
                         x_label = None ,
                         log_scale = False ) -> None:
  """
  Internal.
  1-D plots reporting the correlations between PID variables and
  the kinematic ones for different bins of the kinematic space.
  """
  if corr_var not in ["p", "eta"]:
    raise ValueError ("The correlation variable should be chosen within ['p', 'eta']")

  fig, ax = plt.subplots ( nrows = 2, ncols = 2, figsize = (14,8), dpi = 100 )
  plt.subplots_adjust ( wspace = 0.25, hspace = 0.25 )

  if dh.w_var is not None:
    ref_label = "Original (sWeighted)"
    gen_label = "Generated (reweighted)" if rw_enabled else "Generated (sWeighted)"
  else:
    ref_label = "Original (no sWeights)"
    gen_label = "Generated (no sWeights)"

  if corr_var == "p":
    cond = dh.X[:,0]
    bounds = [0.1e3, 5e3, 10e3, 25e3, 100e3]
  else:
    cond = dh.X[:,1]
    bounds = [1.8, 2.7, 3.5, 4.2, 5.5]

  idx = 0
  for i in range(2):
    for j in range(2):
      ax[i,j] . set_xlabel (x_label, fontsize = 12)
      ax[i,j] . set_ylabel ("Candidates", fontsize = 12)

      query = ( cond >= bounds[idx] ) & ( cond < bounds[idx+1] )
      bin_min = min ( x_ref[query].mean() - 3 * x_ref[query].std() , x_gen[query].mean() - 3 * x_gen[query].std() )
      bin_max = max ( x_ref[query].mean() + 3 * x_ref[query].std() , x_gen[query].mean() + 3 * x_gen[query].std() )
      h_ref, bins, _ = ax[i,j] . hist ( x_ref[query], bins = np.linspace (bin_min, bin_max, 75), 
                                        weights = dh.w[query], color = "dodgerblue", label = ref_label )
      h_gen, _ , _ = ax[i,j] . hist ( x_gen[query], bins = bins, weights = dh.w[query], 
                                      histtype = "step", lw = 1.5, color = "deeppink", label = gen_label )

      if corr_var == "p":
        text = f"$p \in ({bounds[idx]/1e3:.1f}, {bounds[idx+1]/1e3:.1f})$ [GeV/$c$]"
      else:
        text = f"$\eta \in ({bounds[idx]:.1f}, {bounds[idx+1]:.1f})$"

      ax[i,j] . annotate ( text, fontsize = 10, ha = "right", va = "top",
                           xy = (0.95, 0.95), xycoords = "axes fraction" )
      ax[i,j] . legend ( loc = "upper left", fontsize = 10 )

      y_max = max ( h_ref.max(), h_gen.max() )
      if log_scale:
        y_min = min ( h_ref[h_ref>0].min(), h_gen[h_gen>0].min() )
        y_max *= 20
        ax[i,j] . set_yscale ("log")
      else:
        y_min = 0.0
        y_max += 0.2 * y_max
      ax[i,j] . set_ylim (bottom = y_min, top = y_max)
      ax[i,j] . set_xlim (left = bin_min, right = bin_max)

      idx += 1

  report.add_figure(options = "width=45%"); plt.clf(); plt.close()

# +--------------------+
# |    Report setup    |
# +--------------------+

fname = f"val_{report_dir}/{model_name}"

report = Report()
date, hour = str (datetime.now()) . split (" ")
report.add_markdown (f"Report generated on **{date}** at {hour}")
report.add_markdown ("---")
validation_plots (report = report)
report.write_report (filename = f"{fname}.html")
print (f"[INFO] Report correctly exported to {fname}.html")
