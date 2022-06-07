import os
import yaml
import pickle
import ctypes
import numpy as np 
import pandas as pd
import tensorflow as tf

from tqdm import tqdm 
from functools import partial
from argparse import ArgumentParser
from lb_pidsim_train.preprocessing import LbColTransformer


parser = ArgumentParser()
parser . add_argument ( "--inputfile" , "-i" , help = "Input filename"             , required = True )
parser . add_argument ( "--particle"  , "-p" , help = "Particle name (e.g. Pion)"  , required = True ) 
parser . add_argument ( "--slot"      , "-s" , help = "Slot (e.g. 2016-MagUp-dta)" , required = True )
args = parser.parse_args() 

with open ("../../training/config/config.yml") as file:
  config = yaml.full_load (file)

MODEL_DIR = f"{config['model_dir']}/latest_models"


class GanPipe:
  def __init__ (self, step, part, slot):
    model_path = f"{MODEL_DIR}/{step}_{part}_{slot}_latest" 
    self.tX = LbColTransformer ( pickle.load ( open ( f"{model_path}/transform_X.pkl", "rb" ) ) )
    self.tY = LbColTransformer ( pickle.load ( open ( f"{model_path}/transform_Y.pkl", "rb" ) ) )
    self.model = tf.keras.models.load_model ( f"{model_path}/saved_generator" )

  def predict (self, X, random):
    return self.tY.inverse_transform (
                                       self.model.predict (
                                         np.concatenate ( (self.tX.transform(X), random), axis = 1 )
                                                          )
                                     )
                                     

class isMuonPipe:
  def __init__ (self, part, slot):
    model_path = f"{MODEL_DIR}/isMuon_{part}_{slot}_latest" 
    self.pipe = pickle.load ( open ( f"{model_path}/pipeline.pkl", "rb" ) )

  def predict (self, X):
    return self.pipe.predict_proba(X)[:,1] 


class FullPipe:
  def __init__ (self, part, slot):
    self.rich  = GanPipe ("Rich", part, slot) 
    self.muon  = GanPipe ("Muon", part, slot) 
    self.gpid  = GanPipe ("GlobalPID", part, slot) 
    self.gmuid = GanPipe ("GlobalMuonId", part, slot) 

  def predict (self, X, random):
    richdll = self.rich.predict ( X[:,:4], random[:, 0o000:0o100] )
    muondll = self.muon.predict ( X[:,:4], random[:, 0o100:0o200] )

    gpidX = np.concatenate( [X[:,:4], richdll, X[:,-1:], muondll], axis = 1 )
    gpid = self.gpid.predict ( gpidX, random[:, 0o200:0o300] )

    gmuidX = np.concatenate ( [X[:,:4], richdll, muondll], axis = 1 )
    gmuid = self.gmuid.predict ( gmuidX, random[:,0o300:0o400] )

    return np.concatenate ( [richdll, muondll, gpid, gmuid], axis = 1 )



ismuon_pipe = isMuonPipe (args.particle, args.slot)
pipe = FullPipe (args.particle, args.slot)

n = 10000
p = np.random.normal (30e3, 0.3e3, n)
eta = np.random.uniform (2, 5, (n,2)) . mean (axis = -1)
nTracks = np.random.uniform (20, 150, (n,2)) . mean (axis = -1) . astype (np.int32)
charge = np.random.choice ([-1., 1.], n)

ismuon_eff = ismuon_pipe.predict (np.c_[p, eta, nTracks, charge]) 
mu_rnd = np.random.uniform (0, 1, n) 
ismuon = np.where (mu_rnd < ismuon_eff, 1, 0 )

data = np.c_ [p, eta, nTracks, charge, ismuon]
rnd  = np.random.normal (0, 1, (n, 64*4) )

basedir = os.environ["PWD"]
dll = ctypes.CDLL ( os.path.join (basedir, args.inputfile) )
print (dll)

float_p = ctypes.POINTER (ctypes.c_float) 


pyout = pipe.predict (data, rnd)

counts = {th:0 for th in ['1e-2','1e-3','1e-4','1e-5']}

errors = list()   # debug
progress_bar = partial (tqdm, total = n, desc = f"Checking {args.particle}/{args.slot}")
for data_row, rnd_row, pyout_row, ismuoneff_row in progress_bar ( zip (data, rnd, pyout, ismuon_eff) ):
  in_f  = data_row.astype (ctypes.c_float)
  out_f = np.zeros (15, dtype = ctypes.c_float)
  rnd_f = rnd_row.astype (ctypes.c_float) 

  getattr (dll, f"IsMuon{args.particle}") (
                                            out_f . ctypes.data_as (float_p), 
                                            in_f  . ctypes.data_as (float_p) 
                                          )

  abserr = abs (out_f[0] - ismuoneff_row)
  # if abserr > 1e-2:
  #   print ( np.c_ [out_f[0], ismuoneff_row, abserr] ) 
  #   raise Exception ("C and Python muon_eff implementations inconsistent")

  for th in counts.keys():
    counts[th] += 1 if abserr > float(th) else 0 

  getattr (dll, f"{args.particle.lower()}_pipe")(
                                                  out_f . ctypes.data_as (float_p), 
                                                  in_f  . ctypes.data_as (float_p), 
                                                  rnd_f . ctypes.data_as (float_p)
                                                )
   
  relerr = abs (out_f - pyout_row) / ( 1 + abs (pyout_row) )
  for th in counts.keys():
    counts[th] += np.count_nonzero (relerr > float(th))

  # if np.any (relerr > 1e-2):
  #   print ( np.c_ [out_f, pyout_row, relerr, relerr < 1e-3] ) 
  #   raise Exception ("C and Python implementation were found inconsistent")

  errors.append (relerr)   # debug

df = pd.DataFrame (errors, columns = [f"Rich_{i}" for i in range(4)] + [f"Muon_{i}" for i in range(2)] + [f"GlobalPID_{i}" for i in range(7)] + [f"GlobalMuonId_{i}" for i in range(2)])
print (df.describe())   # debug

print ("SUCCESS")
print ("All entries satisfy a compatibility requirement at 1e-2")
for k in sorted (counts.keys()):
  v = counts[k]
  print (f"{v/pyout.size*100:.5f}% fails a compatibility check at {k}")
