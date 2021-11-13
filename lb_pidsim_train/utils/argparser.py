from datetime import datetime
from argparse import ArgumentParser 


MODELS = [ "Rich", "Muon", "GlobalPID", "GlobalMuonId" ]
PARTICLES = [ "Muon", "Pion", "Kaon", "Proton" ]
SAMPLES = [ "2016MagUp", "2016MagDown" ]


def argparser (description = None):   # TODO fix docstring
  """Return a parser with a set of default arguments.

  Parameters
  ----------
    description : `str`, optional
      Description for the ArgumentParser object 
      (`None`, by default).

  Returns
  -------
    parser : `argparse.ArgumentParser`
      Parser with default arguments.
  """
  timestamp = str (datetime.now()) . split (".") [0]
  timestamp = timestamp . replace (" ","_")
  version = ""
  for time, unit in zip ( timestamp.split(":"), ["h","m","s"] ):
    version += time + unit   # YYYY-MM-DD_HHhMMmSSs

  parser = ArgumentParser ( description = description )
  parser . add_argument ( "-m" , "--model"    , required = True , choices = MODELS )
  parser . add_argument ( "-p" , "--particle" , required = True , choices = PARTICLES ) 
  parser . add_argument ( "-s" , "--sample"   , required = True , choices = SAMPLES ) 
  parser . add_argument ( "-v" , "--version"  , default  = timestamp )

  return parser
