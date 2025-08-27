import numpy as np
import pickle
import os.path
import os
from IPython.display import HTML
from feather_io import FeatherWriter, FeatherReader
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from scikinC.decorators import inline_c

def store_as_pickle(obj, path_in_env: str, default_path: str):
    """
    Store an object as a pickle file creating the folder structure if needed. Path is taken from environment.
    """
    path = os.path.abspath(os.environ.get(path_in_env, default_path))
    sub_paths = [path]
    while sub_paths[-1] not in ['/']:
        sub_paths.append(os.path.dirname(sub_paths[-1]))
        
    for sub_path in sub_paths[::-1][:-1]:
        if not os.path.exists(sub_path):
            os.mkdir(sub_path)
            
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        
    return HTML(f"<B>Preprocessing step stored in: </B><PRE>{path}</PRE>")


def load_from_pickle(path_in_env: str, default_path: str):
    """
    Load an object from a pickle file. Path is taken from the environment.
    """
    path = os.path.abspath(os.environ.get(path_in_env, default_path))
    with open(path, "rb") as input_file:
        obj = pickle.load(input_file)
    
    return obj
    

def split_and_store(dataset, fracs_and_dirs: list, chunksize: int, **kwargs):
    """
    Split dataset in subsamples and store them in different dirs with feather_io.
    
    Additional keyword arguments are passed to FeatherWriter.
    
    Example.
    
    split_and_store(dataset, [
        (0.4, "./data/train-set"),
        (0.3, "./data/test-set"),
        (0.3, "./data/validation-set"),
        ])
    
    """
    fracs = [r[0] for r in fracs_and_dirs]
    dirs = [r[1] for r in fracs_and_dirs]
    
    assert abs(sum(fracs) - 1. < 1e-3), "Fractions do not add to 1."
    
    n_entries = []
    
    splits = dataset.random_split(fracs)
    for path, split in zip(dirs, splits):
        print ("Processing", path)
        n_entries.append(
            split
            .repartition(partition_size=chunksize)
            .map_partitions(FeatherWriter(output_dir=path, **kwargs))
            .compute(num_workers=8)
            .sum()
        )

    return n_entries


def peek_from_dataset(
    env_var: str,
    default_path: str = None,
    max_files: int = 3,
    entries: int = 2_000_000,
):
    ds_train =  FeatherReader(
        os.environ.get(env_var, default_path),
        max_files=max_files
    ).as_tf_dataset()
     
    return next(iter(ds_train.batch(entries)))


def read_root(filename, tree=None):
    if tree is None:
        return uproot.open(filename)

    for _ in range(5):
        try:
            with uproot.open(filename) as root_file:
                return pd.DataFrame(root_file[tree].arrays(library='np'))
        except TypeError:
            continue




class DecorrTransformer:
    """
    A minmalistic decorrelation transform based on the eigen vectors of covariance matrix.
    
    This simple transformers removes the linear correlation from the columns of a 
    dataset by computing the eigen vectors of their covariance matrix.
    A matrix is built by stacking the eigen vectors.
    Applying the matrix to the input features, a linear application rotating 
    and normalizing the inputs is obtained. 
    
    The matrix of the eigen vector is orthogonal by construction, hence the 
    inverse of the transform is simply obtained by multiplying the trasposed matrix
    to the transformed input.
    """
    def fit ( self, X, y = None ):
        """
        Computes the covariance matrix of the inputs and its eigen vectors. 
        """
        self.cov = np.cov (X.T)
        _, self.eig = np.linalg.eig ( self.cov )
        return self

    def transform (self, X):
        """
        Applies the direct transformation
        """
        dX = X.dot (self.eig) 
        return dX 

    def inverse_transform (self, dX): 
        """
        Applies the inverted tranformation
        """
        X = dX.dot (self.eig.T) 
        return X 

def c_impl(c_str):
    """Simple decorator to define C implementation of a function for scikinC"""
    def decorator(f):
        f.inC = c_str
        return f
    return decorator

@inline_c("0.3*log(1e-7 + ({x}/(1e-7 + (1 - {x}))))")
def ProbNNTransformer_fwd(X):
    return 0.3*np.log(1e-7 + X/(1e-7 + (1 - X)))

@inline_c("1 / (1 + exp(-{x/0.3}))")
def ProbNNTransformer_bwd(Y):
    return 1 / (1 + np.exp(-Y/0.3))

def makeProbNNTransformer():
    ret = FunctionTransformer(
        func=ProbNNTransformer_fwd,
        inverse_func=ProbNNTransformer_bwd,
        check_inverse=False,
    )

    return ret

    
if __name__ == '__main__':
    a = "A string"
    
    store_as_pickle(a, "OUTPUT_DIR", "test_folder/a_string.pkl")
    
    
    
