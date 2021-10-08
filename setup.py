from os import path
from io import open
from setuptools import setup, find_packages

this_directory = path.abspath ( path.dirname (__file__) )

## Get __version__ from version.py
__version__ = None
ver_file = path.join ("lb_pidsim_train", "version.py")
with open (ver_file) as file:
  exec ( file.read() )

## Load README
def readme():
  readme_path = path.join (this_directory, "README.md")
  with open (readme_path, encoding = 'utf-8') as file:
    return file.read()

## Load requirements
def requirements():
  requirements_path = path.join (this_directory, "requirements/base.txt")
  with open (requirements_path, encoding="utf-8") as file:
    return file.read() . splitlines()

setup (
        name = 'lb-pidsim-train',
        version = __version__,
        description  = 'Training pipeline for the parameterization of the LHCb PID system',
        long_description = readme(),
        long_description_content_type = 'text/markdown',
        url = 'https://github.com/mbarbetti/lb-pidsim-train',
        author = 'Matteo Barbetti',
        author_email = 'matteo.barbetti@fi.infn.it',
        maintainer = "Matteo Barbetti, Lucio Anderlini",
        maintainer_email = "matteo.barbetti@fi.infn.it, lucio.anderlini@fi.infn.it",
        license = 'GPLv3',
        keywords = [],
        packages = find_packages(),
        package_data = {'data': ['Zmumu.root']},
        include_package_data = True,
        install_requires = requirements(),
        python_requires  = '>=3.6, <4',
        classifiers = [
                        'Development Status :: 3 - Alpha',
                        'Intended Audience :: Science/Research',
                        'Intended Audience :: Developers',
                        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
                        'Programming Language :: Python :: 3',
                        'Programming Language :: Python :: 3.6',
                        'Programming Language :: Python :: 3.7',
                        'Programming Language :: Python :: 3.8',
                        'Programming Language :: Python :: 3.9',
                        'Programming Language :: Python :: 3 :: Only',
                        'Topic :: Scientific/Engineering',
                        'Topic :: Scientific/Engineering :: Physics',
                        'Topic :: Scientific/Engineering :: Artificial Intelligence',
                        'Topic :: Software Development',
                        'Topic :: Software Development :: Libraries',
                        'Topic :: Software Development :: Libraries :: Python Modules',
                      ],
  )