import io
import os
PATH = os.path.abspath (os.path.dirname (__file__))

from setuptools import setup, find_packages


with io.open (os.path.join (PATH, 'README.md'), encoding = 'utf-8') as f:
  long_description = f.read()

setup (
        name = 'lb-pidsim-train',
        version = '0.2.0',
        packages = find_packages(),
        package_data = {'data': ['Zmumu.root']},
        author = 'Matteo Barbetti',
        author_email = 'matteo.barbetti@fi.infn.it',
        description  = 'Training pipeline for the parameterization of the LHCb PID system',
        long_description = long_description,
        long_description_content_type = 'text/markdown',
        url = 'https://github.com/mbarbetti/lb-pidsim-train',
        license = 'GPLv3',
        python_requires  = '>=3.6, <4',
        install_requires = [],
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