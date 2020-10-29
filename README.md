# Metagraph

[![Build Status](https://travis-ci.org/metagraph-dev/metagraph.svg?branch=master)](https://travis-ci.org/metagraph-dev/metagraph)
[![Coverage Status](https://coveralls.io/repos/metagraph-dev/metagraph/badge.svg?branch=master)](https://coveralls.io/r/metagraph-dev/metagraph)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/metagraph-dev/metagraph/blob/master/LICENSE)
[![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Documentation Status](https://readthedocs.org/projects/metagraph/badge/?version=latest)](https://metagraph.readthedocs.io/en/latest/?badge=latest)

Python library for running graph algorithms on a variety of hardware backends.
Data representing the graph will be automatically converted between available hardware options
to find an efficient solution.

Visit [ReadTheDocs page](https://metagraph.readthedocs.io/en/latest/) for more details.

## Development Environment

To create a new development environment:

```
conda env create
conda activate mg
pre-commit install  # for black
python setup.py develop
```

To run unit tests + coverage automatically
```
pytest
```


To build web documentation
```
cd docs
make html
```


To build PDF documentation
```
cd docs
make latexpdf
```
