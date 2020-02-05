# metagraph
Python library for running graph algorithms on a variety of hardware backends.
Data representing the graph will be automatically converted between available hardware options
to find an efficient solution.

## Development Environment

To create a new development environment:

```
conda create -n mg python=3.7 pytest pytest-cov coverage black conda-forge::pre-commit
conda activate mg
pre-commit install  # for black
python setup.py develop
```

To run unit tests + coverage automatically
```
pytest
```
