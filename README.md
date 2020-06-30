# metagraph
Python library for running graph algorithms on a variety of hardware backends.
Data representing the graph will be automatically converted between available hardware options
to find an efficient solution.

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
# First install reportlab
# Next install rst2pdf
cd docs
sphinx-build -b pdf . _build/pdf
```