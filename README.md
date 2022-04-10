# Irregular Bag-of-Pattern Feature

Source code for the Irregular Bag-of-Pattern Feature method proposed for my thesis "Representation of Astronomical Time Series using Information Retrieval Theory"

## Documentation in progress..

- reduce scripts to only essential code.
- refactor code to work as a package. (DONE)
- identify requirements and install instructions.
- reduce source code to only essential code.
- refactor source code and add documentation.
- Compact notebooks with only the figures used in the Thesis report.
- Add link to the final thesis report file.

## on working (hold)

- change the call to some settings on my code and AVOCADO code, the idea here is to use only setting file for declaring paths and file names, which will reduce the need of parameters on scripts
- need to check further refactoring on ibopf.method and ibopf.models

# Working (9 april 2022)

## settings

separating the logic from AVOCADO and IBOP, now we have:
- avocado_settings.json: will control the plasticc dataset download and
preprocessing (generating the balanced dataset also), and the AVOCADO method calls
- ibopf_settings.json: will control the IBOPF method calls

these changes have yet to be tested on the whole working pipeline

## requirements

so far we have that te requirements are:

- astropy
- george
- lightgbm
- matplotlib
- pytables (tables on pip)
- requests
- scikit-learn
- numpy
- scipy
- pandas
- tqdm
- numba
- statsmodel
- avocado

to generate a requirement.txt or env.yml we need to solve how to install avocado by default since its installation method is a little different from pip.

## new visualization: UMAP and t-SNE clustering

working on adding a cluster visualization of IBOPF and AVOCADO features, using UMAP and/or t-SNE
this visualization is oriented for the tesis document. We are adding a script that will
read the respective features and labels and apply the respective visualization method. This code is 
included in [notebooks/visualization_on_method_features.ipynb](notebooks/visualization_on_method_features.ipynb)
