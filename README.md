# Comparing Methods of Hurricane Forecast Uncertainty with Neural Networks
***
* author: Elizabeth A. Barnes and Randal J. Barnes
* version: v1.0
* date: 12 November 2021

Neural networks are used to estimate consensus hurricane track and intensity errors, as well as the associated uncertainties of the network predictions.


## References
***
[1] Elizabeth A. Barnes and Randal J. Barnes and Nicolas Gordillo, 2021, Adding Uncertainty to Neural Network Regression Tasks in the Geosciences, arXiv 2109.07250.


## Python Environment
***
The following python environment was used to implement this code.
```
conda create --name hurricanes python=3.9
conda activate hurricanes
pip install tensorflow tensorflow-probability
pip install --upgrade numpy scipy pandas statsmodels matplotlib seaborn
pip install --upgrade palettable progressbar2 tabulate icecream flake8
pip install --upgrade keras-tuner sklearn
pip install --upgrade jupyterlab black isort jupyterlab_code_formatter
pip install silence-tensorflow
```
