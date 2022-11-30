# Machine Learning Algorithms from Square One

Different machine learning algorithms from implemented from square one.

The code is base on: https://github.com/AssemblyAI-Examples/Machine-Learning-From-Scratch

<br>

--------------------------------------------------------------------------------

<br>

## Algorithms

The following algorithms are implemented for various types of problems. The algorithms are implemented from scratch using only numpy and matplotlib.

*Regression*
- [x] Linear Regression


*Classification - binary*
- [x] Logistic Regression
- [x] Decision Tree
- [x] Random Forest
- [x] Naive Bayes
- [x] Perceptron
- [x] Support Vector Machine

*Classification - multi*
- [x] K Nearest Neighbors

*Clustering*
- [x] K Means Clustering

*Dimensionality Reduction*
- [x] Principal Component Analysis



<br>

--------------------------------------------------------------------------------

<br>

## Development Setup

The project uses [Poetry](https://python-poetry.org) to manage the project dependencies. Install dependencies within the respective subfolder via:

    `poetry install`

The entry point for testing the algorithms is the `main.py` file. To run the code, use:

    `poetry run python src/main.py`
