#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import pickle

DATA_DIR="../data"
with open(os.path.join(DATA_DIR, 'input_output_list.pickle'), 'rb') as f:
    a = pickle.load(f)

# ## Split data set into training and testing data
from sklearn.model_selection import train_test_split

X = np.array([d[0] for d in a])
y = np.array([d[1] for d in a])

print(X.shape)
print(y.shape)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

opt = "GP" # can be changed

if opt == "ridge":
    # Ridge regression, linear model, fastest
    reg = Ridge(random_state=23)
elif opt == "random_forest":
    # Random Forest
    reg = RandomForestRegressor(random_state=23)
elif opt == "MLP":
    # Multi-layer perceptron, feed-forward neural networks
    reg = MLPRegressor(random_state=1, max_iter=5000)
else:
    # Gaussian process regression, kriging
    opt = "GP"
    reg = GaussianProcessRegressor(kernel=DotProduct() + WhiteKernel(),
                                   random_state=23)
# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()), # Preprocessing step
    ('classifier', MultiOutputRegressor(reg))
])
# Train the model
pipeline.fit(X_train, y_train)
# Evaluate the model
score = pipeline.score(X_test, y_test)
print(f'Algorithm: {opt}, Model accuracy: {score}')
