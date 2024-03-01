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
from sklearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Define the pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()), # Preprocessing step
    ('classifier', MultiOutputRegressor(Ridge(random_state=23)))
])
# Train the model
pipeline.fit(X_train, y_train)
# Evaluate the model
score = pipeline.score(X_test, y_test)
print(f'Model accuracy: {score}')
