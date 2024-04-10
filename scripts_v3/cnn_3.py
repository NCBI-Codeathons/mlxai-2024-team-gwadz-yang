import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the embeddings DataFrame
DATA_DIR = '../data_v2'
FILE_DATAFRAME = os.path.join(DATA_DIR, 'Dataframe_CurName_features_embedding_biomedbert.pkl')
df = pd.read_pickle(FILE_DATAFRAME)

# Extract embeddings and labels
embeddings = df['features'].tolist()
labels = df['CurName'].tolist()

# Convert embeddings list to a single NumPy array
embeddings_array = np.array(embeddings)

# Normalize the embeddings
scaler = StandardScaler()
embeddings_array = scaler.fit_transform(embeddings_array)

# Convert protein family names to numerical labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(embeddings_array, encoded_labels, test_size=0.2, random_state=42)

# Convert NumPy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train)
y_test_tensor = torch.tensor(y_test)

# Define a neural network architecture with more hidden layers and dropout
class ProteinFamilyClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProteinFamilyClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x

# Define hyperparameters
input_dim = X_train.shape[1]
output_dim = len(np.unique(encoded_labels))

# Initialize the model with the updated architecture
model = ProteinFamilyClassifier(input_dim, output_dim)

# Adjust learning rate and optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Adjust learning rate

# Implement learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Define data loaders
batch_size = 64
train_data = TensorDataset(X_train_tensor, y_train_tensor)
test_data = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Define loss function
criterion = nn.CrossEntropyLoss()

# Train the model
num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
    scheduler.step()  # Step the learning rate scheduler

# Evaluate the model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {correct/total}")
