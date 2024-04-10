import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os

# Load the embeddings DataFrame
DATA_DIR = '../data_v2'
FILE_DATAFRAME = os.path.join(DATA_DIR, 'Dataframe_CurName_features_embedding_biomedbert.pkl')
df = pd.read_pickle(FILE_DATAFRAME)

# Extract embeddings and labels
embeddings = df['features'].tolist()
labels = df['CurName'].tolist()

# Define a simple feedforward neural network
class ProteinFamilyClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(ProteinFamilyClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Convert embeddings list to PyTorch tensor
embeddings_tensor = torch.tensor(embeddings)

# Convert protein family names to PyTorch tensor (assuming you have labels)
# You need to replace 'labels' with your actual labels tensor
labels_tensor = torch.tensor(labels)

# Define hyperparameters
input_dim = len(embeddings[0])  # Dimension of the embedding vectors
hidden_dim = 128  # Number of neurons in the hidden layer
output_dim = len(set(labels))  # Number of unique protein family names

# Initialize the model, loss function, and optimizer
model = ProteinFamilyClassifier(input_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split data into training and testing sets (80% train, 20% test)
split_ratio = 0.8
split_index = int(split_ratio * len(embeddings_tensor))
train_data = TensorDataset(embeddings_tensor[:split_index], labels_tensor[:split_index])
test_data = TensorDataset(embeddings_tensor[split_index:], labels_tensor[split_index:])

# Define data loaders
batch_size = 32
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

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
