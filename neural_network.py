import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
dataset = pd.read_csv("Assest/dataset/ShirtSizeRecommendation.csv")

# Check for NaN values
columns_with_nan = dataset.columns[dataset.isna().any()].tolist()
if columns_with_nan:
    print("Columns with NaN values:", columns_with_nan)
    # Replace NaN values with the mean or median of the respective columns
    for col in columns_with_nan:
        if dataset[col].dtype == 'object':
            dataset[col].fillna(dataset[col].mode()[0], inplace=True)
        else:
            dataset[col].fillna(dataset[col].mean(), inplace=True)

# Convert columns to numeric types
numeric_cols = ['age', 'height']
dataset[numeric_cols] = dataset[numeric_cols].apply(pd.to_numeric)

# Check for NaN values after preprocessing
if dataset.isnull().sum().sum() == 0:
    print("No NaN values remaining in the dataset.")
else:
    print("NaN values still present in the dataset.")

# Step 1: Data Preprocessing
X = dataset[['weight', 'age', 'height']].values
y = dataset['size'].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Ensure correct label encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
num_classes = len(label_encoder.classes_)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.int64)

# Define the neural network model
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

# Initialize the model
input_size = 3
hidden_size = 64
model = Model(input_size, hidden_size, num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 50
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)

    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch+1) % 5 == 0:
        print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, predicted = torch.max(outputs.data, 1)
    accuracy = accuracy_score(y_test_encoded, predicted.numpy())
    print("Test Accuracy:", accuracy)
