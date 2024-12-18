import torch
import torch.nn as nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

device = torch.device('cpu')
image_size = 128
batch_size = 32 
learning_rate = 0.0005  

#torch.manual_seed(42)

#--------------------------------------------------------------------------------------------------
# Pre-processing and loading the data 
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),  
    transforms.Grayscale(num_output_channels=1),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

data_path = './data/BreastUltrasound/'
dataset = datasets.ImageFolder(root=data_path, transform=transform)

# Split the dataset into training (80%) and testing (20%)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


#--------------------------------------------------------------------------------------------------
# Build the model
class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.flatten = torch.nn.Flatten()  
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size), 
            torch.nn.BatchNorm1d(hidden_size),  
            torch.nn.ReLU(), # or make is ELU 

            torch.nn.Linear(hidden_size, hidden_size // 2), 
            torch.nn.BatchNorm1d(hidden_size // 2),  
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),  

            torch.nn.Linear(hidden_size // 2, hidden_size // 4), 
            torch.nn.BatchNorm1d(hidden_size // 4),  
            torch.nn.ReLU(),

            torch.nn.Linear(hidden_size // 4, hidden_size // 8), 
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size // 8, num_classes) 
        )
    def forward(self, x):
        x = self.flatten(x)
        return self.model(x)

input_size = image_size * image_size * 1 
hidden_size = 1024 
num_classes = 3 
model = FeedForward(input_size, hidden_size, num_classes).to(device)

#--------------------------------------------------------------------------------------------------
"""
# Weight Initialization Function
def weights_init(m):
    if isinstance(m, torch.nn.Linear):
        nn.init.xavier_normal_(m.weight)  # Initialize weights with Xavier Normal
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)  # Set bias to 0
    elif isinstance(m, torch.nn.BatchNorm1d):
        nn.init.ones_(m.weight)  # BatchNorm weight should be initialized to 1
        nn.init.zeros_(m.bias)  # BatchNorm bias should be initialized to 0

model.apply(weights_init)  # Apply the weight initialization to the model

"""

#--------------------------------------------------------------------------------------------------
# Training 
class_labels = [sample[1] for sample in dataset]
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(class_labels), y=class_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5) 
loss_fn = nn.CrossEntropyLoss(weight=class_weights)
num_epochs = 10

model.train()
for epoch in range(num_epochs): 
    correct, total = 0, 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        
    scheduler.step()  # Update scheduler
    train_accuracy = 100.0 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, Loss: {loss.item():.6f}')

#--------------------------------------------------------------------------------------------------
# Evaluation 
model.eval()

correct, total = 0, 0 

with torch.no_grad():                                          
    for images, labels in test_loader:  
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = torch.argmax(outputs, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
accuracy = 100.0 * correct / total
print(f"VALIDATION SET ACCURACY: {accuracy:.2f}%")

#--------------------------------------------------------------------------------------------------
# Prediction / testing
img_path = './data/BreastUltrasound/malignant/malignant (1).png' 
img = Image.open(img_path)
img = transform(img).unsqueeze(0)
img = img.to(device)

with torch.no_grad():
    pred = model(img)
    pred_label = torch.argmax(pred)

plt.figure()
plt.title(f"Predicted Label: {'Malignant' if pred_label.item() == 1 else 'Benign' if pred_label.item() == 0 else 'Normal'}")
plt.imshow(img.cpu().squeeze(), cmap="gray")
plt.xticks([])
plt.yticks([])
plt.savefig(f"./prediction_of_image.png", dpi=300)

#71% accuracy 