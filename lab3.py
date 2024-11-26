from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor 
import torch
import matplotlib.pyplot as plt

train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())
train_batches = DataLoader(dataset=train, batch_size=32)

print(f"Number of created batches: {len(train_batches)}")
for batch in train_batches:
    print(f"Type of 'batch': {type(batch)}")        # Check what is "batch"
    data, labels = batch        # Each batch returns the data (the images) and their corresponding labels
    print(f"Type of 'data': {type(data)}")  # Check what are "data" and "labels"
    print(f"Type of 'labels': {type(labels)}")
    print(f"Shape of data Tensor from current batch: {data.shape}")    # Check the shape of "data" and "labels"
    print(f"Shape of first image Tensor from current batch: {data[0].shape}")
    print(f"Shape of labels Tensor from current batch: {labels.shape}")
    print(f"Labels included in the 'labels' Tensor from current batch: {labels}")
    break

plt.figure()

for batch in train_batches:
    # Get the images and the labels for the current training batch
    data, labels = batch 

for i in range(6):              # Show the first 6 images in the plot
    plt.subplot(2, 3, i + 1)    # Create subplot (2 rows, 3 columns)
    plt.tight_layout()          # Adjust spacing between subplots
    plt.imshow(data[i][0], cmap="gray", interpolation="none")  # Display the image
    plt.title(f"Ground Truth label: {labels[i]}")  # Add title
    plt.xticks([])              # Remove x-axis ticks
    plt.yticks([])              # Remove y-axis ticks
plt.savefig("output_plot.png", dpi=300)     # Save the entire figure to a file
print("Plot saved as 'output_plot.png'")

#--------------------------------------------------------------
#now lets build and train a CNN model 

class DigitClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        #manually define the parameters: we basically only need to change the out_n numbers (2^n) 
        kernel_dim = 3
        out_1 = 16 #--> 26 (if kernel is 3x3)
        out_2 = 32 # --> 24 (if kernel is 3x3)
        out_3 = 64 #--> 22 (if kernel is 3x3)
        in_features= out_3 * 22 *22 #always this number (if kernel is 3x3)

        """
        kernel_dim = 5
        out_1 = 16 #--> 24 (if kernel is 5x5)
        out_2 = 32 # --> 20 (if kernel is 5x5)
        out_3 = 64 #--> 16 (if kernel is 5x5)
        in_features= out_3 * 16 * 16 #always this number (if kernel is 5x5)
        """

        self.model = torch.nn.Sequential(
            # Feature extraction stage
            torch.nn.Conv2d(
                in_channels=1,      # Grayscale input (1 channel) input image is 28x28
                out_channels=out_1,    # Number of filters
                kernel_size=(kernel_dim, kernel_dim), 
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_1, 
                out_channels=out_2, 
                kernel_size=(kernel_dim, kernel_dim), 
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out_2, 
                out_channels=out_3, 
                kernel_size=(kernel_dim, kernel_dim), 
            ),
            torch.nn.ReLU(),
            # Classification stage
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=in_features, out_features=10),
        )
    def forward(self, x):
        return self.model(x)

cnn = DigitClassifier()     #instance of the nn 
for batch in train_batches: 
    X, y = batch
    y_pred = cnn(X)
    #exit()                  #if this runs without error: great success :|D
    break

#-------------------------------------------------------
#TRAINING PROCEDURE
cnn = DigitClassifier()         # Instance of the neural network, loss, optimizer
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

num_epochs = 1      # Number of times to pass through every batch
cnn.train(True)     # Set the CNN to training mode
for epoch in range(num_epochs):
    for batch_idx, (X, y) in enumerate(train_batches):
        y_pred = cnn(X)                         # Pass the batch of images to obtain a prediction
        loss = loss_fn(input=y_pred, target=y)  # Compute the loss comparing the predictions made by the CNN with the original labels
        loss.backward()                         # Perform backpropagation with the computed loss to compute the gradients
        opt.step()                              # Update the weights with regard to the computed gradients to minimize the loss
        opt.zero_grad()                         # In each iteration we want to compute new gradients,that is why we set the gradients to 0
        if batch_idx % 50 == 0:
            print(f"Train Epoch: {epoch + 1} [{ batch_idx*len(data)}/{len(train_batches.dataset)} ({100.0 * batch_idx / len(train_batches):.0f}%)]\tLoss: {loss.item():.6f}")
            print(f"Epoch: {epoch + 1} loss is {loss.item()}")

#------------------------------------------------
#EVALUATE
"""
from torch import save, load    # Save the model state
with open("./cnn_model_state.pt", "wb") as file:
    save(cnn.state_dict(), file)
with open("./cnn_model_state.pt", "rb") as file:# Load the model state
    cnn.load_state_dict(load(file))
"""
test_data = datasets.MNIST(root="data", download=True, train=False, transform=ToTensor())   # Download the test data into the "data" folder
test_batches = DataLoader(dataset=test_data, batch_size=32)     # Group the training dataset into batches of size 32
cnn.eval()                                                      # Set the model to evaluation mode to predict data
with torch.no_grad():                                           # Avoid calculating gradients
    for images, labels in test_batches:                         # Predict the images in batches
        test_output = cnn(images)
        y_test_pred = torch.max(test_output, 1)[1].data.squeeze()           # For each image, take the highest probability
        accuracy = (y_test_pred == labels).sum().item() / float(labels.size(0))         # Measure the accuracy
print("VALIDATION SET ACCURACY: %.2f" % accuracy)

#_-----------------------------------------------
#PREDICTION - predict a single image in the dataset 
img_idx = 110 # <-- Change this value to select other image
img_tensor, label = test_data[img_idx]
print(f"Shape of tensor: {img_tensor.shape}")
img_tensor = img_tensor.unsqueeze(0) # <-- Add the right value
print(f"Shape of tensor: {img_tensor.shape }")

with torch.no_grad(): # Avoid calculating gradients
    img_pred = cnn(img_tensor)
    pred_label = torch.argmax(img_pred)
plt.figure()
plt.suptitle(f"Original label: {label}")
plt.title(f"Predicted label: {pred_label}")
plt.imshow(img_tensor[0, 0, :, :], cmap="gray")
plt.xticks([])
plt.yticks([])
plt.savefig(f"./prediction_of_test_image_{img_idx}.png")












