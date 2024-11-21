#hands on lab created by following the guiding document 

import numpy as np 
import matplotlib.pyplot as plt

# CREATING A NEURON

class Neuron: 
    def __init__(self, n_inputs, activation_function, activation_derivative):
        # Initializes the neuron w/ random weights and bias, and sets the activation function.
        # Parameters:
        # n_inputs: the num of input connections to the neuron 
        # activation_function: to be used by the neuron 

        self.weights = np.random.uniform(-1, 1, n_inputs)
        self.bias = np.random.uniform(-1, 1)
        self.output=0
        self.inputs=0
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

    def forward(self, inputs): 
        #performs the forward pass of the nn layer
        #inputs: input data to the layer 
        self.inputs = inputs 
        total = np.dot (self.weights, inputs) + self.bias 
        self.output = self.activation_function(total)
        return self.output 

    def backpropagate(self, d_error, learning_rate): 
        #d_error is the gradient fo the error w.r.t. the output of the neuron 
        d_2 = d_error * self.activation_derivative(self.output)
        d_3 = d_2
        d_4 = d_2 * self.inputs
        self.weights -= learning_rate * d_4
        self.bias -= learning_rate * d_3
        return d_2 * self.weights


def sigmoid(x):
    return 1/ (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x* (1-x)

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

#x = np.linspace(-10, 10, 100) 
#y = sigmoid(x)
#plt.plot(x, y)
#plt.xlabel("X")
#plt.ylabel("sigmoid(x)")
#plt.title("sigmoid function")
#plt.grid()
#plt.show() #now we have a working neuron

neuron = Neuron(n_inputs=2, activation_function=sigmoid, activation_derivative=sigmoid_derivative)
input = [1, 2]
output = neuron.forward(input)
print(f"Input is {input}, output is {output}")

#------------------------------------------------------------------------------------------
#CREATING A NN

#create the hidden layers 
n_inputs = 2
hidden_layer_1 = [Neuron(n_inputs=n_inputs, activation_function=sigmoid, activation_derivative=sigmoid_derivative) for _ in range(4)]
hidden_layer_2 = [Neuron(n_inputs=4, activation_function=sigmoid, activation_derivative=sigmoid_derivative) for _ in range(4)]
hidden_layers=[hidden_layer_1, hidden_layer_2]
output_layer= [Neuron(n_inputs=4, activation_function=sigmoid, activation_derivative=sigmoid_derivative)]

input_data = np.array([1, 2])

#here, need to fix the printing, need a new version of python - when i have good network connection 
for i, layer in enumerate(hidden_layers):
    print(f"-> Input data for the layer {i} is: {input_data}")
    input_data = np.array([neuron.forward(input_data) for neuron in layer])
    print(f"<- Output data for the layer {i} is: {input_data}")

print(f"-> Input data for the output layer is: {input_data}")
output_data = np.array([neuron.forward(input_data) for neuron in output_layer])
print(f"<- Output data for the output layer is: {output_data}")

class NeuralNetwork: 
    def __init__(self, hidden_layers, output_layer):
        #initialize a neural network w/ given architeture
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
    
    def forward(self, inputs):
        #perform the forward pass of the nn
        for layer in self.hidden_layers: 
            inputs = np.array([neuron.forward(inputs) for neuron in layer])
        final_outputs = np.array([neuron.forward(inputs) for neuron in self.output_layer])
        return final_outputs

    def backpropagate(self, inputs, targets, learning_rate): 
        final_outputs = self.forward(inputs) #compute the output of the neural network
        d_error_d_output = -(targets - final_outputs) #calculate the error as the gradient
        #backpropagate through the output layer 
        hidden_errors = [neuron.backpropagate(d_error_d_output[i], learning_rate) for i, neuron in enumerate(self.output_layer)]
        #add the errors of the output layer 
        hidden_errors = np.sum(hidden_errors, axis = 0)
        #backpropagate through the hidden layer 
        for layer in reversed(self.hidden_layers):
            hidden_errors = np.sum([neuron.backpropagate(hidden_errors[i], learning_rate) for i, neuron in enumerate(layer)], axis = 0)

num_inputs = 2
hidden_layers = [[Neuron(2, sigmoid, sigmoid_derivative), Neuron(2, sigmoid, sigmoid_derivative),Neuron(2, sigmoid, sigmoid_derivative), Neuron(2, sigmoid, sigmoid_derivative)]]
output_layer = [Neuron(4, sigmoid, sigmoid_derivative)]


#here we present 2 kinds of output printing:
#1)
inputs = np.array([1, 2])
for i, layer in enumerate(hidden_layers):
    inputs = np.array([neuron.forward(inputs) for neuron in layer])

output_data = np.array([neuron.forward(inputs) for neuron in output_layer])
print(f"For-loop approach output: {output_data}")

# 2) - Code using NeuralNetwork class
inputs = np.array([1, 2])
neural_net = NeuralNetwork(hidden_layers, output_layer)
output = neural_net.forward(inputs)
print(f"Neural network output: {output}")

#------------------------------------------------------------------------------------------------------
#TRAINING LOOP

# This will make our experiments reproducible (not random)
np.random.seed(42)
def train(network, inputs, targets, epochs=10000, learning_rate=0.1, print_every=1000):
    """
    inputs (list or np.ndarray): The input data for training.
    targets (list or np.ndarray): The target outputs corresponding to the input data.
    epochs (int, optional): The number of training iterations. Default is 10000.
    learning_rate (float, optional): The learning rate for the backpropagation algorithm. Default is 0.1.
    print_every (int, optional): The interval (in epochs) at which to print the total error. Default is 1000.
    """
    for epoch in range(epochs):
        # Train on each input-output pair
        for input_data, target in zip(inputs, targets):
            network.backpropagate(input_data, target, learning_rate)
        # Every once in a while, print the total error to see progress (we should see
        # it decrease)
        if epoch % print_every == 0:
            total_error = 0
            for input_data, target in zip(inputs, targets):
                predicted_output = network.forward(input_data)
                total_error += np.sum((target - predicted_output) ** 2)
            print(f"Epoch {epoch} - Total Error: {total_error:.4f}")
            
#-------------------------------------------------------------------------------
#EVALUATE OUR MODEL
"""
# XOR FUNCTION 
#define the date of the function that we want to lean 
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

#example network architecture: 2 inputs, 2 hidden layers w/ 2 neurons each, 1 output neuron
num_inputs = 2
hidden_layers = [
    [
        Neuron(2, sigmoid, sigmoid_derivative),
        Neuron(2, sigmoid, sigmoid_derivative),
    ]
]
output_layer = [Neuron(2, sigmoid, sigmoid_derivative)]
network = NeuralNetwork(hidden_layers=hidden_layers, output_layer=output_layer)

train(network, inputs, targets, epochs = 10000, learning_rate=0.5)
print("Testing the neural network after training: ")
for input_data, target in zip(inputs, targets): 
    predicted_output = network.forward(input_data)
    print("Input: ")
    print(input_data)
    print(" - Predicted output: ")
    print(predicted_output)
    print(" - real output: ")
    print(target)
"""
#--------------------------------------------------------------

# ALTERNATIVELY WE COULD SE A MORE COMPLEX FUNCTION IN A REAL SCENARIO - QUADRATIC FUNCITON
# Define the quadratic function: y = x1^2 + x2^2
def quadratic_function(x):
    return np.sum(np.square(x), axis=1, keepdims=True)

number_of_training_points = 35
number_of_testing_points = 40

# Generate training data
inputs = np.random.uniform(-1, 1, (number_of_training_points, 2))  # 35 random points in 2D space
targets = quadratic_function(inputs)

# Example network architecture: 2 inputs, 1 hidden layer with 4 neurons, 1 output neuron
hidden_layers = [
    [Neuron(2, sigmoid, sigmoid_derivative) for _ in range(4)],
    [Neuron(4, sigmoid, sigmoid_derivative) for _ in range(4)],
]
output_layer = [Neuron(4, sigmoid, sigmoid_derivative)]

network = NeuralNetwork(hidden_layers=hidden_layers, output_layer=output_layer)
train(network, inputs, targets, epochs=4500, learning_rate=0.1, print_every=1000)


#---------------------------------------------------------------------------
#TESTING OUR MODEL 


x_values = np.linspace(-1, 1, number_of_testing_points) # Test the network on a grid of points and plot the results
y_values = np.linspace(-1, 1, number_of_testing_points)
xx, yy = np.meshgrid(x_values, y_values)
test_inputs = np.c_[xx.ravel(), yy.ravel()]


real_outputs = quadratic_function(test_inputs) # Get real function outputs


predicted = np.array([network.forward(input_data) for input_data in test_inputs]) # Get predicted outputs from the neural network
predicted = predicted.reshape(xx.shape)

plt.figure(figsize=(12, 6))

# Plot real function
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, real_outputs.reshape(xx.shape), levels=100, cmap="viridis")
plt.colorbar(label="Real Output (y = x1^2 + x2^2)")
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets.ravel(), cmap="viridis", edgecolors="w", s=100)
plt.title("Real Quadratic Function: $y = x_1^2 + x_2^2$")
plt.xlabel("Input $x_1$")
plt.ylabel("Input $x_2$")

# Plot approximated function
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, predicted, levels=100, cmap="viridis")
plt.colorbar(label="Predicted Output")
plt.title("Neural Network Approximation")
plt.xlabel("Input $x_1$")
plt.ylabel("Input $x_2$")
plt.tight_layout()
plt.show()

#we can put some extra improvements into our code (pdf pg 23)