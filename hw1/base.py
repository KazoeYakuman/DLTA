
import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        assert magic == 2051, 'Invalid magic number ' + str(magic)
        num_images = int.from_bytes(f.read(4), byteorder='big')
        num_rows = int.from_bytes(f.read(4), byteorder='big')
        num_cols = int.from_bytes(f.read(4), byteorder='big')
        data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_images, num_rows, num_cols)
        return data


def load_label(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), byteorder='big')
        assert magic == 2049, 'Invalid magic number ' + str(magic)
        num_labels = int.from_bytes(f.read(4), byteorder='big')
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels


train_data_path = './data/train_data'
train_label_path = './data/train_label'
test_data_path = './data/test_data'
test_label_path = './data/test_label'
train_data = load_data(train_data_path)
train_label = load_label(train_label_path)
test_data = load_data(test_data_path)
test_label = load_label(test_label_path)


class NeuralNetwork:
    def __init__(self, input_size, output_size, *hidden_sizes):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.parameters = self.initialize_parameters()

    def initialize_parameters(self):
        parameters = {}
        layer_dims = [self.input_size] + (list)(self.hidden_sizes) + [self.output_size]
        for i in range(1, len(layer_dims)):
            parameters[f'W{i}'] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
            parameters[f'b{i}'] = np.zeros((layer_dims[i], 1))
        return parameters

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=-1, keepdims=True)

    def compute_loss(self, A_output, Y):
        return -np.sum(Y * np.log(A_output + 1e-8)) / Y.shape[-1]

    def relu_derivative(self, Z):
        return np.where(Z > 0, 1, 0)

    def forward_propagation(self, X):
        caches = []
        A = X
        for i in range(1, len(self.hidden_sizes) + 1):
            Z = np.dot(self.parameters[f'W{i}'], A) + self.parameters[f'b{i}']
            A = self.relu(Z)
            caches.append((A, Z))
        Z_output = np.dot(self.parameters[f'W{len(self.hidden_sizes)+1}'], A) + self.parameters[f'b{len(self.hidden_sizes)+1}']
        A_output = self.softmax(Z_output)
        caches.append((A_output, Z_output))
        return A_output, caches

    def backward_propagation(self, X, Y, caches):
        grads = {}
        m = X.shape[-1]
        L = len(caches)

        A_output, _ = caches[-1]
        dZ_output = A_output - Y
        grads[f'dW{L}'] = np.dot(dZ_output, caches[-2][0].T) / m
        grads[f'db{L}'] = np.sum(dZ_output, axis=1, keepdims=True) / m

        dA_prev = np.dot(self.parameters[f'W{L}'].T, dZ_output)
        for i in reversed(range(1, L)):
            A, Z = caches[i-1]
            dZ = dA_prev * self.relu_derivative(Z)
            grads[f'dW{i}'] = np.dot(dZ, A.T) / m
            grads[f'db{i}'] = np.sum(dZ, axis=-1, keepdims=True) / m
            dA_prev = np.dot(self.parameters[f'W{i}'].T, dZ)

        return grads

    def update_parameters(self, grads, lr):
        for i in range(1, len(self.hidden_sizes) + 2):
            self.parameters[f'W{i}'] -= lr * grads[f'dW{i}']
            self.parameters[f'b{i}'] -= lr * grads[f'db{i}']

    def train(self, X, Y, epochs, lr):
        for epoch in range(epochs):
            A_output, caches = self.forward_propagation(X)
            grads = self.backward_propagation(X, Y, caches)
            loss = self.compute_loss(A_output, Y)
            self.update_parameters(grads, lr)
            if epoch % 100 == 0:
                print(f'epoch {epoch}: loss {loss}')

    def predict(self, X):
        A_output, _ = self.forward_propagation(X)
        return np.argmax(A_output, axis=1)


train_data = train_data.reshape(train_data.shape[0], -1).T / 255.0
test_data = test_data.reshape(test_data.shape[0], -1).T / 255.0
num_classes = 10
train_label_one_hot = np.eye(num_classes)[train_label].T
test_label_one_hot = np.eye(num_classes)[test_label].T

nn = NeuralNetwork(train_data.shape[0], num_classes, 256, 64)
nn.train(train_data, train_label_one_hot, epochs=1000, lr=0.001)
predictions = nn.predict(test_data)
accuracy = np.mean(predictions == test_label)
print(f'accuracy: {accuracy}')
