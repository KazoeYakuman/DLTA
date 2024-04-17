
import numpy as np
import time
from tqdm import tqdm


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
        # num_labels = int.from_bytes(f.read(4), byteorder='big')
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


train_data = train_data.reshape(train_data.shape[0], -1) / 255.0  # (60000, 784)
test_data = test_data.reshape(test_data.shape[0], -1) / 255.0
num_classes = 10
train_label_one_hot = np.eye(num_classes)[train_label]  # (60000, 10)
test_label_one_hot = np.eye(num_classes)[test_label]


class MLP:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W1 = np.random.randn(input_size, 256)
        self.W2 = np.random.randn(256, 64)
        self.W3 = np.random.randn(64, output_size)
        self.b1 = np.zeros(256)
        self.b2 = np.zeros(64)
        self.b3 = np.zeros(output_size)

    def softmax(self, x):
        x = x - np.max(x, axis=-1, keepdims=True)
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def cross_entropy(self, y_pred, y_true):
        return np.mean(-np.sum(y_true * np.log(y_pred + 1e-8), axis=-1))

    def relu(self, x):
        return np.maximum(0, x)

    def train(self, X, y, epoch=100, lr=0.2):
        for e in tqdm(range(epoch)):
            h1 = X @ self.W1 + self.b1
            h1 = self.relu(h1)
            h2 = h1 @ self.W2 + self.b2
            h2 = self.relu(h2)
            y_pred = h2 @ self.W3 + self.b3
            y_pred = self.softmax(y_pred)
            loss = self.cross_entropy(y_pred, y)
            if e % 10 == 9:
                print(loss)
            dy = y_pred - y
            dW3 = h2.T @ dy / X.shape[0]
            db3 = np.sum(dy, axis=0) / X.shape[0]
            dh2 = dy @ self.W3.T * (h2 > 0) / X.shape[0]
            dW2 = h1.T @ dh2 / X.shape[0]
            db2 = np.sum(dh2, axis=0) / X.shape[0]
            dh1 = dh2 @ self.W2.T * (h1 > 0) / X.shape[0]
            dW1 = X.T @ dh1 / X.shape[0]
            db1 = np.sum(dh1, axis=0) / X.shape[0]
            self.W3 -= lr * dW3
            self.b3 -= lr * db3
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

    def test(self, X, y):
        h1 = X @ self.W1 + self.b1
        h1 = self.relu(h1)
        h2 = h1 @ self.W2 + self.b2
        h2 = self.relu(h2)
        y_pred = h2 @ self.W3 + self.b3
        y_pred = self.softmax(y_pred)
        acc = np.mean(np.argmax(y_pred, axis=-1) == np.argmax(y, axis=-1))
        print(acc)


model = MLP(784, 10)
model.train(train_data, train_label_one_hot, epoch=100, lr=0.01)
model.test(test_data, test_label_one_hot)
model.test()
time.sleep(999)
