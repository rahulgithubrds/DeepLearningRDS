import numpy as np

# tiny databse XOR concept

X = np.array([[0, 0], [0, 1], [1,0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)

class NeuralNetwork:
    def __init__(self, lr=0.1, hidden_size=2):
        # hyperparameters
        self.lr = lr
        self.hidden_size = hidden_size

        # weights & bias initialization
        self.W1 = np.random.randn(2, hidden_size) * 0.5  # input to hidden layer - weights
        print("w1 shape:", self.W1.shape)
        print("w1:", self.W1)
        self.b1 = np.zeros((1, hidden_size))
        print("b1 shape:", self.b1.shape)
        print("b1:", self.b1)
        self.W2 = np.random.randn(hidden_size, 1) * 0.5
        print("w2 shape:", self.W2.shape)
        print("w2:", self.W2)
        self.b2 = np.zeros((1, 1))
        print("b2 shape:", self.b2.shape)
        print("b2:", self.b2)

    def tanh(self, x):
        return np.tanh(x)
    
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def forward(self, X):
        # layer 1 - hidden layer
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.tanh(self.z1)

        # layer 2 - output layer
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        # Output layer error
        d_z2 = (output - y)
        
        # hidden layer error
        d_a1 = np.dot(d_z2, self.W2.T)
        d_tanh = 1 - np.square(self.a1)
        d_z1 = d_a1 * d_tanh

        # update weights 
        self.W2 -= self.lr * np.dot(self.a1.T, d_z2)
        self.b2 -= self.lr * np.sum(d_z2, axis=0, keepdims=True)
        self.W1 -= self.lr * np.dot(X.T, d_z1)
        self.b1 -= self.lr * np.sum(d_z1, axis=0, keepdims=True)
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            # print(f"Epoch {epoch}: Output = {output.flatten()}")
            self.backward(X, y, output)
            
            if epoch % 200 == 0:
                # avoid log(0) by clipping output
                epsilon = 1e-8

                output_clipped = np.clip(output, epsilon, 1 - epsilon)
                # print(f"Epoch {epoch}: Clipped Output = {output_clipped.flatten()}")

                # Loss = -[y*log(ŷ) + (1-y)*log(1-ŷ)]
                loss = np.mean(-y * np.log(output_clipped) - (1 - y) * np.log(1-output_clipped))
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

                if epoch % 2000 == 0 or epoch == epochs-1:
                    print("Predictions:")
                    print(np.round(output, 4))
    
nn = NeuralNetwork(lr=0.5, hidden_size=4)
nn.train(X, y, epochs=2000)

# final evaluation
final_output = nn.forward(X)
print("Final Output:")
print(np.round(final_output, 4))
print("round predictions:")
print(np.round(final_output).astype(int))