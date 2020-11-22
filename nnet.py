import numpy as np
import tensorflow as tf
import random


# hyperparameters
x_len = 60000
batch_size = 16
LR = 0.01
epochs = 5
cost = 'cross_entropy'

neurons = [[784,16], [16,16], [16,10]]
activations = ['relu', 'relu', 'sigmoid']

# dataset
(x_train, label_train), (x_test, label_test) = tf.keras.datasets.mnist.load_data()

# make pixel values between 0 an 1
X = [x_train[0:x_len]/255]
x_test = [x_test/255]

def create_labels(y):
    ''' Make an array out of the labels '''
    x = []
    for i in y[0]:
        template = [0,0,0,0,0,0,0,0,0,0]
        template[i] = 1
        x.append(template)
    return x

y = create_labels([label_train[0:x_len]])
y_test = create_labels([label_test])

y_test = np.array(y_test)
x_test = np.array(x_test)

y = np.array(y)
X = np.array(X)

X = np.reshape(X, (x_len, 784))

training_data = []
test_data = []

for i in range(len(X)):
    training_data.append((X[i],y[i]))
for i in range(len(x_test)):
    test_data.append((x_test[i],y_test[i]))

class DenseLayer:
    def __init__(self, n_inputs, n_neurons, activation):
        self.size = (n_inputs, n_neurons)

        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.activation = activation

    def forward(self, inputs):
        self.z = np.dot(inputs, self.weights) + self.biases

        if self.activation == 'relu':
            self.a = Activations().ReLU(self.z)

        elif self.activation == 'sigmoid':
            self.a = Activations().sigmoid(self.z)

        else:
            print('no activation found')


class Network:
    def __init__(self, n_layers, n_neurons, activations, cost='MSE'):
        self.layers = []
        self.cost = cost
        self.error = None
        self.eval = []
        self.acc = 0
        self.graph = []

        if len(n_neurons) == n_layers:
            for i in range(n_layers):
                self.layers.append(DenseLayer(n_neurons[i][0],n_neurons[i][1],
                 activations[i]))
        self.layers


    def updated_biases(self, delta):
        tmp = []

        for d in delta[0]:
            tmp.append(d)
        return np.array(tmp).T

    def updated_weights(self, delta, a):
        tmp = []

        for d in delta[0]:
            w_n = [] # weight of neuron n
            for i in a[0]:
                w_n.append(i*d) # al-1 * deltal

            tmp.append(w_n)
        return np.array(tmp).T

    def delta_L(self, aL, zL,  y):
        ''' Delta of the final layer(L) '''
        #a = aL - y
        a = CostFunctions().ce_gradient(aL, y)
        b = Activations().sigmoid_prime(zL)

        return a * b

    def delta_l(self, wl_1, deltal_1, zl):
        ''' Delta of layer l
            delta_l_1 = delta_L in the second to last layer '''

        deltal = np.dot(deltal_1, wl_1.T) * Activations().ReLU_prime(zl)
        return deltal


    def backprop(self, x, y):
        # 0 is the first hidden layer
        # 2 is the output layer

        self.layers[0].forward(x)
        self.layers[1].forward(self.layers[0].a)
        self.layers[2].forward(self.layers[1].a)

        deltaL = self.delta_L(self.layers[2].a, self.layers[2].z, y) # deltal_2
        deltal_1 = self.delta_l(self.layers[2].weights, deltaL, self.layers[1].z)
        deltal_0 = self.delta_l(self.layers[1].weights, deltal_1, self.layers[0].z)


        w3 = self.updated_weights(deltaL, self.layers[1].a)
        w2 = self.updated_weights(deltal_1, self.layers[0].a)
        w1 = self.updated_weights(deltal_0, [x])

        b3 = self.updated_biases(deltaL)
        b2 = self.updated_biases(deltal_1)
        b1 = self.updated_biases(deltal_0)

        # data
        #self.eval = [self.layers[2].a, y]
        #self.error = deltaL

        alist = self.layers[2].a.tolist()[0]
        pred = np.max(alist)
        ind = alist.index(pred)

        if ind == y.tolist().index(1):
            self.acc += 1
        #----
        return [w1, w2, w3], [b1, b2, b3]

    def SGD(self, training_data, LR, batch_size, epochs):
        for e in range(epochs):
            print(f'epoch: {e+1}\n')
            self.acc = 0

            random.shuffle(training_data)

            nabla_w = []
            nabla_b = []

            for i in range(len(training_data)):

                x, y = training_data[i][0], training_data[i][1]
                ws, bs = self.backprop(x, y)

                if len(nabla_w) == 0:
                    nabla_w = ws
                    nabla_b = bs

                for f in range(len(ws)):
                    nabla_w[f] += ws[f]
                    nabla_b[f] += bs[f]


                if i % batch_size == 0:
                    self.update_batch(batch_size, LR, nabla_w, nabla_b)
                    nabla_w = []
                    nabla_b = []
                    #print(f'{self.cost}: {CostFunctions().MSE(self.eval[0], self.eval[1])}')
                    #print(f'error: {self.error}')
                    print(f'acc: {round((self.acc/(i+1))*100, 2)}', end="\r", flush=True)

                    #print(f'{round(i / len(training_data), 2)}% ', end="\r", flush=True)
                    self.graph.append(self.acc/(i+1)*100)


    def update_batch(self, batch_size, LR, nabla_w, nabla_b):
        for i, l in enumerate(self.layers):

            self.layers[i].weights -= (LR / batch_size) * nabla_w[i]
            self.layers[i].biases -= (LR / batch_size) * nabla_b[i]


    def evaluation(self, data):
        acc = 0
        for x in data:
            x = x[0]
            y = x[1]
            self.layers[0].forward(x)
            self.layers[1].forward(self.layers[0].a)
            self.layers[2].forward(self.layers[1].a)

            alist = self.layers[2].a.tolist()[0]
            pred = np.max(alist)
            ind = alist.index(pred)

            if ind == y.tolist().index(1):
                acc += 1
        return acc/len(data)


class CostFunctions:
    def MSE(self, aL, y):
        return ((aL - y)**2).sum() / len(aL)

    def MSE_gradient(self, aL, y):
        ''' partial derivative dC/da  '''
        return 2*(aL - y)

    def cross_entropy(self, aL, y):
        ''' log() = ln()  log10() =log()'''

        return -(y * np.log(aL) + (1-y) * log(1-aL)).sum()  / len(aL)

    def ce_gradient(self, aL, y):
        ''' log() = ln()  log10() =log()'''
        l = -y/aL
        r = (1-y) / (1-aL)
        return (l + r) / len(aL)
        #return (aL - y)/((1-aL)*aL)

class Activations:
    def sigmoid(self, inputs):
        return 1/(1 + np.exp(-inputs))

    def ReLU(self, inputs):
        return np.maximum(0, inputs)

    def sigmoid_prime(self, inputs):
        '''derivative of sigmoid function'''
        sigmoid = self.sigmoid(inputs)
        return sigmoid * (1-sigmoid)

    def ReLU_prime(self, inputs):
        out = []
        for i in inputs[0]:
            if i > 0:
                out.append(1)
            else:
                out.append(0)
        return np.array(out)

def printNN(nn):
    for l in range(len(nn)):
        print(nn[l].size, nn[l].activation)
    print('\n')


#init network
nnet = Network(len(neurons), neurons, activations, cost)

# sgd
nnet.SGD(training_data, LR, batch_size, epochs)


printNN(nnet.layers)



from matplotlib import pyplot as plt

y = nnet.graph
x = range(len(y))

plt.plot(x, y)
plt.show()
