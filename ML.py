import numpy as np
import matplotlib.pyplot as plt
import os
import copy as c
import pickle as pk
import seaborn as sb
class Logistic_regression_model:
    def __init__(self , X , Y , learning_rate = .1 , EPOCHS = 3000):
        self.X = X
        self.Y = Y
        self.learning_rate = learning_rate
        self.EPOCHS = EPOCHS
        self.W = None
        self.b = None
        self.costs = []
    
    def sigmoid(self , Z):
        g = 1 /(1 + np.exp(-1 * Z))
        return g

    def initialize(self):
        m , n = self.X.shape
        W = np.random.randn(1 , n)
        b = 0
        return W , b
           
    def compute_linear(self , W , b , X):
        Z = np.dot(W , X.T) + b
        g = self.sigmoid(Z)

        return g
    
    def cost(self , g):
        m , n = self.X.shape
        Y = self.Y.T
        loss = Y * np.log(g) + (1 - Y) * np.log(1 - g)

        cost_v = float((-1/m) * np.sum(loss))

        return cost_v
    
    def derivatives(self , g):
        m , n = self.X.shape

        dW = (1/m) * np.dot(g - self.Y.T , self.X)
        db = (1/m) * np.sum(g - self.Y.T)

        return dW , db
    

    def update(self , dW, db):
        
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db
    def fit(self):
        self.W , self.b = self.initialize()

        for _ in range(self.EPOCHS):
            g = self.compute_linear(self.W , self.b , self.X)
            cost = self.cost(g)
            self.costs.append(cost)
            dW , db = self.derivatives(g)
            self.update(dW , db)
    
    def predict(self , X , threshold = 0.5):
        predictions = self.compute_linear(self.W , self.b , X)
        
        return (predictions > threshold).astype("int")[0]


def plot_costs(costs):
    if not os.path.exists("graphs"):
        os.mkdir("graphs")
    fig = plt.figure(figsize=[10 , 10])
    ax = fig.add_subplot(1 , 1, 1)
    ax.plot([i + 1 for i in range(len(costs))] , costs)
    ax.set_xlabel("number of iterations")
    ax.set_ylabel("cost value")
    fig.savefig("graphs/costs_figure.png")
    plt.show()

def plot_accuracies(library_accuracy , my_accuracy , filename = "accuracy_figure" , name_1 = "library_model" , name_2 = "my_model"):
    if not os.path.exists("graphs"):
        os.mkdir("graphs")
    fig = plt.figure(figsize=[10 , 10])
    ax = fig.add_subplot(1 , 1, 1)
    ax.bar([name_1 , name_2] , height=[library_accuracy , my_accuracy])
    ax.set_xlabel("models")
    ax.set_ylabel("accuracy value")
    fig.savefig(f"graphs/{filename}.png")
    plt.show()

def save_model(saved_model , filename):
    if not os.path.exists("models"):
        os.mkdir("models")
    with open(f"models/{filename}.pkl" , "wb") as f:
        pk.dump(saved_model , f)

def load_model(filename):
    with open(f"{filename}.pkl" , "rb") as f:
        saved_model = pk.load(f)
    return saved_model


def complex_matrix(y_true , y_pred , num_classes = 2):
    matrix = np.zeros(shape = (num_classes , num_classes))
    y_true = list(y_true)
    y_pred = list(y_pred)
    for i in range(len(y_true)):
        actual = y_true[i]
        predicted = y_pred[i] 
        matrix[int(actual)][int(predicted)] += 1
    
    return matrix


def save_heatmap(cm , filename = "complex_matrix" , name = "complex_matrix"):
    ax = sb.heatmap(cm , annot = True , fmt = '.0f')
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(name)
    plt.show()
    ax.get_figure().savefig(f"graphs/{filename}.png")

class NN_Model:
    def __init__(self , X , Y , learning_rate = .1 , EPOCHS = 3000 , layers_dims = [1]):

        self.X = X
        if type(Y) != np.ndarray:
            self.Y = Y.to_numpy()
        else:
            self.Y = Y
        self.Y = self.Y.reshape((self.Y.shape[0] , 1))
        self.learning_rate = learning_rate
        self.EPOCHS = EPOCHS
        self.layers = [self.X.shape[1]] + layers_dims + [1]
        self.parameters = None
    
    def _sigmoid(self , Z):
        A = 1 / (1 + np.exp(-1 * Z))

        cache = {"A" : A , "Z": Z}

        return cache
    
    def _derivative_sigmoid(self , Z):
        cache = self._sigmoid(Z)
        A = cache['A']
        derivative = A * (1 - A)

        return derivative
    
    def _relu(self , Z):
        zeros = np.zeros(shape = Z.shape)
        A = np.maximum(zeros , Z)
        cache = {"A" : A , "Z" : Z}
        
        return cache
    

    def _derivatvie_relu(self , Z):
        
        derivative = (Z > 0).astype('int')
        
        return derivative

    def _linear(self , W , b , A):


        Z = np.dot(W , A) + b
        cache = {'Z' : Z , 'W' : W , "b" : b}
        return cache
    
    def initialize_parameters(self):
        L = len(self.layers)
        all_W = []
        all_b = []
        for l in range(1 , L):
            W = np.random.randn(self.layers[l] , self.layers[l - 1]) * np.sqrt(2/self.layers[l - 1])
            b = np.zeros(shape = (self.layers[l] , 1))
            all_W.append(W)
            all_b.append(b)

        parameters = {'all_W' : all_W, 'all_b': all_b}

        self.parameters = parameters
        return parameters
    

    def forward_propagation(self , parameters , X = None):

        all_A = []
        all_Z = []
        all_W = parameters['all_W']
        all_b = parameters['all_b']
        A = X.T
        L = len(self.layers)
        

        for l in range(L - 2):
                A_prev = A
                w = all_W[l]
                b = all_b[l]
                cache_1 = self._linear(w , b , A_prev)
                Z = cache_1['Z']
                cache_2 = self._relu(Z)
                A = cache_2['A']
                all_A.append(A)
                all_Z.append(Z)

        
        w = all_W[L - 2]
        b = all_b[L - 2]
        cache_1 = self._linear(w , b , A)
        Z = cache_1['Z']
        all_Z.append(Z)
        cache_2 = self._sigmoid(Z)
        AL = cache_2['A']
        all_A.append(AL)
        cache = {'AL' :AL , 'all_W' :all_W , "all_b" : all_b , 'all_A' : all_A , 'all_Z' :all_Z}

        return cache

    def cost(self , AL):
        m , n = self.X.shape
        Y = self.Y.T

        loss = Y * np.log(AL) + (1 - Y) * np.log(1 - AL)

        cost = (-1/m) * np.sum(loss)

        return cost
    

    def linear_backprop(self , dA , Z , activation):
        

        if activation == 'sigmoid':
            dZ = dA * self._derivative_sigmoid(Z)

        if activation == 'relu':
            dZ = dA * self._derivatvie_relu(Z)
        
        cache = {'dZ' : dZ , 'dA' : dA , 'Z' : Z}

        return cache
    
    def derivative_backprop(self , dZ , A_prev , W):
        m , n = self.X.shape

        dW = (1/m) * np.dot(dZ , A_prev.T)
        db = (1/m) * np.sum(dZ , axis = 1 , keepdims = True)
        dA_prev = np.dot(W.T , dZ)

        cache = {'dW' : dW , 'db' : db , 'dA_prev' : dA_prev}

        return cache 
    
    def _calculate_dAL(self , AL):
        Y = self.Y.T
        dAL = (-Y/AL) + ((1 - Y)/(1 - AL))

        return dAL

    def back_propagation(self , parameters):
        
        AL = parameters['AL']
        all_W = parameters['all_W']
        L = len(all_W)
        all_A = [self.X.T] + parameters['all_A']
        all_Z = parameters['all_Z']
        all_dW = [0] * len(all_W)
        all_db = [0] * len(all_W) 

        dAL = self._calculate_dAL(AL)
        A_prev = all_A[-2]
        W = all_W[-1]
        Z = all_Z[-1]
        cache_1 = self.linear_backprop(dAL , Z , activation='sigmoid')
        dZ = cache_1['dZ']
        cache_2 = self.derivative_backprop(dZ , A_prev  , W)
        dW = cache_2['dW']
        db = cache_2['db']
        all_dW[-1] = dW
        all_db[-1] = db
        dA_prev = cache_2['dA_prev']

        
        for l in range(L - 2 , -1 , -1):
            
            A_prev = all_A[l]
            W = all_W[l]
            Z = all_Z[l]

            cache_1 = self.linear_backprop(dA_prev , Z , activation='relu')
            dZ = cache_1['dZ']

            cache_2 = self.derivative_backprop(dZ , A_prev  , W)
            dW = cache_2['dW']
            db = cache_2['db']

            all_dW[l] = dW
            all_db[l] = db
            dA_prev = cache_2['dA_prev']


        parameters = {'all_dW' : all_dW , 'all_db' : all_db , **parameters}
        return parameters 

    def update(self , parameters):
        all_dW = parameters['all_dW']
        all_db = parameters['all_db']
        all_W  = c.deepcopy(parameters['all_W'])
        all_b  = c.deepcopy(parameters['all_b'])

        for i in range(len(all_W)):
            all_W[i] -= self.learning_rate * all_dW[i]
            all_b[i] -= self.learning_rate * all_db[i]

        new_parameters = {"all_W"  : all_W , "all_b" : all_b}

        return new_parameters
    

    def fit(self):

        parameters = self.initialize_parameters()
        costs = []
        for _ in range(self.EPOCHS):

            cache = self.forward_propagation(parameters , X = self.X)
            AL = cache['AL']
            cost = self.cost(AL)
            costs.append(cost)
            parameters_backprop = self.back_propagation(cache)
            parameters = self.update(parameters=parameters_backprop)

        self.parameters = parameters

        return parameters , costs
    
    def predict(self , X , threshold = .5):

        result = self.forward_propagation(self.parameters , X)['AL']


        return (result > threshold).astype('int')[0]
