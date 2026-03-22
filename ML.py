import numpy as np
import matplotlib.pyplot as plt
import os

import pickle as pk
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

def plot_accuracies(library_accuracy , my_accuracy):
    if not os.path.exists("graphs"):
        os.mkdir("graphs")
    fig = plt.figure(figsize=[10 , 10])
    ax = fig.add_subplot(1 , 1, 1)
    ax.bar(["library_model" , "my_model"] , height=[library_accuracy , my_accuracy])
    ax.set_xlabel("models")
    ax.set_ylabel("accuracy value")
    fig.savefig("graphs/accuracy_figure.png")
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

