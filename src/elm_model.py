import numpy as np
import torch
from read_matlab_data import read_matlab_data
import matplotlib.pyplot as plt

class ELM():
    def __init__(self, training_data, config) -> None:
        self.n_dim = config.n_dim
        self.n_hidden = config.n_hidden
        self.stiffness_learning = config.stiffness_learning
        self.n_dim_kp = config.n_dim_kp
        self.feedback = config.feedback

        input_data = training_data[:, 0:self.n_dim]
        target_data = training_data[:, self.n_dim:]
        w1 = 2*np.random.rand(self.n_hidden, self.n_dim)-1
        adjust = np.max(abs(input_data), axis=0)
        w1 = w1/np.tile(adjust, (self.n_hidden, 1))
        b1= 2*np.random.rand(self.n_hidden)-1
        w1 = w1.T
        tempH = np.dot(input_data, w1) + b1
        h = np.tanh(tempH)
        w2 = np.dot(np.linalg.pinv(h), target_data)

        self.w1 = w1
        self.b1 = b1
        self.theta0 = np.reshape(w2, (-1, 1))
        self.theta = self.theta0


    def predict(self, w, ref_w, theta_eps):

        if self.feedback == 0:
            tempH = np.dot(ref_w, self.w1) + self.b1   
            # print(tempH)
        elif self.feedback == 1:
            tempH = np.dot(w, self.w1) + self.b1 
        else:
            raise Exception("Please the value of feedback in the configuration!")

        h = np.tanh(tempH)
        w2 = np.reshape(theta_eps, (self.n_hidden, -1))
        output = np.dot(h, w2)
        return output


        

