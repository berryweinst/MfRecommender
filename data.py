import numpy as np
from data import *


class MFModel(object):

    def __init__(self, rates_arr, hidden_dim, lr, gamma, opechs):
        """
        Performs matrix factorization
        Args
        1) ratings - rating matrix (as NDarray)
        2) hidden_dim  - number of latent dimensions (as integer)
        3) lr - learning rate (as float)
        4) gamma - regulaizer (as float)
        5) opechs - as int
        """

        self.ratings = rates_arr
        self.num_users, self.num_items = rates_arr.shape
        self.K = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.opechs = opechs

    def train(self):
        # Initialize user and item latent feature matrice
        self.U = np.random.normal(scale=1. / self.K, size=(self.num_users, self.K))
        self.V = np.random.normal(scale=1. / self.K, size=(self.num_items, self.K))

        # Initialize the biases
        self.b_u = np.zeros(self.num_users)
        self.b_v = np.zeros(self.num_items)
        # self.b = np.mean(self.ratings[np.where(self.ratings != 0)])

        # Create a list of training samples
        self.samples = [
            (i, j, self.ratings[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.ratings[i, j] > 0
        ]

        # Perform stochastic gradient descent for number of iterations
        training_process = []
        for i in range(self.opechs):
            np.random.shuffle(self.samples)
            self.LearnModelFromDataUsingSGD()
            rmse = self.RMSE()
            training_process.append((i, rmse))
            if (i + 1) % 5 == 0:
                print("Iteration: %d ; error = %.4f" % (i + 1, rmse))

        return training_process

    def RMSE(self):
        xs, ys = self.ratings.nonzero()
        pred = self.b_u[:,np.newaxis] + self.b_v[np.newaxis:,] + self.U.dot(self.V.T)
        error = 0
        for x, y in zip(xs, ys):
            error += pow(self.ratings[x, y] - pred[x, y], 2)
        return np.sqrt(error / xs.shape[0])



    def LearnModelFromDataUsingSGD (self):
        for i, j, rate in self.samples:
            prediction = self.b_u[i] + self.b_v[j] + self.U[i, :].dot(self.V[j, :].T)
            e = (rate - prediction)

            # bias update
            self.b_u[i] += self.lr * (e - self.gamma * self.b_u[i])
            self.b_v[j] += self.lr * (e - self.gamma * self.b_v[j])

            # user item metrices
            self.U[i, :] += self.lr * (e * self.V[j, :] - self.gamma * self.U[i, :])
            self.V[j, :] += self.lr * (e * self.U[i, :] - self.gamma * self.V[j, :])



if __name__ == '__main__':
    X_train, X_test = create_user_item_map('../ml-1m/ratings.dat', '', percent=0.8, sort_ratings=True)
    mf = MFModel(X_train, 5, 0.01, 0.01, 100)
    res = mf.train()
