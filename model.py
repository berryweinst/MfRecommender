import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy import sparse
from data import *


class MFModel(object):

    def __init__(self, rates_dict, test_rates_dict, num_users, num_items, hidden_dim, lr, gamma, epochs, optimizer,
                 reg_u, reg_i, confidence=1):
        """ Params:
        1) ratings - rating matrix (as dict)
        2) hidden_dim  - number of latent dimensions (as integer)
        3) lr - learning rate (as float)
        4) gamma - regulaizer (as float)
        5) opechs - as int
        """

        self.ratings_dict = rates_dict
        self.test_ratings_dict = test_rates_dict
        self.num_users = num_users
        self.num_items = num_items
        self.K = hidden_dim
        self.optimizer = optimizer
        self.lr = lr
        self.gamma = gamma
        self.regulizer_items = reg_i
        self.regulizer_users = reg_u
        self.confidence = confidence
        self.epochs = epochs
        self.scbd = []

    def print_hyper_params(self):
        print("latent dimention %d" % (self.K))
        print("learning rate %r" % (self.lr))
        print("lambda SGD %r" % (self.gamma))
        print("lambda ALS users %r" % (self.regulizer_users))
        print("lambda ALS items %r" % (self.regulizer_items))
        print("Confidence ALS %d" % (self.confidence))
        print("Epochs %d" % (self.epochs))

    def init(self, isize, ik, kind='normal'):
        if kind == 'normal':
            return np.random.normal(scale=1.0 / ik, size=(isize, ik))
        elif kind == 'uniform':
            return np.random.uniform(-1.0 / ik, 1.0 / ik)

    def train_model(self):

        def reset(samples):
            np.random.shuffle(samples)

        self.samples = []
        # prepare the samples samples
        for u, items in self.ratings_dict.items():
            for i, r, _ in items:
                self.samples += [(u, i, r)]

        self.U = self.init(self.num_users, self.K, kind='normal')
        self.V = self.init(self.num_items, self.K, kind='normal')

        #sparse intermidiate representation
        self.Us = sparse.csr_matrix(self.U)
        self.Vs = sparse.csr_matrix(self.V)

        # Initialize the biases
        self.bu = np.zeros(self.num_users)
        self.bv = np.zeros(self.num_items)

        for epoch in range(self.epochs):
            reset(self.samples)
            if self.optimizer == 'sgd':
                self.LearnModelFromDataUsingSGD()
            elif self.optimizer == 'als':
                self.LearnModelFromDataUsingALS()
            train_rmse = self.RMSE(is_train=True)
            val_rmse = self.RMSE()
            print("[Epoch %d] : train RMSE = %.4f" % (epoch, train_rmse))
            if epoch % 5 == 0:
                print("[Epoch %d] : validation RMSE = %.4f" % (epoch, val_rmse))
            self.scbd.append((epoch, train_rmse, val_rmse))

    def LearnModelFromDataUsingSGD(self):
        """
        Performs one iteration on one of the User/Items matrices
        """
        for u, v, rate in self.samples:
            prediction = self.bu[u] + self.bv[v] + self.U[u, :].dot(self.V[v, :].T)
            err = (rate - prediction)

            # update biases
            self.bu[u] += self.lr * (err - self.gamma * self.bu[u])
            self.bv[v] += self.lr * (err - self.gamma * self.bv[v])

            # update user item metrices
            self.U[u, :] += self.lr * (err * self.V[v, :] - self.gamma * self.U[u, :])
            self.V[v, :] += self.lr * (err * self.U[u, :] - self.gamma * self.V[v, :])



    def LearnModelFromDataUsingALS(self):
        """
        Performs one iteration on one of the User/Items matrices
        """
        rate_mat = np.zeros((self.num_users, self.num_items))
        for user, item in self.ratings_dict.items():
            for movie, rate, _ in item:
                rate_mat[user][movie] = rate
        rate_mat = sparse.csr_matrix(rate_mat) * self.confidence

        U_i = sparse.eye(self.num_users)
        V_i = sparse.eye(self.num_items)

        # UtU = self.Us.T.dot(self.Us)
        # VtV = self.Vs.T.dot(self.Vs)

        # Users update
        lambda_u = sparse.eye(self.K) * self.regulizer_users
        for u_idx, u_rate in enumerate(rate_mat):
            u_dense = u_rate.toarray()
            u_bin = u_dense.copy()
            u_bin[u_bin != 0] = 1.0
            u_diag = sparse.diags(u_dense, [0])
            VtV = self.Vs.T.dot(u_diag).dot(self.Vs)
            VtV_u = self.Vs.T.dot(u_diag).dot(u_dense.T)
            self.Us[u_idx, :] = spsolve(VtV + lambda_u, VtV_u)



        # Items update
        lambda_i = sparse.eye(self.K) * self.regulizer_items
        for i_idx, i_rate in enumerate(rate_mat.T):
            i_dense = i_rate.toarray()
            i_bin = i_dense.copy()
            i_bin[i_bin != 0] = 1.0
            i_diag = sparse.diags(i_dense, [0])
            UtU = self.Us.T.dot(i_diag).dot(self.Us)
            UtU_i = self.Us.T.dot(i_diag).dot(i_dense.T)
            self.Vs[i_idx, :] = spsolve(UtU + lambda_i, UtU_i)

        self.U = self.Us.toarray()
        self.V = self.Vs.toarray()



    def RMSE(self, is_train=False):
        mod_pred = self.bu[:, np.newaxis] + self.bv[np.newaxis:, ] + self.U.dot(self.V.T)
        total_err = 0
        r_cnt = 0
        ratings_dict = self.ratings_dict if is_train else self.test_ratings_dict
        for u, items in ratings_dict.items():
            for i, r, _ in items:
                total_err += pow(r - mod_pred[u, i], 2)
                r_cnt += 1
        return np.sqrt(total_err / r_cnt)