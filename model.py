import numpy as np
from numpy.linalg import solve
from data import *





class MFModel(object):

    def __init__(self, rates_dict, ratings_test, num_users, num_items, hidden_dim, lr, gamma, epochs, optimizer, reg_u, reg_i):
        """ Params:
        1) ratings - rating matrix (as dict)
        2) hidden_dim  - number of latent dimensions (as integer)
        3) lr - learning rate (as float)
        4) gamma - regulaizer (as float)
        5) opechs - as int
        """

        self.ratings_dict = rates_dict
        self.ratings_test = ratings_test
        self.num_users = num_users
        self.num_items = num_items
        self.K = hidden_dim
        self.lr = lr
        self.optimizer = optimizer
        self.regulizer_items = reg_i
        self.regulizer_users = reg_u
        self.gamma = gamma
        self.epochs = epochs

    def init(self, isize, ik, kind='normal'):
        if kind == 'normal':
            return np.random.normal(scale=1.0/ik, size=(isize, ik))
        elif kind == 'uniform':
            return np.random.uniform(-1.0/ik, 1.0/ik)


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

        # Initialize the biases
        self.bu = np.zeros(self.num_users)
        self.bv = np.zeros(self.num_items)


        for epoch in range(self.epochs):
            reset(self.samples)
            if self.optimizer == 'sgd':
                self.LearnModelFromDataUsingSGD()
            elif self.optimizer == 'als':
                self.LearnModelFromDataUsingALS()
            train_rmse = self.RMSE()
            print("[Epoch %d] : train RMSE = %.4f" % (epoch, train_rmse))
            if epoch % 5 == 0:
                eval_rmse = self.RMSE(self.ratings_test)
                print("[Epoch %d] : evaluation RMSE = %.4f" % (epoch, eval_rmse))


    def LearnModelFromDataUsingSGD (self):
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
        # Users update
        VtV = self.V.T.dot(self.V)
        lambda_i = np.eye(self.K) * self.regulizer_users

        for u_idx in range(self.U.shape[0]):
            user_items = np.zeros([self.num_items])
            for i, r, _ in self.ratings_dict[u_idx]:
                user_items[i] = r
            self.U[u_idx,:] = solve((VtV + lambda_i), user_items.dot(self.V))

        # Items update
        UtU = self.U.T.dot(self.U)
        lambda_i = np.eye(self.K) * self.regulizer_items

        for i_idx in range(self.V.shape[0]):
            item_users = np.zeros([self.num_users])
            for u_idx, items in self.ratings_dict.items():
                rate = [t[1] for t in items if i_idx == t[0]] or None
                if rate:
                    item_users[u_idx] = rate[0]
            self.V[i_idx,:] = solve((UtU + lambda_i), item_users.dot(self.U))



    def RMSE(self, rating_dict=None):
        rating_dict = rating_dict if rating_dict != None else self.ratings_dict
        mod_pred = self.bu[:,np.newaxis] + self.bv[np.newaxis:,] + self.U.dot(self.V.T)
        total_err = 0
        r_cnt = 0
        for u, items in rating_dict.items():
            for i, r, _ in items:
                total_err += pow(r - mod_pred[u, i], 2)
                r_cnt += 1
        return np.sqrt(total_err / r_cnt)



# TODO: Igal to implement evaluation class (take RMSE from above)
# class EvalModel(MFModel):
#     def __init__(self, ratings_dict, model):
#         """ Params:
#        ratings_dict - rating matrix (as dict)
#        """
#
#     def EvalModel(self, input, kind='rmse'):
#         if kind == 'rmse':
#             return self.RMSE(input)









if __name__ == '__main__':
    X_train, X_test, users, items = create_user_item_map('../ml-1m/ratings.dat', '', percent=0.8, sort_ratings=True)
    model = MFModel(X_train, X_test, users, items, hidden_dim=5, lr=0.005, gamma=0.001, epochs=100, reg_u=0.01,
                    reg_i=0.01, optimizer='als')
    print("Starting training ...")
    model.train_model()
    # predict
    test_rmse = model.RMSE(X_test)
    print("Test RMSE: %.4f" % (test_rmse))
