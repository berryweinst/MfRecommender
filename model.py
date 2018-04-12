import numpy as np
from data import *





class MFModel(object):

    def __init__(self, rates_dict, num_users, num_items, hidden_dim, lr, gamma, epochs):
        """ Params:
        1) ratings - rating matrix (as dict)
        2) hidden_dim  - number of latent dimensions (as integer)
        3) lr - learning rate (as float)
        4) gamma - regulaizer (as float)
        5) opechs - as int
        """

        self.ratings_dict = rates_dict
        self.num_users = num_users
        self.num_items = num_items
        self.K = hidden_dim
        self.lr = lr
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
            self.LearnModelFromDataUsingSGD()
            rmse = self.RMSE()
            if epoch % 5 == 0:
                print("[Epoch %d] : RMSE = %.4f" % (epoch, rmse))


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



    def RMSE(self):
        mod_pred = self.bu[:,np.newaxis] + self.bv[np.newaxis:,] + self.U.dot(self.V.T)
        total_err = 0
        r_cnt = 0
        for u, items in self.ratings_dict.items():
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
    model = MFModel(X_train, users, items, hidden_dim=5, lr=0.005, gamma=0.001, epochs=100)
    print("Starting training ...")
    model.train_model()
    # predict
    model.ratings_dict = X_test
    test_rmse = model.RMSE()
    print("Test RMSE: %.4f" % (test_rmse))
