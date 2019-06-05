from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class MF(object):
    def __init__(self, Y_data, K, lam = 0.1, Xinit = None, Winit = None,
            learning_rate = 0.5, max_iter = 1000, print_every = 100, user_based = 1):
            self.Y_raw_data = Y_data
            self.K = K
            self.lam = lam # hệ số cho regularization ( weight decay)
            self.learning_rate = learning_rate # lr for gradient descent
            self.max_iter = max_iter # maximum number of iterations
            self.print_every = print_every # print result after print_every iterations
            self.user_based = user_based # user-based or item-based
            self.n_users = int(np.max(Y_data[:, 0])) + 1 # number of users, items and ratings. Remember to add 1 since id start from 0
            self.n_items = int(np.max(Y_data[:, 1])) + 1
            self.n_ratings = Y_data.shape[0]

            if Xinit is None: # new
                self.X = np.random.randn(self.n_items, K)
            else: # or from saved data
                self.X = Xinit
            
            if Winit is None:
                self.W = np.random.randn(K, self.n_users)
            else:
                self.W = Winit
            self.Y_data_n = self.Y_raw_data.copy() # normalized data, update later in normalized_Y function

    
    def normalized_Y(self):
        if self.user_based:
            user_col = 0
            item_col = 1
            n_objects = self.n_users
        else:         # if we want to normalize based on item, just switch first two columns of data
            user_col = 1
            item_col = 0
            n_objects = self.n_items


        users = self.Y_raw_data[:, user_col]
        self.mu = np.zeros((n_objects, ))
        for n in range(n_objects):
            ids = np.where(users == n)[0].astype(np.int32)  # row indices of rating done by user n, since indices need to be integers, we need to convert
            item_ids = self.Y_data_n[ids, item_col] # indices of all rating associated with user n
            ratings = self.Y_data_n[ids, 2] # and the corresponding ratings
            m = np.mean(ratings) # take mean
            if np.isnan(m):
                m = 0 # to avoid empty array and nan value
            self.mu[n] = m
            self.Y_data_n[ids, 2] = ratings - self.mu[n] # nomalize

    def loss(self):
        L = 0
        for i in range(self.n_ratings):
            n, m, rate = int(self.Y_data_n[i, 0]), int(self.Y_data_n[i, 1]), int(self.Y_data_n[i, 2])
            L += 0.5*(rate - self.X[m, :].dot(self.W[:, n]))**2
        L /= self.n_ratings  # take average (lấy trung bình)
        L += 0.5*self.lam*(np.linalg.norm(self.X, 'fro') + np.linalg.norm(self.W, 'fro')) # regularization
        return L

    
    '''
    Xác định các items được đánh giá bởi 1 user và các ratings tương ứng
    '''
    def get_items_rated_by_user(self, user_id):
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        item_ids = self.Y_data_n[ids, 1].astype(np.int32)
        ratings = self.Y_data_n[ids, 2]
        return (item_ids, ratings)


    '''
    Xác định các users  đánh giá  1 item và các ratings tương ứng
    '''
    def get_users_who_rate_item(self, item_id):
        ids = np.where(self.Y_data_n[:, 1] == item_id)[0]
        user_ids = self.Y_data_n[ids, 0].astype(np.int32)
        ratings = self.Y_data_n[user_ids, 2]
        return (user_ids, ratings)

    
    def update_X(self):
        for m in range(self.n_items):
            user_ids, ratings = self.get_users_who_rate_item(m)
            Wm = self.W[:, user_ids]
            grad_xm = -(ratings - self.X[m, :].dot(Wm)).dot(Wm.T)/self.n_ratings + self.lam*self.X[m, :] # gradient
            self.X[m, :] -= self.learning_rate*grad_xm.reshape((self.K, ))

    def update_W(self):
        for n in range(self.n_users):
            item_ids, ratings = self.get_items_rated_by_user(n)
            Xn = self.X[item_ids, :]
            grad_wn = -Xn.T.dot(ratings - Xn.dot(self.W[:, n]))/self.n_ratings + self.lam*self.W[:, n]
            self.W[:, n] -= self.learning_rate*grad_wn.reshape((self.K, ))

    # predict
    def pred(self, u, i):
        u = int(u)
        i = int(i)
        if self.user_based:
            bias = self.mu[u]
        else:
            bias = self.mu[i]

        pred = self.X[i, :].dot(self.W[:, u]) + bias
#        if pred <0:
#            return 0
#        if pred > 5:
#            return 5
#        return pred
        return max(0, min(5, pred))

    # predict ratings one user give all unrated items
    def pred_for_user(self, user_id):
        ids = np.where(self.Y_data_n[:, 0] == user_id)[0]
        items_rate_by_u = self.Y_data_n[ids, 1].tolist()

        y_pred = self.X.dot(self.W[:, user_id]) + self.mu[user_id]
        predicted_ratings = []
        for i in range(self.n_items):
            if i not in items_rate_by_u:
                predicted_ratings.append((i, y_pred[i]))
        return predicted_ratings
 



    # đánh giá kết quả
    def evaluate_RMSE(self, rate_test):
        n_tests = rate_test.shape[0]
        SE = 0 # squared error
        for n in range(n_tests):
            pred = self.pred(rate_test[n, 0], rate_test[n, 1])
            SE += (pred - rate_test[n, 2])**2
        RMSE = np.sqrt(SE/n_tests)
        return RMSE

    def MF_fit(self):
        self.normalized_Y()
        for it in range(self.max_iter):
            self.update_X()
            self.update_W()
            if (it + 1) % self.print_every == 0:
                rmse_train = self.evaluate_RMSE(self.Y_raw_data)
                print('iter =', it+1, ', loss =', self.loss(), ', RMSE train= ', rmse_train)


r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']

ratings_base = pd.read_csv('./data/ml-100k/ub.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('./data/ml-100k/ub.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.values
rate_test = ratings_test.values

# indices start from 0

rate_train[:, :2] -= 1
rate_test[:, :2] -=1

'''
user_base
'''
rs = MF(rate_train, K = 10, lam = .1, print_every = 10,
        learning_rate = 0.75, max_iter = 100, user_based = 1)

rs.MF_fit()

RMSE = rs.evaluate_RMSE(rate_test)

print('user-based MF, RMSE =', RMSE)

'''
item_base
'''
rs = MF(rate_train, K = 10, lam = .1, print_every = 10,
        learning_rate = 0.75, max_iter = 100, user_based = 0)

rs.MF_fit()

RMSE = rs.evaluate_RMSE(rate_test)
print('item-based MF, RMSE =', RMSE)

'''
item_base without regulariztion ( lamda = 0)
'''
rs = MF(rate_train, K = 10, lam = 0, print_every = 10,
        learning_rate = 0.75, max_iter = 100, user_based = 0)

rs.MF_fit()

RMSE = rs.evaluate_RMSE(rate_test)
print('item-based MF without regulariztion, RMSE =', RMSE)


    








        
            

            





