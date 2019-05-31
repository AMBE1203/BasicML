from __future__ import print_function
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

class CF(object):
    def __init__(self, Y_data, k, dist_func = cosine_similarity, uucF = 1):
        self.uuCF = uucF # user-user (1) or item-item (0) CF
        self.Y_data = Y_data if uucF else Y_data[:, [1, 0, 2]] # đối với item-item đổi vị trí của 2 cột đầu tiên là thành ma trận chuyển vị (item rate user)
        self.k = k # number of neighbor points
        self.dist_func = dist_func
        self.Ybar_data = None
        # number of users and items. Remember to add 1 since id starts from 0
        self.n_users = int(np.max(self.Y_data[:, 0])) + 1
        self.n_items = int(np.max(self.Y_data[:, 1])) + 1

    def add(self, new_data):
        """
        Update Y_data matrix when new ratings come.
        For simplicity, suppose that there is no new user or item.
        """
        self.Y_data = np.concatenate((self.Y_data, new_data), axis = 0)

# tính toán normalized matrix and similarity matrix

    def normalize_Y(self):
        users = self.Y_data[:, 0] # all user - first col of the Y_data
        self.Ybar_data = self.Y_data.copy()
        self.mu = np.zeros((self.n_users, ))
        for n in range(self.n_users):
            # row indices of rating done by user n
            # since indices need to be integer, we need to convert
            ids = np.where(users == n)[0].astype(np.int32)
            # indices of all ratings associated with user n
            item_ids = self.Y_data[ids, 1]
            # and the corresponding (tương ứng) rating
            ratings = self.Y_data[ids, 2]
            # take mean
            m = np.mean(ratings)
            if np.isnan(n):
                m = 0 # to avoid empty array and na value
            # normalize
            self.Y_data[ids, 2] = ratings - self.mu[n]
        ################################################
        # form the rating matrix as a sparse matrix. Sparsity is important 
        # for both memory and computing efficiency. For example, if #user = 1M, 
        # #item = 100k, then shape of the rating matrix would be (100k, 1M), 
        # you may not have enough memory to store this. Then, instead, we store 
        # nonzeros only, and, of course, their locations.

        self.Ybar = sparse.coo_matrix((self.Ybar_data[:, 2],
            (self.Ybar_data[:, 1], self.Ybar_data[:, 0])), (self.n_items, self.n_users))
        self.Ybar = self.Ybar.tocsr()

    def similarity(self):
        self.S = self.dist_func(self.Ybar.T, self.Ybar.T)

    # Thực hiện 2 hàm phía trên nếu có thêm dữ liệu

    def refresh(self):
        """
        Normalize data and calculate similarity matrix again (after
        some few ratings added)
        """
        self.normalize_Y()
        self.similarity() 

    def fit(self):
        self.refresh()


    """
    Hàm __pred là hàm dự đoán rating mà user u cho item i cho trường hợp User-user CF. 
    Vì trong trường hợp Item-item CF, chúng ta cần hiểu ngược lại nên hàm pred sẽ thực hiện đổi vị trí hai biến của __pred. 
    Để cho API được đơn giản, tôi cho __pred là một phương thức private, 
    chỉ được gọi trong class CF; pred là một phương thức public, 
    thứ tự của biến đầu vào luôn là (user, item), bất kể phương pháp sử dụng là User-user CF hay Item-item CF.
    """

    def __pred(self, u, i, normalized = 1):
        """
        predict the rating of user u for item i (nomalized)
        if you need the un
        """

        # step 1: find all users who rated i
        ids = np.where(self.Y_data[:, 1] == i)[0].astype(np.int32)
        # step 2
        users_rated_i = (self.Y_data[ids, 0]).astype(np.int32)
        # step 3: find similarity btw the current user and others who already rated i
        sim = self.S[u, users_rated_i]
        # step 4: find the k most similarity users
        a = np.argsort(sim)[-self.k:]
        # and the corresponding similarity levels
        nearest_s = sim[a]
        # how did each of 'near' users rated item i
        r = self.Ybar[i, users_rated_i[a]]
        if normalized:
            # add a small number, for instance, 1e-8, to avoid dividing by 0
            return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8)

        return (r*nearest_s)[0]/(np.abs(nearest_s).sum() + 1e-8) + self.mu[u]

    def pred(self, u, i, normalized = 1):
        """
        predict the rating of user u for item i (nomalized)
        if you need the un
        """
        if self.uuCF:
            return self.__pred(u, i, normalized)
        return self.__pred(i, u, normalized)

    """
    Tìm tất cả các items nên được gợi ý cho user u trong trường hợp User-user CF (uucF = 1), 
    hoặc tìm tất cả các users có khả năng thích item u trong trường hợp Item-item CF (uucF = 0)
    """

    def recommend(self, u, normalized = 1):
        ids = np.where(self.Y_data[:, 0] == u)[0]
        items_rated_by_u = self.Y_data[ids, 1].tolist()
        recommended_items = []
        for i in range(self.n_items):
            if i not in items_rated_by_u:
                rating = self.__pred(u, i)
                if rating > 0:
                    recommended_items.append(i)
        return recommended_items

    # print result
    def print_recommendation(self):
        print('Recommendation: ')
        for u in range(self.n_users):
            recommended_items = self.recommend(u)
            if self.uuCF:
                print('Recommend item(s) ', recommended_items, 'to user', u)
            else:
                print('Recommend item ', u, 'to user(s)', recommended_items)

"""
demo with uucf
"""
# data file
r_cols = ['user_id', 'item_id', 'rating']
ratings = pd.read_csv('../ex.dat', sep = ' ', names=r_cols, encoding='latin-1')
Y_data = ratings.values

rs = CF(Y_data, k = 2, uuCF = 1)
rs.fit()

rs.print_recommendation()
        










