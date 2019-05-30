from __future__ import print_function
import pandas as pd
import numpy as np
import math
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import Ridge # linear regression vs weight decay la norm2
from sklearn import linear_model


# reading user file
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
user = pd.read_csv('./data/ml-100k/u.user', sep='|', names=u_cols,
 encoding='latin-1')
n_users = user.shape[0]
print('Number of user: ', n_users)

# reading rate file
r_cols = ['user_id', 'movie_id', 'rating', 'unix_timestamp']
ratings_base = pd.read_csv('./data/ml-100k/ua.base', sep='\t', names=r_cols, encoding='latin-1')
ratings_test = pd.read_csv('./data/ml-100k/ua.test', sep='\t', names=r_cols, encoding='latin-1')

rate_train = ratings_base.values
rate_test = ratings_test.values
print('Number of train rates: ', rate_train.shape[0])
print('Number of test rates: ', rate_test.shape[0])

#Reading items file:
i_cols = ['movie id', 'movie title' ,'release date','video release date', 'IMDb URL', 'unknown', 'Action', 'Adventure',
 'Animation', 'Children\'s', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

items = pd.read_csv('./data/ml-100k/u.item', sep='|', names=i_cols,
 encoding='latin-1')

n_items = items.shape[0]
print('Number of items: ', n_items)

# vì ta chỉ dựa trên thể loại của phim, nên chỉ quan tâm đến 19 giá trị nhị phân ở cuối mỗi hàng
X0 = items.values
X_train_counts = X0[:,-19:]
# xây dựng feature vector cho mỗi item dựa trên ma trận về thể loại phim và ma trận về TF-IDF
# sau bước này, mỗi hàng của tf-ifd ứng với feature vector của 1 bộ phim

transformer = TfidfTransformer(smooth_idf=True, norm='l2')
tfidf = transformer.fit_transform(X_train_counts.tolist()).toarray()

# với mỗi user, đi tìm bộ phim user đó đã rated và giá trị của rating đó

def get_items_rated_by_user(rate_matrix, user_id):
    """
    in each line of rate_matrix, we have infor: user_id, item_id, rating (scores), time_stamp
    we care about the first three values
    return (item_ids, scores) rated by user user_id
    """
    y = rate_matrix[:, 0] # all user
    # item indices rated by user_id
    # we need to +1 to user_id since in the rate_matrix, id starts from 1 
    # while index in python starts from 0
    ids = np.where( y == user_id + 1)[0]
    item_ids = rate_matrix[ids, 1] -1  # index starts from 0
    scores = rate_matrix[item_ids, 2]
    return (item_ids, scores)

d = tfidf.shape[1] # dimention
W = np.zeros((d, n_users))
b = np.zeros((1, n_users))

for n in range(n_users):
    ids, scores = get_items_rated_by_user(rate_train, n)
    clf = Ridge(alpha=0.01, fit_intercept=True) # alpha la he so cua weight decay
    Xhat = tfidf[ids, :]
    clf.fit(Xhat, scores)
    W[:, n] = clf.coef_
    b[0, n] =clf.intercept_

 # sau khi tính được W và b, thì rating cho mỗi items được dự đoán bằng cách tính
Yhat = tfidf.dot(W) + b   

# example with user_id = 10
n = 10
np.set_printoptions(precision=2) # 2 digits after . 
ids, scores = get_items_rated_by_user(rate_test, n)
Yhat[n, ids]
print('Rated movies ids :', ids )
print('True ratings     :', scores)
print('Predicted ratings:', Yhat[ids, n])

"""
Để đánh giá mô hình tìm được, chúng ta sẽ sử dụng Root Mean Squared Error (RMSE), 
tức căn bậc hai của trung bình cộng bình phương của lỗi. 
Lỗi được tính là hiệu của true rating và predicted rating:
"""
def evaluate(Yhat, rates, W, b):
    se = 0
    cnt = 0
    for n in range(n_users):
        ids, scores_truth = get_items_rated_by_user(rates, n)
        scores_pred = Yhat[ids, n]
        e = scores_truth - scores_pred 
        se += (e*e).sum(axis = 0)
        cnt += e.size 
    return math.sqrt(se/cnt)

print('RMSE for training:', evaluate(Yhat, rate_train, W, b))
print('RMSE for test    :', evaluate(Yhat, rate_test, W, b))
