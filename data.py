import random
import math
import numpy as np



def create_user_item_map(ratings_path, movies_path, percent=0.8, sort_ratings=True):
    users_map = dict()
    items_map = dict()
    user_id = 0
    item_id = 0
    ratings_data = dict()

    rating_cnt = 0
    with open(ratings_path, 'r') as ratings:
        for line in ratings:
            line_split = line.split('::')
            user = int(line_split[0])
            item = int(line_split[1])
            rating = float(line_split[2])
            timestamp = int(line_split[3])
            if user not in users_map:
                users_map[user] = user_id
                user_id += 1
            if item not in items_map:
                items_map[item] = item_id
                item_id += 1
            rating_cnt += 1
            if users_map[user] not in ratings_data:
                ratings_data[users_map[user]] = []
            ratings_data[users_map[user]].append((items_map[item], rating, timestamp))

    num_users = user_id
    num_items = item_id
    X_train = dict()
    X_test = dict()
    training_items = set()
    for user_id in ratings_data.keys():
        if len(ratings_data[user_id]) < 2:
            continue
        if sort_ratings: # sort ratings by their timestamp
            ratings = sorted(ratings_data[user_id], key=lambda x: x[2])
        else:
            ratings = [rate for rate in ratings_data[user_id]]
            ratings = random.sample(ratings, len(ratings))
        t_idx = math.floor(percent * len(ratings))
        X_train[user_id] = ratings[:t_idx]
        for rating_item in ratings[:t_idx]:
            training_items.add(rating_item[0])  # keep list of items in the training set
        X_test[user_id] = ratings[t_idx:]
    # Better to remove items that are not appear in the training set
    for user_id, user_rates in X_test.items():
        X_test[user_id] = [rating for rating in user_rates if rating[0] in training_items]
    # print(user_id, item_id)

    # train_mat = np.zeros((user_id+1, item_id+1))
    # for user, item in X_train.items():
    #     for movie, rate, _ in item:
    #         train_mat[user][movie] = rate
    #
    # test_mat = np.zeros((user_id+1, item_id+1))
    # for user, item in X_test.items():
    #     for movie, rate, _ in item:
    #         test_mat[user][movie] = rate


    return X_train, X_test, num_users, num_items


# if __name__ == '__main__':
#     X_train, X_test = create_user_item_map('../ml-1m/ratings.dat', '', percent=0.8, sort_ratings=True)