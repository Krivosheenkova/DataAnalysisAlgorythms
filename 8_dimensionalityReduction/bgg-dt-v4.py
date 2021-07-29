import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns


def accuracy_metric(y, y_pred):
    pass


def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

def r2_score(y_pred, y):
    sst = np.sum((y - y.mean()) ** 2)
    ssr = np.sum((y_pred - y) ** 2)
    r2 = 1 - (ssr / sst)
    return r2

data = pd.read_csv('tutors-expected-math-exam-results/train.csv')
test_data = pd.read_csv('tutors-expected-math-exam-results/test.csv')
# удаление колонки Id
data = data.drop(['Id'], axis=1)
test_data = test_data.drop(['Id'], axis=1)
# удаление бесполезных признаков, простой отбор
categ_features = ['chemistry', 'english', 'geography', 'history']
data.drop(categ_features,axis=1,inplace=True)
# оптимизация использования памяти
reduce_mem_usage(data)
# инициализация X и y
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=41)

########################################################################
#                                                                      #
#                         DECISION TREE REGRESSOR                      #
#                                                                      #
########################################################################

class Node():

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red

        # for leaf node
        self.value = value


class DecisionTreeRegressor:

    def __init__(self, min_samples_split=2, max_depth=2, min_gain=0.1):
        # initialize the root of the tree
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.min_gain = min_gain

        self.value=None

    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''

        X, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if len(list(best_split.items())) != 0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["var_red"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(y)
        self.value = leaf_value
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''

        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_var_red = self.variance_reduction(y, left_y, right_y)

                    # if curr_var_red < self.min_gain:
                    #     continue
                    # update the best split if needed
                    if curr_var_red > max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red

        # return best split
        return best_split

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''

        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])

        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])

        return dataset_left, dataset_right

    def variance_reduction(self, parent, l_child, r_child):
        ''' function to compute variance reduction '''

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction

    def calculate_leaf_value(self, y):
        ''' function to compute leaf node '''

        val = np.mean(y)
        return val

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, y):
        ''' function to train the tree '''

        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset)

    def make_prediction(self, x, tree):
        ''' function to predict new dataset '''

        if tree.value:
            return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def predict(self, X):
        ''' function to predict a single data point '''
        print(self.root)
        try:

            prediction = self.make_prediction(X, self.root)
            return prediction
        except Exception:
            preditions = [self.make_prediction(x, self.root) for x in X]
            return preditions


########################################################################
#                                                                      #
#                              RFREGRE                                 #
#                                                                      #
########################################################################
class RandomForestClassifier:
    def __init__(self, *args, n_trees, random_state=None, oob_metric=False, **kwargs):
        self.n_trees = n_trees
        self.args = args
        self.kwargs = kwargs

        self.forest = []
        self.oob_metric = oob_metric
        if self.oob_metric:
            self.oob_accuracy = None

        if random_state:
            np.random.seed(random_state)

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        preds = []
        for _ in range(y.shape[0]):
            preds.append([])
        bootstraps = self.get_bootstrap_indexes(X, y, n_trees=self.n_trees)
        for obs_indexes, feat_indexes in bootstraps:
            tree = DecisionTreeRegressor(*self.args, **self.kwargs)
            tree.fit(X[obs_indexes][:, feat_indexes], y[obs_indexes])
            self.forest.append(tree)

        if self.oob_metric:
            oob_indexes = (i for i in range(X.shape[0]) if i not in obs_indexes)
            for i in oob_indexes:
                preds[i].append(tree.predict(X[i]))

            oob_preds = []
            y_copy = np.copy(y)
            for obj_preds in preds:
                avg_pred = max(set(obj_preds), key=obj_preds.mean) if len(obj_preds) > 0 else None
                oob_preds.append(avg_pred)

            oob_indexes = [i for i, value in enumerate(oob_preds) if value is not None]
            self.oob_accuracy = r2_score(np.array(oob_preds)[oob_indexes], y[oob_indexes])

    def get_bootstrap_indexes(self, X, y, *, n_trees, subsample=True):
        len_subsample = int(X.shape[1] / 3)
        for i in range(n_trees):
            n_obs, n_feat = X.shape
            obs_indexes = np.random.choice(np.arange(n_obs), size=n_obs, replace=True)
            feat_indexes = np.arange(n_feat)
            if subsample:
                feat_indexes = np.random.choice(feat_indexes, size=len_subsample, replace=True)
            yield obs_indexes, feat_indexes

    def predict(self, X):
        X = np.array(X)
        y_pred = []

        for predictions in zip(*[tree.predict(X) for tree in self.forest]):
            y_pred.append(np.mean(predictions))
        return y_pred

    # def predict(self, X):
    #     if X.ndim == 0:
    #         X = np.array([X])
    #     preds = np.array(
    #         [estimator.predict(X[features]) for estimator, features in zip(self.estimators, self.features)])
    #     return preds.mean()


rf = RandomForestClassifier(n_trees=10, max_depth=5, random_state=123, oob_metric=False)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

r2_score(y_pred, y_test)
# regressor = DecisionTreeRegressor(min_samples_split=3, max_depth=5, min_gain=0.1)
# regressor.fit(X_train, y_train)
#
# y_pred = regressor.predict(X_test)
# # оценка
# r2_score(y_pred, y_test.T[0])
# y_test_pred = regressor.predict(np.array(test_data))
#
# d = {'Id': np.arange(0 + 10000, 20000), 'mean_exam_points': np.array(y_test_pred)}
# y_test_pred = pd.DataFrame(data=d)
#
# # запись прогноза в файл
# y_test_pred.to_csv('./submission-v4.csv',index=False)