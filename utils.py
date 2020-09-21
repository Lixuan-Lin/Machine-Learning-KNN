import numpy as np
from knn import KNN

############################################################################
# DO NOT MODIFY ABOVE CODES
############################################################################


# TODO: implement F1 score
def f1_score(real_labels, predicted_labels):
    """
    Information on F1 score - https://en.wikipedia.org/wiki/F1_score
    :param real_labels: List[int]
    :param predicted_labels: List[int]
    :return: float
    """
    assert len(real_labels) == len(predicted_labels)

    TP = np.sum(np.multiply(real_labels, predicted_labels))
    FP = np.sum(np.multiply(np.logical_not(real_labels), predicted_labels))
    TN = np.sum(np.multiply(np.logical_not(real_labels),
                            np.logical_not(predicted_labels)))
    FN = np.sum(np.multiply(real_labels, np.logical_not(predicted_labels)))

    if ((TP+FP) == 0 or (TP+FN) == 0):
        F1 = 0
    else:
        Precision = TP / (TP+FP)
        Recall = TP / (TP + FN)
        if ((Recall+Precision) == 0):
            F1 = 0
        else:
            F1 = 2*Precision*Recall / (Recall+Precision)

    return F1
    raise NotImplementedError


class Distances:
    @staticmethod
    # TODO
    def canberra_distance(point1, point2):
        denominator = np.abs(np.asarray(point1))+np.abs(np.asarray(point2))
        for i in range(denominator.shape[0]):
            if (denominator[i] == 0):
                denominator[i] = 1

        quotient = np.abs(np.asarray(point1)-np.asarray(point2)) / denominator
        cd = np.sum(quotient)
        return cd
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        raise NotImplementedError

    @staticmethod
    # TODO
    def minkowski_distance(point1, point2):
        power = np.power(np.abs(np.asarray(point1)-np.asarray(point2)), 3)
        total = np.sum(power)
        md = total ** (1/3)
        return md
        """
        Minkowski distance is the generalized version of Euclidean Distance
        It is also know as L-p norm (where p>=1) that you have studied in class
        For our assignment we need to take p=3
        Information on Minkowski distance - https://en.wikipedia.org/wiki/Minkowski_distance
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        raise NotImplementedError

    @staticmethod
    # TODO
    def euclidean_distance(point1, point2):
        sub = np.subtract(point1, point2)
        dot = np.dot(sub, sub)
        ed = np.sqrt(dot)
        return ed
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        raise NotImplementedError

    @staticmethod
    # TODO
    def inner_product_distance(point1, point2):
        ipd = np.dot(point1, point2)
        return ipd
        """
        :param point1: List[float]
        :param point2: List[float]
        :return: float
        """
        raise NotImplementedError

    @staticmethod
    # TODO
    def cosine_similarity_distance(point1, point2):
        dot = np.dot(point1, point2)
        norm1 = np.linalg.norm(point1)
        norm2 = np.linalg.norm(point2)
        csd = 1 - (dot / (norm1*norm2))
        return csd
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        raise NotImplementedError

    @staticmethod
    # TODO
    def gaussian_kernel_distance(point1, point2):
        sub = np.subtract(point1, point2)
        dot = np.dot(sub, sub)
        gkd = -1*np.exp(dot / (-2))
        return gkd
        """
       :param point1: List[float]
       :param point2: List[float]
       :return: float
       """
        raise NotImplementedError


class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # TODO: find parameters with the best f1 score on validation dataset
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        """
        In this part, you should try different distance function you implemented in part 1.1, and find the best k.
        Use k range from 1 to 30 and increment by 2. Use f1-score to compare different models.

        :param distance_funcs: dictionary of distance functions you must use to calculate the distance.
            Make sure you loop over all distance functions for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val:  List[List[int]] Validation data set will be used on your KNN predict function to produce
            predicted labels and tune k and distance function.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_function and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function and self.best_model respectively.
        NOTE: self.best_scaler will be None

        NOTE: When there is a tie, choose model based on the following priorities:
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance fuction, choose model which has a less k.
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_model = None

        best_f1 = -1
        for k in range(1, 30, 2):
            for name, function in distance_funcs.items():
                model = KNN(k, function)
                model.train(x_train, y_train)
                prediction = model.predict(x_val)
                f1 = f1_score(y_val, prediction)
                if (best_f1 < f1):
                    best_f1 = f1
                    self.best_k = k
                    self.best_distance_function = name
                    self.best_model = model
        return self

        raise NotImplementedError

    # TODO: find parameters with the best f1 score on validation dataset, with normalized data
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        """
        This part is similar to Part 1.3 except that before passing your training and validation data to KNN model to
        tune k and disrance function, you need to create the normalized data using these two scalers to transform your
        data, both training and validation. Again, we will use f1-score to compare different models.
        Here we have 3 hyperparameters i.e. k, distance_function and scaler.

        :param distance_funcs: dictionary of distance funtions you use to calculate the distance. Make sure you
            loop over all distance function for each data point and each k value.
            You can refer to test.py file to see the format in which these functions will be
            passed by the grading script
        :param scaling_classes: dictionary of scalers you will use to normalized your data.
        Refer to test.py file to check the format.
        :param x_train: List[List[int]] training data set to train your KNN model
        :param y_train: List[int] train labels to train your KNN model
        :param x_val: List[List[int]] validation data set you will use on your KNN predict function to produce predicted
            labels and tune your k, distance function and scaler.
        :param y_val: List[int] validation labels

        Find(tune) best k, distance_funtion, scaler and model (an instance of KNN) and assign to self.best_k,
        self.best_distance_function, self.best_scaler and self.best_model respectively

        NOTE: When there is a tie, choose model based on the following priorities:
        For normalization, [min_max_scale > normalize];
        Then check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
        If they have same distance function, choose model which has a less k.
        """

        # You need to assign the final values to these variables
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

        best_f1 = -1
        for k in range(1, 30, 2):
            for distance_name, distance_func in distance_funcs.items():
                for scaling_name, scaling_class in scaling_classes.items():
                    scaler = scaling_class()
                    normalized_x_train = scaler(x_train)
                    normalized_x_val = scaler(x_val)

                    model = KNN(k, distance_func)
                    model.train(normalized_x_train, y_train)
                    prediction = model.predict(normalized_x_val)
                    f1 = f1_score(y_val, prediction)
                    if (best_f1 < f1):
                        best_f1 = f1
                        self.best_k = k
                        self.best_distance_function = distance_name
                        self.best_scaler = scaling_name
                        self.best_model = model
        return self

        raise NotImplementedError


class NormalizationScaler:
    def __init__(self):
        pass

    # TODO: normalize data
    def __call__(self, features):

        normalized_features = []
        for i in range(len(features)):
            denominator = np.sqrt(np.dot(features[i], features[i]))
            if (denominator == 0):
                normalized = len(features[i]) * [float(0)]
                normalized_features.append(normalized)
            else:
                normalized = (features[i] / denominator).tolist()
                normalized_features.append(normalized)
        return normalized_features
        """
        Normalize features for every sample

        Example
        features = [[3, 4], [1, -1], [0, 0]]
        return [[0.6, 0.8], [0.707107, -0.707107], [0, 0]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        raise NotImplementedError


class MinMaxScaler:
    """
    Please follow this link to know more about min max scaling
    https://en.wikipedia.org/wiki/Feature_scaling
    You should keep some states inside the object.
    You can assume that the parameter of the first __call__
    will be the training set.

    Hints:
        1. Use a variable to check for first __call__ and only compute
            and store min/max in that case.

    Note:
        1. You may assume the parameters are valid when __call__
            is being called the first time (you can find min and max).

    Example:
        train_features = [[0, 10], [2, 0]]
        test_features = [[20, 1]]

        scaler1 = MinMaxScale()
        train_features_scaled = scaler1(train_features)
        # train_features_scaled should be equal to [[0, 1], [1, 0]]

        test_features_scaled = scaler1(test_features)
        # test_features_scaled should be equal to [[10, 0.1]]

        new_scaler = MinMaxScale() # creating a new scaler
        _ = new_scaler([[1, 1], [0, 0]]) # new trainfeatures
        test_features_scaled = new_scaler(test_features)
        # now test_features_scaled should be [[20, 1]]

    """

    def __init__(self):
        self.first_call = True
        self.max = []
        self.min = []
        pass

    def __call__(self, features):
        if (self.first_call == True):
            self.first_call = False
            self.max = np.max(features, axis=0)
            self.min = np.min(features, axis=0)

        difference = self.max - self.min

        normalized_features = []
        # for i in range(len(features)):
        #     normalized_feature = np.divide(np.subtract(
        #         features[i], self.min), difference, dtype=float)
        #     normalized_features.append(normalized_feature)
        for i in range(len(features)):
            nf = []
            for j in range(len(features[i])):
                if (difference[j] == 0):
                    nf.append(0)
                else:
                    nf.append(np.divide(np.subtract(
                        features[i][j], self.min[j]), difference[j], dtype=float))

            normalized_features.append(nf)

        return normalized_features
        """
        normalize the feature vector for each sample . For example,
        if the input features = [[2, -1], [-1, 5], [0, 0]],
        the output should be [[1, 0], [0, 1], [0.333333, 0.16667]]

        :param features: List[List[float]]
        :return: List[List[float]]
        """
        raise NotImplementedError
