class DumpAnomalyData:
    def __init__(self, t_f_dt, t_f_knn, t_f_dt_val, t_f_knn_val, t_l, t_l_val, test_f_dt, test_f_knn, test_l,
                 feature_set_min_max_knn, predicted_dt_val, predicted_knn_val, predicted_dt_test, predicted_knn_test):
        self.train_features_dt = t_f_dt
        self.train_features_knn = t_f_knn
        self.train_features_dt_val = t_f_dt_val
        self.train_features_knn_val = t_f_knn_val
        self.train_labels = t_l
        self.train_labels_val = t_l_val

        self.test_features_dt = test_f_dt
        self.test_features_knn = test_f_knn
        self.test_labels = test_l

        self.feature_set_min_max_knn = feature_set_min_max_knn
        self.predicted_dt_val = predicted_dt_val
        self.predicted_knn_val = predicted_knn_val
        self.predicted_dt_test = predicted_dt_test
        self.predicted_knn_test = predicted_knn_test
