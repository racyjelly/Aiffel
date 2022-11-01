# 1. Libraries import
from sklearn.model_selection import train_test_split
import numpy as np

def Preprocess(load_data, split=True, test_size=0.2, random_state=5):
    load_data = load_data
    train_data = load_data.data
    label_data = load_data.target
    print("Describe:", load_data.DESCR)
    print("Featrue names:", load_data.feature_names)
    print("Feature number:", len(load_data.feature_names))
    print("Label names:", load_data.target_names)
    print("Label:", np.unique(label_data))

    if split==True:
        X_train, X_test, Y_train, Y_test= train_test_split(train_data,
                                                            label_data,
                                                            test_size=test_size,
                                                            random_state=random_state)
        return X_train, X_test, Y_train, Y_test