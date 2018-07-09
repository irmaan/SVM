import numpy as np
from sklearn import metrics
import mnist

from model.sklearn_multiclass import sklearn_multiclass_prediction
from model.self_multiclass import MulticlassSVM

if __name__ == '__main__':

    mndata=mnist

    X_train = mndata.train_images()
    y_train = mndata.train_labels()

    X_test = mndata.test_images()
    y_test = mndata.test_labels()



    print('Training Sklearn One Vs Rest...')
    y_pred_train, y_pred_test = sklearn_multiclass_prediction(
        'ovr', X_train, y_train, X_test)
    print('Sklearn One Vs Rest Accuracy (train):',
          metrics.accuracy_score(y_train, y_pred_train))
    print('Sklearn One Vs Rest Accuracy (test) :',
          metrics.accuracy_score(y_test, y_pred_test))

    print('Training Sklearn One By One...')
    y_pred_train, y_pred_test = sklearn_multiclass_prediction(
        'ovo', X_train, y_train, X_test)
    print('Sklearn One By One Accuracy (train):',
          metrics.accuracy_score(y_train, y_pred_train))
    print('Sklearn One Vs One Accuracy (test) :',
          metrics.accuracy_score(y_test, y_pred_test))


    print('Training self One Vs Rest...')
    self_ovr = MulticlassSVM('ovr')
    self_ovr.fit(X_train, y_train)
    print('Self One Vs Rest Accuracy (train):',
          metrics.accuracy_score(y_train, self_ovr.predict(X_train)))
    print('Self One Vs Rest Accuracy (test) :',
          metrics.accuracy_score(y_test, self_ovr.predict(X_test)))

    print('Training self One By One...')
    self_ovo = MulticlassSVM('ovo')
    self_ovo.fit(X_train, y_train)
    print('Self One By One Accuracy (train):',
          metrics.accuracy_score(y_train, self_ovo.predict(X_train)))
    print('Self One By One Accuracy (test) :',
          metrics.accuracy_score(y_test, self_ovo.predict(X_test)))

