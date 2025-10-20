import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from metrics_util import *




class SVM:

    def __init__(self):
        self.results = {
            "linear": {"trues": [], "preds": [], "probs": []},
            "poly": {"trues": [], "preds": [], "probs": []},
            "rbf": {"trues": [], "preds": [], "probs": []}
        }

    
    def compute_svm(self, X, Y):
        kf = StratifiedKFold()
        for kernel in self.results:
            svm = SVC(kernel=kernel, probability=True)

            for train, test in kf.split(X, Y):
                X_train = X.iloc[train]
                X_test = X.iloc[test]
                Y_train = Y.iloc[train]
                Y_test = Y.iloc[test]

                # We split our data using 5 K fold
                print(f"Training {kernel} SVM")
                svm.fit(X_train, Y_train)
                # Train on our X and Y train samples

                Y_pred = svm.predict(X_test)
                # Based on our X test, predict some classes

                Y_prob = svm.predict_proba(X_test)
                # based on X test, predict probabilities of certain classes

                


                self.results[kernel]["preds"].append(Y_pred)
                # Append all the predictions into an array
                self.results[kernel]["trues"].append(Y_test)
                # Append the real results into an array
                self.results[kernel]["probs"].append(Y_prob)
                # Append the probabilities of classes into an array



    def compute_svm_metrics(self, labels):

        for kernel in self.results:
            print(f"==================={kernel} Kernel metrics ================")
            compute_all_metrics(self.results[kernel]["trues"], self.results[kernel]["preds"], self.results[kernel]["probs"], labels)





