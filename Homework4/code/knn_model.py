# This will be the file that builds the KNN model, trains and fits

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from metrics_util import calculate_metrics, plot_confusion_matrix, plot_roc_curve, plot_pr_curve

class KNN:

    def __init__(self):
        self.knn = KNeighborsClassifier(n_neighbors=5)
        self.Y_preds = []
        self.Y_trues = []
        self.Y_probs = []


    def compute_knn(self, X, Y):
        kf = StratifiedKFold()

        for train, test in kf.split(X, Y):
            X_train = X.iloc[train]
            X_test = X.iloc[test]
            Y_train = Y.iloc[train]
            Y_test = Y.iloc[test]

            # We split our data using 5 K fold

            self.knn.fit(X_train, Y_train)
            # Train on our X and Y train samples

            Y_pred = self.knn.predict(X_test)
            # Based on our X test, predict some classes

            Y_prob = self.knn.predict_proba(X_test)
            # based on X test, predict probabilities of certain classes

            self.Y_preds.append(Y_pred)
            # Append all the predictions into an array
            self.Y_trues.append(Y_test)
            # Append the real results into an array
            self.Y_probs.append(Y_prob)
            # Append the probabilities of classes into an array



    def compute_knn_metrics(self, labels):

        knn_cr = calculate_metrics(self.Y_trues, self.Y_preds, labels=labels)

        knn_cf = plot_confusion_matrix(self.Y_trues, self.Y_preds, labels=labels)

        plot_roc_curve(self.Y_trues, self.Y_probs, labels)

        plot_pr_curve(self.Y_trues, self.Y_probs, labels)

        return knn_cr, knn_cf







