# This will be just for calling our task functions for the assignment
# This is to maintain modularity where each file will serve its own purpose
# data_utils.py: Handle loading, cleaning, splitting the dataset
# metrics_util.py: All evaluation metrics & plotting functions
# knn_model.py: KNN training, prediction, cross-validation
# svm_model.py: SVM (linear, poly, RBF) training & evaluation
# kmeans_model.py: KMeans clustering & visualization
from data_utils import load_data, scale_data
from knn_model import KNN
from svm_model import SVM
from kmeans_model import KMEANS
import sys


def main():
    X, Y, labels = load_data("lncRNA_5_Cancers.csv")
    # We load our data and get out features, classes and unique labels

    X_scaled = scale_data(X)
    # We scale our features so outliers do not have big effects

    #knn = KNN()

    #print("Printing metrics")
    #knn.compute_knn(X_scaled, Y)

    #knn.compute_knn_metrics(labels)

    #svm = SVM()

    #svm.compute_svm(X_scaled, Y)

    #svm.compute_svm_metrics(labels)

    kmeans = KMEANS()

    kmeans.compute_kmeans(X_scaled)







    


if __name__ =="__main__":
        main()