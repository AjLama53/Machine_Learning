# This will be for calculating all the metrics for both models

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.preprocessing import label_binarize
import numpy as np
import matplotlib.pyplot as plt


def calculate_metrics(Y_true, Y_pred, labels):
    cr = classification_report(Y_true, Y_pred, labels=labels)


    return cr

    


def plot_confusion_matrix(Y_true, Y_pred, labels):

    cf = confusion_matrix(Y_true, Y_pred, labels=labels)

    return cf

def binarize_classes(Y_trues, Y_probs, labels):

    bin_trues = label_binarize(Y_trues, classes=labels)

    return bin_trues, Y_probs




def plot_roc_curve(Y_trues, Y_probs, labels):
    bin_trues, Y_all_probs = binarize_classes(Y_trues, Y_probs, labels)

    display = RocCurveDisplay.from_predictions(bin_trues.ravel(), Y_all_probs.ravel())

    display.plot()
    plt.show()





def plot_pr_curve(Y_trues, Y_probs, labels):
    bin_trues, Y_all_probs = binarize_classes(Y_trues, Y_probs, labels)

    display = PrecisionRecallDisplay.from_predictions(bin_trues.ravel(), Y_all_probs.ravel())

    display.plot()
    plt.show()


def compute_all_metrics(Y_trues, Y_preds, Y_probs, labels):

        Y_trues = np.concatenate(Y_trues)
        Y_preds = np.concatenate(Y_preds)
        Y_probs = np.concatenate(Y_probs)

        knn_cr = calculate_metrics(Y_trues, Y_preds, labels=labels)

        knn_cf = plot_confusion_matrix(Y_trues, Y_preds, labels=labels)

        plot_roc_curve(Y_trues, Y_probs, labels)

        plot_pr_curve(Y_trues, Y_probs, labels)

        print("=================Classification Report=================")
        print(knn_cr)

        print("=================Confusion Matrix======================")
        print(labels)
        print(knn_cf)

