import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, RocCurveDisplay, PrecisionRecallDisplay





def task_two(model, input, output):
    skf = StratifiedKFold(n_splits=5)
    # Creates a stratified k fold object and assigns it to a variable
    # We use this because it preserves the sizes of each train/test split

    temp = 1
    # Variable to be able to get the confusion matrix for one fold

    scaler = StandardScaler()
    # A scaler to scale our features to not have outliers affect the data greatly

    one_fold_cf = None
    labels = None
    # Placeholder for the confusion matrix so it can be used outside the for loop scope

    Y_tests = []
    Y_preds = []
    Y_scores = []
    # Create arrays to holds all the test values and the predictions, that way 5 fold cf can use this dont create the confusion matrix


    for train, test in skf.split(input, output):
        # Here skf.split(input, output) divides the data into 5 folds
        # This then returns two arrays one with train indicies and another with test indices for each foldSo 
        # Therefore if I call input[train] it should give me the indcies from the features that is using to train, same with test

        X_train = input.iloc[train]
        Y_train = output.iloc[train]
        X_test = input.iloc[test]
        Y_test = output.iloc[test]
        # Now these variables have all the indices that are associated with the proper train test data

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        # This finds the scale in which the data should be at

        model.fit(X_train_scaled, Y_train)
        # Trains our model based on x_train and y_train

        Y_pred = model.predict(X_test_scaled)
        # Now we tell our model to predict based on X_test
        Y_score = model.predict_proba(X_test_scaled)
        # We predict the probabilities of each label based on X_test

        Y_tests.append(Y_test)
        Y_preds.append(Y_pred)
        Y_scores.append(Y_score)
        # We append all the different tests from the folds and the different predictions from said folds to be used in the final confusion matrix

        if temp > 0:
            # On our first fold we get the confusion matrix

            labels = sorted(output.unique())
            # We get the unique labels from the class and sort them to ensure the exact same runs every time
            one_fold_cf = confusion_matrix(Y_test, Y_pred, labels=labels)
            # Create our confusion matrix based on Y_test and Y_pred on only the first fold
            temp -= 1
            # Decrease temp that way we dont get the confusion matrix for every fold

    Y_true = np.concatenate(Y_tests)
    Y_all_preds = np.concatenate(Y_preds)
    Y_all_scores = np.concatenate(Y_scores)
    # We have to concatenate the arrays because it is a bunch of vectors in vectors and the confusion matrix function takes in a 1d vector
    # We need to concatenate all the values into one normal vector
        
    five_fold_cf = confusion_matrix(Y_true, Y_all_preds, labels=labels)
    # Produce the 5 fold confusion matrix

    cr = classification_report(Y_true, Y_all_preds, labels=labels)
    # Create the classification report based on the predictions

    precision = precision_score(Y_true, Y_all_preds, average='micro')
    recall = recall_score(Y_true, Y_all_preds, average='micro')
    f1 = f1_score(Y_true, Y_all_preds, average='micro')

    micro_avg = f"   micro avg       {precision:.2f}      {recall:.2f}      {f1:.2f}"
    # Have to get the micro your self since it is not included in the classification report

    lb_classes = label_binarize(Y_true, classes=labels)
    # This binarizes our multi class labels


    roc_display = RocCurveDisplay.from_predictions(lb_classes.ravel(), Y_all_scores.ravel())
    # We flatten out the arrays inside of the two parameters and call the function

    pr_display = PrecisionRecallDisplay.from_predictions(lb_classes.ravel(), Y_all_scores.ravel())
    # We flatten out the arrays inside of the two parameters and call the function


    return labels, one_fold_cf, five_fold_cf, cr, micro_avg, roc_display, pr_display
    # Return the values

def task_three(model, input, output):
    ...



def main():
    df = pd.read_csv('lncRNA_5_Cancers.csv')
    # Reads the csv file and formats it into a dataframe

    Y = df['Class']
    X = df.drop(['Ensembl_ID','Class'], axis=1)
    # Assigns our class (label) column to variable Y
    # Assigns everything except the labels and the samples (AKA All the features) to variable X
    # We need to declare axis one because if not it will think we are looking for rows

    svm = SVC(kernel="rbf", probability=True, random_state=42)
    # Initializes a SVC to variable svm

    while True: 

        print("Which task would you like to test?")
        print("Task 2")
        print("Task 3")
        print("Task 4")
        print("Exit")

        command = input("Enter command here: ").lower()

        if command == "task 2":
            labels, one_fold_cf, five_fold_cf, cr, micro_avg, roc_display, pr_display = task_two(svm, X, Y)
            print()
            print("==============One Fold Confusion Matrix=======================")
            print(labels)
            print(one_fold_cf)
            print()
            print("==============Five Fold Confusion Matrix======================")
            print(labels)
            print(five_fold_cf)
            print()
            print("==============Classification Report===========================")
            print(cr, end="")
            print(micro_avg)
            print()
            roc_display.plot()
            plt.show()

            pr_display.plot()
            plt.show()


            # complete task one using our svm model and our label and features

        elif command == "task 2":
            ...

        elif command == "task 2":
            ...

        elif command == "exit":
            break
    

    







if __name__ == "__main__":
    main()