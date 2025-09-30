import pandas as pd
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

"""
Algorithms needed for this assignment:
    -Decision Trees
    -Random Forest
    -Gradient Boosting Machine
    -XGBoost
    -LightGBM
    -CatBoost
"""

def task_one(input, output):
    # Task one states that we need to find the best tree-based classifier using all the algorithms
    # We need to use metrics Accuracy and F1 score

    algorithms = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting Machine": GradientBoostingClassifier(),
        "XGBoost": xgb.XGBClassifier()
    }

    metrics = {}


    for name, algorithm in algorithms.items():

        n_splits = 5
        skf = StratifiedKFold(n_splits=n_splits)
        # First we need to split the data so that when we train our tree, it doesnt just memorize the data

        model = algorithm
        # We create our decision tree

        accuracy_avg = 0
        # Set our accuracy variable

        f1_avg = 0
        # Set our f1 variable


        for train, test in skf.split(input, output):
            # We iterate through each split in our fold

            X_train = input.iloc[train]
            X_test = input.iloc[test]
            Y_train = output.iloc[train]
            Y_true = output.iloc[test]
            # Assign variables pertaining to the appropriate part of the split


            model.fit(X_train, Y_train)
            # Train our tree based on our input and output train data

            Y_predict = model.predict(X_test)
            # Have our model predict based on our input tests data

            accuracy_avg += accuracy_score(Y_true, Y_predict)
            # Calculate the accuracy and add it to our variable

            f1_avg += f1_score(Y_true, Y_predict, average='weighted')
            # Calculate the f1 and add it to our variable

        
        accuracy_avg /= n_splits
        # Divide by n_splits to get our avg

        f1_avg /= n_splits
        # Divide by n_splits to get our avg

        metrics[name] = [accuracy_avg, f1_avg]


    return metrics









def task_two():
    ...

def task_three():
    ...

def task_four():
    ...


def main():
    df = pd.read_csv("lncRNA_5_Cancers.csv")

    X = df.drop(['Ensembl_ID', 'Class'], axis=1)

    Y = df['Class']


    metrics = task_one(X, Y)
    for name, metric in metrics.items():
        print(f"================={name} Stats================")
        print()
        print(f"Algorithm: {name}")
        print(f"Accuracy Score Average: {metric[0]:.2f}")
        print(f"F1 Score Average: {metric[1]:.2f}")
        print()




    





if __name__ == "__main__":
    main()