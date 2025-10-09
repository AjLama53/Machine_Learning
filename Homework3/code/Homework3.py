import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

"""
Algorithms needed for this assignment:
    -Decision Trees
    -Random Forest
    -Gradient Boosting Machine
    -XGBoost
    -LightGBM
    -CatBoost
"""


def task_one(input, output, algorithms, best):
    # Task one states that we need to find the best tree-based classifier using all the algorithms
    # We need to use metrics Accuracy and F1 score

    le = LabelEncoder()

    new_out = le.fit_transform(output)

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

        print(f"Compiling: {name}")


        for fold_number, (train, test) in enumerate(skf.split(input, new_out)):
            # We iterate through each split in our fold

            print(f"fold: {fold_number}/{n_splits}")

            X_train = input.iloc[train]
            X_test = input.iloc[test]
            Y_train = new_out[train]
            Y_true = new_out[test]
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


    for n, metric in metrics.items():
        if metric[0] > best["Metrics"][0] and metric[1] > best["Metrics"][1]:
            best["Metrics"][0] = metric[0]
            best["Metrics"][1] = metric[1]
            best["Name"] = n




    return metrics, best



def task_two(input, output, algorithms, best):

    model = algorithms[best["Name"]]

    le = LabelEncoder()
    new_out = le.fit_transform(output)
    class_names = le.classes_

    X_train, X_test, Y_train, Y_test = train_test_split(
        input, new_out, test_size=0.20, stratify=new_out, random_state=42
    )

    print("Training model")
    model.fit(X_train, Y_train)
    print("Model Trained")

    print("Creating explainer and computing SHAP values")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    print("SHAP values computed")

    # ----- (a) per-cancer top-10 SHAP features -----
    print("Generating per-cancer top 10 SHAP feature plots...")


    for i, cls in enumerate(class_names):
        print(f"  Plotting top 10 features for {cls}...")
        shap.plots.bar(shap_values[:, :, i], max_display=10, show=False)
        plt.title(f"Top 10 SHAP Features for {cls}")
        plt.savefig(f"Homework3/images/shap_top10_{cls}.png", bbox_inches="tight")
        plt.close()

    print("All per-cancer SHAP plots saved successfully.")

    # ----- (b) force plots for one patient (ID: TCGA-39-5011-01A) -----
    print("Generating SHAP force plots for patient TCGA-39-5011-01A...")

    # Find the matching patient row (if present)
    if "Ensembl_ID" in input.columns:
        patient_mask = input["Ensembl_ID"] == "TCGA-39-5011-01A"
    else:
        patient_mask = input.index == "TCGA-39-5011-01A"

    if patient_mask.sum() > 0:
        patient_row = input[patient_mask]
        patient_shap = explainer(patient_row)

        for i, cls in enumerate(class_names):
            shap.plots.force(
                explainer.expected_value[i],
                patient_shap.values[:, :, i],
                patient_row,
                matplotlib=True,
                show=False
            )
            plt.title(f"SHAP Force Plot – Patient TCGA-39-5011-01A ({cls})")
            plt.savefig(f"Homework3/images/shap_force_TCGA-39-5011-01A_{cls}.png", bbox_inches="tight")
            plt.close()
        print("Force plots generated successfully.")
    else:
        print("Patient TCGA-39-5011-01A not found in dataset.")
    return





def task_three(input, output, algorithms, best):

    metrics = {}

    for name, algorithm in algorithms.items():

        n_splits = 5
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        # First we need to split the data so that when we train our tree, it doesnt just memorize the data

        model = algorithm
        # We create our decision tree

        mae_avg = 0
        # Set our mae variable

        mse_avg = 0
        # Set our mse variable

        r2_avg = 0
        # Set our r2 variable

        rmse_avg = 0
        # Set our rmse variable

        print(f"Compiling: {name}")


        for fold_number, (train, test) in enumerate(skf.split(input, output)):
            # We iterate through each split in our fold

            print(f"fold: {fold_number}/{n_splits}")

            X_train = input.iloc[train]
            X_test = input.iloc[test]
            Y_train = output.iloc[train]
            Y_true = output.iloc[test]
            # Assign variables pertaining to the appropriate part of the split


            model.fit(X_train, Y_train)
            # Train our tree based on our input and output train data

            Y_predict = model.predict(X_test)
            # Have our model predict based on our input tests data

            mae_avg += mean_absolute_error(Y_true, Y_predict)
            # Calculate the mae and add it to our variable

            mse_avg += mean_squared_error(Y_true, Y_predict)
            # Calculate the mse and add it to our variable

            r2_avg += r2_score(Y_true, Y_predict)
            # Calculate the r2 and add it to our variable

            rmse_avg += root_mean_squared_error(Y_true, Y_predict)
            # Calculate the rmse and add it to our variable

        
        mae_avg /= n_splits
        # Divide by n_splits to get our avg

        mse_avg /= n_splits
        # Divide by n_splits to get our avg

        r2_avg /= n_splits
        # Divide by n_splits to get our avg

        rmse_avg /= n_splits
        # Divide by n_splits to get our avg


        metrics[name] = [mae_avg, mse_avg, r2_avg, rmse_avg]


    for n, metric in metrics.items():
        if (
            metric[0] < best["Metrics"][0] and
            metric[1] < best["Metrics"][1] and
            metric[2] > best["Metrics"][2] and
            metric[3] < best["Metrics"][3]
        ):
            best["Metrics"][0] = metric[0]
            best["Metrics"][1] = metric[1]
            best["Metrics"][2] = metric[2]
            best["Metrics"][3] = metric[3]
            best["Name"] = n




    return metrics, best

def task_four(input, output, algorithms, best, drugs):
    model = algorithms[best["Name"]]

    drug_names = sorted(drugs.unique())

    X_train, X_test, Y_train, Y_test, drugs_train, drugs_test = train_test_split(
        input, output, drugs, test_size=0.20, random_state=42
    )

    print("Training model")
    model.fit(X_train, Y_train)
    print("Model Trained")

    print("Creating explainer and computing SHAP values")
    explainer = shap.TreeExplainer(model)
    print("Explainer ready")

    os.makedirs("Homework3/images", exist_ok=True)

    # ----- (a) per-drug top-10 SHAP features -----
    print("Generating per-drug top 10 SHAP feature plots...")

    for drug in drug_names:
        mask = drugs_test == drug
        if mask.sum() == 0:
            continue

        X_sub = X_test[mask]
        Y_sub = Y_test[mask]
        shap_values_sub = explainer(X_sub)

        shap.plots.bar(shap_values_sub, max_display=10, show=False)
        plt.title(f"Top 10 SHAP Features for {drug}")
        plt.savefig(f"Homework3/images/shap_top10_{drug}.png", bbox_inches="tight")
        plt.close()

    print("All per-drug SHAP plots saved successfully.")

    # ----- (b) least-error drug–cell-line pair -----
    print("Finding least-error drug–cell-line pair...")
    Y_pred = model.predict(X_test)
    abs_errors = np.abs(Y_pred - Y_test)
    min_idx = np.argmin(abs_errors)

    least_error_drug = drugs_test.iloc[min_idx]
    least_error_cell = X_test.index[min_idx]

    print(f"Least-error sample: Drug={least_error_drug}, CellLine={least_error_cell}")

    shap_values_single = explainer(X_test.iloc[[min_idx]])

    shap.plots.bar(shap_values_single, max_display=10, show=False)
    plt.title(f"Top 10 SHAP Features  {least_error_drug} ({least_error_cell})")
    plt.savefig(f"Homework3/images/shap_least_error_{least_error_drug}_{least_error_cell}.png", bbox_inches="tight")
    plt.close()

    print("Least-error SHAP plot saved.")

    return



def main():
    algorithms_classifier = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting Machine": HistGradientBoostingClassifier(),
        "XGBoost": XGBClassifier(tree_method="hist", device="cuda", max_depth=3, n_estimators=100),
        "LightGBM": LGBMClassifier(device="gpu"),
        "CatBoost": CatBoostClassifier(iterations=50,task_type="GPU")
    }

    best_classifier = {    
        "Name": "XGBoost",
        "Metrics": [0,0]
    }

    algorithms_regressor = {
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting Machine": HistGradientBoostingRegressor(),
        "XGBoost": XGBRegressor(tree_method="hist", device="cuda", max_depth=3, n_estimators=100),
        "LightGBM":LGBMRegressor(device="gpu"),
        "CatBoost":CatBoostRegressor(iterations=50, task_type="GPU")
    }

    best_regressor = {
        "Name": "Gradient Boosting Machine",
        "Metrics": [float("inf"),float("inf"),float("-inf"),float("inf")]
    }

    df1 = pd.read_csv("lncRNA_5_Cancers.csv")

    X1 = df1.drop(['Ensembl_ID', 'Class'], axis=1)

    Y1 = df1['Class']

    df2 = pd.read_csv('hw3-drug-screening-data.csv')

    X2 = df2.drop(['CELL_LINE_NAME', 'DRUG_NAME', 'LN_IC50'], axis=1)
    Y2 = df2['LN_IC50']

    drugs = df2['DRUG_NAME']

    while True:

        print("Select a task:")
        print("Task 1")
        print("Task 2")
        print("Task 3")
        print("Task 4")

        cmd = input("Enter a command: ")


        if cmd.lower() == "task 1":
            metrics, best = task_one(X1, Y1, algorithms_classifier, best_classifier)
            for name, metric in metrics.items():
                print(f"================={name} Stats================")
                print()
                print(f"Algorithm: {name}")
                print(f"Accuracy Score Average: {metric[0]:.4f}")
                print(f"F1 Score Average: {metric[1]:.4f}")
                print()

            print(f"Best Algorithm: {best['Name']} with accuracy: {best['Metrics'][0]} and f1: {best['Metrics'][1]}")

        elif cmd.lower() == "task 2":
            task_two(X1, Y1, algorithms_classifier, best_classifier)

        elif cmd.lower() == "task 3":
            metrics, best = task_three(X2, Y2, algorithms_regressor, best_regressor)
            for name, metric in metrics.items():
                print(f"================={name} Stats================")
                print()
                print(f"Algorithm: {name}")
                print(f"MAE Score Average: {metric[0]:.4f}")
                print(f"MSE Score Average: {metric[1]:.4f}")
                print(f"R2 Score Average: {metric[2]:.4f}")
                print(f"RMSE Score Average: {metric[3]:.4f}")
                print()

            print(f"Best Algorithm: {best['Name']}")
            print(f"MAE:  {best['Metrics'][0]:.4f}")
            print(f"MSE:  {best['Metrics'][1]:.4f}")
            print(f"R2:   {best['Metrics'][2]:.4f}")
            print(f"RMSE: {best['Metrics'][3]:.4f}")

        elif cmd.lower() == "task 4":
            task_four(X2, Y2, algorithms_regressor, best_regressor, drugs)

        elif cmd.lower() == "exit":
            break




if __name__ == "__main__":
    main()