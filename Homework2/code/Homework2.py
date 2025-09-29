import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def principle_component_analysis(input, output):

    scaler = StandardScaler()

    scaled_input = scaler.fit_transform(input)
    # Good practice to standardize data to mitigate bias
    
    pca = PCA(n_components=2).fit_transform(scaled_input)
    # PCA function that needs how many dimensions we want, and we fit and tranform it after, returns a 2D numpy array

    data = {'PCA1': pca[:, 0],
            'PCA2':pca[:, 1],
            'Class': output
        }
    
    # We make a data dictionary that has labels PCA1 with first column, PCA2 with the second column
    # And class is just the same as before

    pca_df = pd.DataFrame(data)
    # We feed our data into the constructor to make a dataframe from our dictionary

    plt.figure()
    sns.scatterplot(pca_df, x="PCA1", y="PCA2", hue="Class")
    plt.savefig("images/pca_sp.png")
    # Plot and save the scatterplot

    plt.figure()
    sns.violinplot(pca_df, x="Class", y="PCA1")
    plt.savefig("images/pca_v1.png")
    # PLot the violin plot for PCA1

    plt.figure()
    sns.violinplot(pca_df, x="Class", y="PCA2")
    plt.savefig("images/pca_v2.png")
    # Plot the violin plot for PCA2

    plt.show()

    return


def tSNE(input, output):

    scaler = StandardScaler()

    input_scaled = scaler.fit_transform(input)

    tsne = TSNE(n_components=2, perplexity=50, learning_rate='auto').fit_transform(input_scaled)

    data = {
        'tsne1': tsne[:, 0],
        'tsne2': tsne[:, 1],
        'Class': output
    }

    tsne_df = pd.DataFrame(data)

    plt.figure()
    sns.scatterplot(tsne_df, x="tsne1", y="tsne2", hue="Class")
    plt.savefig("images/tsne_sp.png")

    plt.figure()
    sns.violinplot(tsne_df, x="Class", y="tsne1")
    plt.savefig("images/tsne_v1.png")

    plt.figure()
    sns.violinplot(tsne_df, x="Class", y="tsne2")
    plt.savefig("images/tsne_v2.png")

    plt.show()


    return 





def main():
    
    df = pd.read_csv("code/lncRNA_5_Cancers.csv")

    X = df.drop(['Ensembl_ID', 'Class'], axis=1)

    Y = df['Class']

    while True:

        print("Select from the following options: ")
        print("PCA")
        print("TSNE")
        print("Exit")
        cmd = input("Selection: ")

        if cmd.lower() == "pca":
            principle_component_analysis(X, Y)

        elif cmd.lower() == "tsne":
            tSNE(X, Y)

        elif cmd.lower() == "exit":
            break


if __name__ == "__main__":
    main()





