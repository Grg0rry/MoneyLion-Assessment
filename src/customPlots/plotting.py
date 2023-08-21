import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import learning_curve
from sklearn.metrics import roc_curve, roc_auc_score


# cluster plot
def plot_multiple_hist(df, columns, col_num=3):
    row_num = math.ceil(len(columns)/col_num)
    fig, axis = plt.subplots(nrows=row_num, ncols=col_num, figsize=(15,5*row_num))

    for i, column in enumerate(columns):
        ax = axis[i//col_num, i%col_num]
        ax.hist(df[column])
        ax.set_title(f"Histogram of {column}")
    plt.show()

# correlation plot
def plot_corr(df):
    corrMatrix = df.corr()
    plt.gcf().set_size_inches(15,10)
    sns.heatmap(corrMatrix, annot=True)
    plt.show()

# line plot
def plot_line(data, xlabel):
    plt.plot(data)
    plt.xlabel(xlabel)
    plt.show()

# Learning Curve
def plot_learning_curve(estimator, X_train, y_train, cv, scoring):
    train_size, train_score, test_score = learning_curve(estimator, X_train, y_train, cv=cv, scoring=scoring)

    plt.plot(train_size, 1-np.mean(train_score, axis=1), label='Training Loss')
    plt.plot(train_size, 1-np.mean(test_score, axis=1),  label='Validation Loss')
    plt.xlabel('Training sample sizes')
    plt.ylabel(scoring)
    plt.legend()
    plt.show()

# ROC Curve
def plot_roc_curve(y_preds, y_test):
    for name, preds in y_preds:
        fpr, tpr, threshold = roc_curve(y_test, preds)
        auc_roc = roc_auc_score(y_test, preds)
        plt.plot(fpr, tpr, lw=2, label=f'{name} AUC = %0.2f' % auc_roc)

    plt.plot([0, 1], [0, 1],'r--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.title('ROC Curve')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend(loc = 'lower right')
    plt.grid()
    plt.show()

# cluster plot
def plot_cluster_dist(columns, cluster, col_num=3, alpha=0.5):
    row_num = math.ceil(len(columns)/col_num)
    fig, axis = plt.subplots(nrows=row_num, ncols=col_num, figsize=(15,5*row_num))

    for i, column in enumerate(columns):
        ax = axis[i//col_num, i%col_num]

        for cluster_group, df in cluster:
            ax.hist(df[column], alpha=alpha, label=cluster_group)

        ax.set_title(column)
        ax.set_ylabel('Frequency')
        ax.legend()
    plt.show()