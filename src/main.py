from algo.decision_tree import DecisionTree
from algo.k_means import KMeans
from algo.knn import Knn
from algo.linear_reg import LinearRegression
from algo.logistic_reg import LogisticRegression
from algo.naive_bayes import NaiveBayes
from algo.perceptron import Perceptron
from algo.principle_components_analysis import PCA
from algo.random_forest import RandomForest
from algo.random_forest import RandomForest
from algo.support_vector_machine import SVM
import util

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


def main():
    #--- load data ------------------------------------------------------------#

    # load multi class classification data
    iris = datasets.load_iris()
    X_multi_class, y_muli_class = iris.data, iris.target
    X_multic_train, X_multic_test, y_multic_train, y_multic_test = \
        train_test_split(X_multi_class, y_muli_class, test_size=0.2)

    # load binary class classification data
    bc = datasets.load_breast_cancer()
    X_bin_class, y_bin_class = bc.data, bc.target
    X_binc_train, X_binc_test, y_binc_train, y_binc_test = \
        train_test_split(X_bin_class, y_bin_class, test_size=0.2)

    # load regression data
    X_reg, y_reg = datasets.make_regression(
        n_samples=1000,
        n_features=1,
        noise=20,
    )
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
        X_reg,
        y_reg,
        test_size=0.2
    )

    # load clustering data
    X_cluster, y_cluster = datasets.make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    clusters = len(np.unique(y_cluster))
    
    
    #--- def algorithm pipelines ----------------------------------------------#
    
    print(util.bcolors.OKCYAN + "\n# --- regression --- #" + util.bcolors.ENDC)
    # init regression algorithms
    regr_linear_regr = LinearRegression(lr=0.01, n_iters=1000)
    
    regr_algos = [regr_linear_regr]
    
    # run regression pipelines
    for algo in regr_algos:
        util.regression_pipeline(
            regression=algo,
            X_train=X_reg_train,
            X_test=X_reg_test,
            y_train=y_reg_train,
            y_test=y_reg_test
        )
    
    print(util.bcolors.OKCYAN + "\n# --- multi class classification --- #"
          + util.bcolors.ENDC)
    # init multi class classification algorithms
    clf_knn = Knn(k=5)
    
    binary_class_algos = [clf_knn]
    
    # run multi class classification pipelines
    for algo in binary_class_algos:
        util.classification_pipeline(
            classifier=algo,
            X_train=X_multic_train,
            X_test=X_multic_test,
            y_train=y_multic_train,
            y_test=y_multic_test
        )

    print(util.bcolors.OKCYAN + "\n# --- binary classification --- #"
          + util.bcolors.ENDC)
    # init binary classification algorithms
    clf_logistic_reg = LogisticRegression(lr=0.01, n_iters=1000)
    clf_decision_tree = DecisionTree(max_depth=10)
    clf_random_forest = RandomForest(n_trees=10, min_samples_split=3)
    clf_naive_bayes = NaiveBayes()
    clf_percepton = Perceptron(learning_rate=0.01, n_iters=1000)
    clf_svm = SVM(learning_rate=0.01, lambda_param=0.1, n_iters=1000)
    
    bin_class_algos = [
        clf_logistic_reg,
        clf_decision_tree,
        clf_random_forest,
        clf_naive_bayes,
        clf_percepton,
        clf_svm
    ]
    
    # run binary classification pipelines
    for algo in bin_class_algos:
        util.classification_pipeline(
            classifier=algo,
            X_train=X_binc_train,
            X_test=X_binc_test,
            y_train=y_binc_train,
            y_test=y_binc_test
        )
    
    print(util.bcolors.OKCYAN + "\n# --- dimensionality reduction --- #"
          + util.bcolors.ENDC)
    # init dimensionality reduction algorithms
    dr_pca = PCA(n_components=2)
    
    dim_reduction_algos = [dr_pca]
    
    # run dimensionality reduction pipelines   
    for algo in dim_reduction_algos:
        util.dim_reduction_pipeline(
            dim_reduction=algo,
            X_train=X_multic_train,
            X_test=X_multic_test
        )
    
    print(util.bcolors.OKCYAN + "\n# --- clustering --- #"
          + util.bcolors.ENDC)
    # init clustering algorithms
    cl_kmeans = KMeans(K=clusters, max_iters=100, plot_steps=False)
    
    cluster_algos = [cl_kmeans]
    
    # run clustering pipelines
    for algo in cluster_algos:
        util.k_clustering_pipeline(
            clustering=algo,
            X_train=X_cluster,
            y_train=y_cluster
        )


#--- run algorithm pipelines --------------------------------------------------#

if __name__ == '__main__':
    main()