from algo.knn import Knn
from algo.decision_tree import DecisionTree
from algo.linear_reg import LinearRegression
from algo.logistic_reg import LogisticRegression
from algo.naive_bayes import NaiveBayes
from algo.pca import PCA
from algo.random_forest import RandomForest
from algo.random_forest import RandomForest
import util
from sklearn import datasets
from sklearn.model_selection import train_test_split

#--- load data ----------------------------------------------------------------#

# load multi class classification data
iris = datasets.load_iris()
X_multi_class, y_muli_class = iris.data, iris.target
X_multic_train, X_multic_test, y_multic_train, y_multic_test = train_test_split(
    X_multi_class,
    y_muli_class,
    test_size=0.2
)

# load binary class classification data
bc = datasets.load_breast_cancer()
X_bin_class, y_bin_class = bc.data, bc.target
X_binc_train, X_binc_test, y_binc_train, y_binc_test = train_test_split(
    X_bin_class,
    y_bin_class,
    test_size=0.2
)

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

#--- def algorithm pipelines --------------------------------------------------#

def main():
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

    # init binary classification algorithms
    clf_logistic_reg = LogisticRegression(lr=0.01, n_iters=1000)
    clf_decision_tree = DecisionTree(max_depth=10)
    clf_random_forest = RandomForest(n_trees=10, min_samples_split=3)
    clf_naive_bayes = NaiveBayes()
    
    multi_class_algos = [
        clf_logistic_reg,
        clf_decision_tree,
        # clf_random_forest,
        clf_naive_bayes
    ]
    
    # run binary classification pipelines
    for algo in multi_class_algos:
        util.classification_pipeline(
            classifier=algo,
            X_train=X_binc_train,
            X_test=X_binc_test,
            y_train=y_binc_train,
            y_test=y_binc_test
        )
        
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
    

#--- run algorithm pipelines --------------------------------------------------#

if __name__ == '__main__':
    main()