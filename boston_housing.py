"""Load the Boston dataset and examine its target (label) distribution."""

# Load libraries
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn import grid_search
from sklearn import cross_validation

################################
### ADD EXTRA LIBRARIES HERE ###
################################
from sklearn.neighbors import KNeighborsRegressor
from matplotlib import cm

def load_data():
    """Load the Boston dataset."""
    boston = datasets.load_boston()
    return boston


def explore_city_data(city_data):
    """Calculate the Boston housing statistics."""

    # Get the labels and features from the housing data
    housing_prices = city_data.target
    housing_features = city_data.data

    ###################################
    ### Step 1. YOUR CODE GOES HERE ###
    ###################################

    # Please calculate the following values using the Numpy library
    # Size of data (number of houses)?

    data_size = housing_prices.shape[0] # number of rows
    #print(data_size)
    # Number of features?
    feature_size = housing_features.shape[1] #number of features
    #print(feature_size)
    # Minimum price?
    min_price = np.min(housing_prices)
    print("MIN_PRICE : ", min_price)
    # Maximum price?
    max_price = np.max(housing_prices)
    print("MAX_PRICE : ", max_price)
    # Calculate mean price?
    mean_price = np.mean(housing_prices)
    print("MEAN_PRICE : ", mean_price)
    # Calculate median price?
    median_price = np.median(housing_prices)
    print("MEDIAN_PRICE : ", median_price)
    # Calculate standard deviation?
    std_dev = np.std(housing_prices)
    print("STD_DEV : ", std_dev)


def split_data(city_data):
    """Randomly shuffle the sample set. Divide it into 70 percent training and 30 percent testing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    ###################################
    ### Step 2. YOUR CODE GOES HERE ###
    ###################################
    
    #Custom Implementation of splitting
    #data_size = X.shape[0]
    #pivot = data_size * 7/10
    #xy = zip(X,y)
    #np.random.shuffle(xy)
    #X_raw,y_raw = zip(*xy)
    #
    #X_train = X_raw[:pivot]
    #y_train = y_raw[:pivot]
    #X_test = X_raw[pivot:]
    #y_test = y_raw[pivot:]
    
    #Using SKLearn's splitting
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,random_state=0,test_size=0.3)
    return X_train, y_train, X_test, y_test


def performance_metric(label, prediction):
    """Calculate and return the appropriate error performance metric."""
    # list of labels, list of predictions.
    ###################################
    ### Step 3. YOUR CODE GOES HERE ###
    ###################################
    # The following page has a table of scoring functions in sklearn:
    # http://scikit-learn.org/stable/modules/classes.html#sklearn-metrics-metrics
    print("LABEL : ", label)
    return metrics.mean_absolute_error(label,prediction)


def learning_curve(depth, X_train, y_train, X_test, y_test):
    """Calculate the performance of the model after a set of training data."""

    # We will vary the training set size so that we have 50 different sizes
    sizes = np.round(np.linspace(1, len(X_train), 50))
    train_err = np.zeros(len(sizes))
    test_err = np.zeros(len(sizes))

    print "Decision Tree with Max Depth: "
    print depth
    for i, s in enumerate(sizes):
        s = int(s)
        # Create and fit the decision tree regressor model
        regressor = DecisionTreeRegressor(max_depth=depth)
        regressor.fit(X_train[:s], y_train[:s])

        # Find the performance on the training and testing set
        train_err[i] = performance_metric(y_train[:s], regressor.predict(X_train[:s]))
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))


    # Plot learning curve graph
    learning_curve_graph(sizes, train_err, test_err)


def learning_curve_graph(sizes, train_err, test_err):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Training Size')
    pl.plot(sizes, test_err, lw=2, label = 'test error')
    pl.plot(sizes, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Training Size')
    pl.ylabel('Error')
    pl.show()


def model_complexity(X_train, y_train, X_test, y_test):
    """Calculate the performance of the model as model complexity increases."""

    print "Model Complexity: "

    # We will vary the depth of decision trees from 2 to 25
    max_depth = np.arange(1, 25)
    train_err = np.zeros(len(max_depth))
    test_err = np.zeros(len(max_depth))

    for i, d in enumerate(max_depth):
        # Setup a Decision Tree Regressor so that it learns a tree with depth d
        regressor = DecisionTreeRegressor(max_depth=d)

        # Fit the learner to the training data
        regressor.fit(X_train, y_train)

        # Find the performance on the training set
        train_err[i] = performance_metric(y_train, regressor.predict(X_train))

        # Find the performance on the testing set
        test_err[i] = performance_metric(y_test, regressor.predict(X_test))

    # Plot the model complexity graph
    model_complexity_graph(max_depth, train_err, test_err)


def model_complexity_graph(max_depth, train_err, test_err):
    """Plot training and test error as a function of the depth of the decision tree learn."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Max Depth')
    pl.plot(max_depth, test_err, lw=2, label = 'test error')
    pl.plot(max_depth, train_err, lw=2, label = 'training error')
    pl.legend()
    pl.xlabel('Max Depth')
    pl.ylabel('Error')
    pl.show()

def fit_predict_model(city_data):
    """Find and tune the optimal model. Make a prediction on housing data."""

    # Get the features and labels from the Boston housing data
    X, y = city_data.data, city_data.target

    # Setup a Decision Tree Regressor
    regressor = DecisionTreeRegressor()

    parameters = {'max_depth':(1,2,3,4,5,6,7,8,9,10)}

    ###################################
    ### Step 4. YOUR CODE GOES HERE ###
    ###################################

    # 1. Find an appropriate performance metric. This should be the same as the
    # one used in your performance_metric procedure above:
    # http://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html
    scorer = metrics.make_scorer(performance_metric,greater_is_better=False)
   
    # 2. We will use grid search to fine tune the Decision Tree Regressor and
    # obtain the parameters that generate the best training performance. Set up
    # the grid search object here.
    # http://scikit-learn.org/stable/modules/generated/sklearn.grid_search.GridSearchCV.html#sklearn.grid_search.GridSearchCV
    reg = grid_search.GridSearchCV(regressor,param_grid=parameters,scoring=scorer)

    # Fit the learner to the training data to obtain the best parameter set
    reg.fit(X, y)
    print("Optimal Depth : ", reg.best_params_)

    # Use the model to predict the output of a particular sample
    x = [11.95, 0.00, 18.100, 0, 0.6590, 5.6090, 90.00, 1.385, 24, 680.0, 20.20, 332.09, 12.13]
    y = reg.predict(x)
    print "House: " + str(x)
    print "Prediction: " + str(y)

    #Simple Verification with Nearest Neighbor
    nnVerify(city_data,x,y)
    nnVerify_2(city_data,x,y)

################################
######## MY FUNCTIONS ##########
################################

# Simple Nearest-Neighbor Verification
def nnVerify(city_data,x,y):
    """Comparison against Most 'Similar' Data """
    X, Y = city_data.data, city_data.target
    similar_x = 0.0;
    similar_y = 0.0;
    err = None;
    for _x, _y in zip(X,Y):
        tmp_err = np.linalg.norm(np.subtract(x,_x))
        if err is None or err > tmp_err:
            err = tmp_err
            similar_x = _x
            similar_y = _y
    print("X", x, "Y:" , y)
    print("SX:",  similar_x, "SY:", similar_y)

def nnVerify_2(city_data,x,y):
    """ Using SKLearn's KNeighborsRegressor """
    X,Y = city_data.data, city_data.target
    clf = KNeighborsRegressor(n_neighbors=2)
    clf.fit(X,Y)
    y_pred = clf.predict(x)
    print("KNeighborsRegressor")
    print("Y pred(KNN) : ", y_pred)

def identify_relevant(city_data):
    X,Y = city_data.data, city_data.target
    scatter_plot(X,Y)

def scatter_plot(features,prices):
    """Plot training and test error as a function of the training size."""

    pl.figure()
    pl.title('Decision Trees: Performance vs Feature')
    feature_size = features.shape[1]
    colR = np.random.rand(feature_size)
    colG = np.random.rand(feature_size)
    colB = np.random.rand(feature_size)
    labels = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD','TAX', 'PTRATIO', 'B', 'LSTAT'] 
    for i in range(feature_size):
        feature = features[:,i]
        feature -= np.min(features[:,i])
        feature /= np.max(feature)
        #pl.scatter(feature,prices,label=labels[i],c=(colR[i],colG[i],colB[i]),alpha=0.5)
        pl.scatter(feature,prices,c=(colR[i],colG[i],colB[i]),alpha=1.0)
        #pl.legend()
        pl.xlabel(labels[i])
        pl.ylabel('Price')
        pl.show()
   
    pl.legend()
    pl.xlabel('Feature')
    pl.ylabel('Price')
    pl.show()

###################################


#In the case of the documentation page for GridSearchCV, it might be the case that the example is just a demonstration of syntax for use of the function, rather than a statement about 
def main():
    """Analyze the Boston housing data. Evaluate and validate the
    performanance of a Decision Tree regressor on the housing data.
    Fine tune the model to make prediction on unseen data."""

    # Load data
    city_data = load_data()
    # Explore the data
    explore_city_data(city_data)
    # Identify Relevant Features -- Added
    identify_relevant(city_data)
    # Training/Test dataset split
    X_train, y_train, X_test, y_test = split_data(city_data)
    
    print("====")
    print(X_train)
    print("====")
    print(y_train)
    print("====")
    print(X_test)
    print("====")
    print(y_test)
    print("====")

    # Learning Curve Graphs
    max_depths = [1,2,3,4,5,6,7,8,9,10]
    for max_depth in max_depths:
        learning_curve(max_depth, X_train, y_train, X_test, y_test)
    # Model Complexity Graph
    model_complexity(X_train, y_train, X_test, y_test)

    # Tune and predict Model
    fit_predict_model(city_data)

if __name__ == "__main__":
    main()
