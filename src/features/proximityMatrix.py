import numpy as np
from scipy.spatial.distance import pdist, squareform

def proximityMatrix(model, df, normalize=True, dist=True):
    
    '''
    Takes in a model and the model data matrix and calculates the proximity matrix
    The proximity matrix is converted into a distance matrix by taking 1 - proximity
    
    Parameters
    ----------
    model : model object
        A trained sklearn model with apply method

    df : Pandas DataFrame
        Model Matrix for calculating proximity matrix. 
        Must have the same features as matrix used to train the model

    normalize : bool, optional (default=True)
        Converts the entries in the matrix to values between 0 and 1
        
    dist : bool, optional (default=True)
        Converts proximity matrix into a distance matrix.
        The smaller the value, the closer the distance.

    '''
    
    # Returns the number of trees for which a pair of observations 
    # are in the same terminal node (have the same leaf index)
    def dfun(u, v):
        ret = (u == v).sum()
        return ret
    
    leaf_indices_df = model.apply(df)
    prox_mat = pdist(leaf_indices_df, dfun)
    prox_mat = squareform(prox_mat)
    
    # Set diagonals to non-zero
    np.fill_diagonal(prox_mat, leaf_indices_df.shape[1])

    if dist:
        prox_mat = leaf_indices_df.shape[1] - prox_mat

    if normalize:
        prox_mat = prox_mat / leaf_indices_df.shape[1]
    
    return prox_mat


if __name__ == "__main__":

    import pandas as pd
    from sklearn import datasets
    from sklearn.ensemble import RandomForestClassifier
    
    #### Read in data and preprocess

    iris = datasets.load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)

    # Create a list of the feature column's names
    features = df.columns[:4]

    # Add a new column with the species names, this is what we are going to try to predict
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

    # train['species'] contains the actual species names. Before we can use it,
    # we need to convert each species name into a digit. So, in this case there
    # are three species, which have been coded as 0, 1, or 2.
    y = pd.factorize(df['species'])[0]

    #### Fit random forest on full dataset

    # Create a random forest Classifier. By convention, clf means 'Classifier'
    clf = RandomForestClassifier(random_state=0, n_estimators=100, oob_score=True)

    # Train the Classifier to take the training features and learn how they relate
    # to the training y (the species)
    clf.fit(df[features], y)

    # Test proximity matrix function
    print('\nnormalize=True, dist=True')
    print(proximityMatrix(clf, df[features], normalize=True, dist=True))

    print('\nnormalize=False, dist=True')
    print(proximityMatrix(clf, df[features], normalize=False, dist=True))

    print('\nnormalize=True, dist=False')
    print(proximityMatrix(clf, df[features], normalize=True, dist=False))

    print('\nnormalize=False, dist=False')
    print(proximityMatrix(clf, df[features], normalize=False, dist=False))