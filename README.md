# About
Sklearn is a popular machine-learning library (https://github.com/scikit-learn/scikit-learn) with hundreds of thousands of lines of code.` The target of this project is to make a miniaturized version of sklearn under 1000 lines of code.` This would make the library very readable for beginners.

## Sklearn-mini
 - Models
    1. Regression
       1. Linear regression
       2. Regression Trees
       3. MLP
    2. Classification
       1. Logistic Regression
       2. Classification Trees
       3. MLP
       4. SVM
       5. k-nearest neighbours
    3. Dimensionality reduction
       1. PCA?
    4. Clustering
       1. K-means
 - Datasets
   - Load some basic datasets like titanic, iris, etc.
   - Ability to make datasets
 - Model_Selection
   - Select best models' weights, hyperparameters, etc.
 - Preprocessing
     - Train-test split
  
#### Note
- Built on Numpy
- Probably no Cuda support (because of the 1K Loc limit)
