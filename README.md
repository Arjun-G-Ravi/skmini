# Skmini(short for sklearn-mini)
Sklearn is a popular machine-learning library (https://github.com/scikit-learn/scikit-learn) with hundreds of thousands of lines of code.` The target of this project is to make a miniaturized version of sklearn under 1000 lines of code.` The current purpose of the library is just to act as a 'simple library' which beginners easily approach.

## Structure
 - Models
    1. Regression
       - [x] Linear regression
       - [ ] Regression Trees
    2. Classification
       - [x] Logistic Regression
       - [ ] Classification Trees
       - [ ] SVM
       - [ ] k-nearest neighbours
    3. Dimensionality reduction
       - [ ] PCA?
    4. Clustering
       - [ ] K-means
    5. Preprocessing
       - [ ] StandardScaler # standardisation
       - [ ] MinMaxScaler # normalization
       - [ ] Feature Encoding: Include LabelEncoder and OneHotEncoder
       - [ ] Principal Component Analysis # Dimensionality Reduction
       - [ ] Singular Value Decomposition # Dimensionality Reduction
    6. Model selection
       - [ ] train_test_split
       - [ ] grid search
       - [ ] pipeline framework
 - Datasets
   - [ ] Load some basic datasets like titanic, iris, etc.
     - [ ] Tabular
      - [ ] Titanitc
      - [ ] Iris
     - [ ] House price prediction
   - [ ] NLP
     - [ ] Squad
   - [ ] Image
     - [ ] MNIST
     - [ ] cifar10
   - [ ] Ability to make datasets
       - [ ] Make regression, make classification 
   - Model_Selection
   - [ ] Select best models' weights, hyperparameters, etc.
     - [ ] Train-test split
  
#### Note
- Built on Numpy
- no Cuda support (because of the 1K Loc limit)

### TO-DO
- [ ] Write more asserts in tests to ensure models work well
- [ ] Use decorator or something to initialize model and stuff in test functions
- [ ] The download path mightnot be valid for non-linux (this could be aldready solved)
- [ ] Add certificate verification when downloading datasets