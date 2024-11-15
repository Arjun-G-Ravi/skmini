from .downloader import download_to_cache
import numpy as np

def load_iris(force_download=False):
    path = download_to_cache("https://archive.ics.uci.edu/static/public/53/iris.zip", 'iris', force_download=force_download)
    with open(path/'iris.data') as f:
        dataset = f.read()
    target_dict = {'Iris-setosa':0,'Iris-versicolor':1, 'Iris-virginica':2}
    for target in target_dict:
        dataset = dataset.replace(target, str(target_dict[target]))

    ds = []
    # cant think of a better way than this
    for line in dataset.splitlines():
        if line: ds.append(line.split(','))
    dataset_np = np.array(ds, dtype=float)

    return {'data':dataset_np[:,:-1],
            'target':dataset_np[:,-1],
            'target_names':np.array(list(target_dict.keys())),
            'feature_names':['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',  'petal width (cm)']}

def load_diabetes(force_download=False):
    path = (download_to_cache("https://github.com/scikit-learn/scikit-learn/raw/main/sklearn/datasets/data/diabetes_data_raw.csv.gz", 'diabetes','.csv', force_download=force_download)).parent / 'diabetes.csv'
    path_y = (download_to_cache("https://github.com/scikit-learn/scikit-learn/raw/main/sklearn/datasets/data/diabetes_target.csv.gz", 'diabetes_target','.csv', force_download=force_download)).parent / 'diabetes_target.csv'
    with open(path ) as f:
        dataset = f.read()
    ds = []
    for line in dataset.split('\n'):
        i = line.split(' ')
        if len(i) == 10:
            ds.append(i)
    with open(path_y) as f:
        dataset_y = f.read()
    ds_y = []
    for line in dataset_y.split('\n'):        
        if len(line) == 24:
            ds_y.append(float(line))



    return {'data':np.array(ds,dtype=float), 
            'target':np.array(ds_y,dtype=float), 
            }

def load_digits(force_download=False):
    path = download_to_cache('https://archive.ics.uci.edu/static/public/80/optical+recognition+of+handwritten+digits.zip', 'mnist', force_download=force_download)
    with open(path/'optdigits.tra') as f:
        dataset = f.read()
    train_data = dataset.split('\n')
    mnist_train = []
    for i in train_data:
        if i:
            mnist_train.append(i.split(','))
    with open(path/'optdigits.tes') as f:
        dataset = f.read()
    test_data = dataset.split('\n')
    mnist_test = []
    for i in test_data:
        if i:
            mnist_test.append(i.split(','))
    feature_name = ['pixel_0_0', 'pixel_0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'pixel_0_5', 'pixel_0_6', 'pixel_0_7', 'pixel_1_0', 'pixel_1_1', 'pixel_1_2', 'pixel_1_3', 'pixel_1_4', 'pixel_1_5', 'pixel_1_6', 'pixel_1_7', 'pixel_2_0', 'pixel_2_1', 'pixel_2_2', 'pixel_2_3', 'pixel_2_4', 'pixel_2_5', 'pixel_2_6', 'pixel_2_7', 'pixel_3_0', 'pixel_3_1', 'pixel_3_2', 'pixel_3_3', 'pixel_3_4', 'pixel_3_5', 'pixel_3_6', 'pixel_3_7', 'pixel_4_0', 'pixel_4_1', 'pixel_4_2', 'pixel_4_3', 'pixel_4_4', 'pixel_4_5', 'pixel_4_6', 'pixel_4_7', 'pixel_5_0', 'pixel_5_1', 'pixel_5_2', 'pixel_5_3', 'pixel_5_4', 'pixel_5_5', 'pixel_5_6', 'pixel_5_7', 'pixel_6_0', 'pixel_6_1', 'pixel_6_2', 'pixel_6_3', 'pixel_6_4', 'pixel_6_5', 'pixel_6_6', 'pixel_6_7', 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixel_7_3', 'pixel_7_4', 'pixel_7_5', 'pixel_7_6', 'pixel_7_7']
    target_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    X_train = np.array(mnist_train, dtype=float)[:,:-1].reshape(-1, 8,8)
    y_train = np.array(mnist_train, dtype=int)[:,-1]

    X_test = np.array(mnist_test, dtype=float)[:,:-1].reshape(-1, 8,8)
    y_test = np.array(mnist_test, dtype=int)[:,-1]
    return {
        'data': X_train,
        'target': y_train,
        'feature_names':  feature_name,
        'target_name': target_name,
        'test_data': X_test,
        'test_target': y_test,
    }

def load_cifar10(force_download=False):
    # https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdfhttps://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf
    path = download_to_cache('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz', 'cifar10', force_download=force_download)/'cifar-10-batches-py'
    print(path)
    import pickle
    import os
    files = os.listdir(path)
    X_train, y_train = [], []
    X_test, y_test = [], []
    def unpickle(file):
        with open(file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')
        return data
    for f in files:
        if '.' not in f: # to ignore batches.meta and readme.html
            if 'test' in f:
                print(path/f)
                data = unpickle(path/f)
                print(data.keys())
                X_test.extend(data[b'data'])
                y_test.extend(data[b'labels'])
            else:
                print(path/f)
                data = unpickle(path/f)
                print(data.keys())
                X_train.extend(data[b'data'])
                y_train.extend(data[b'labels'])

    X_train = np.array(X_train, dtype=float)
    X_test = np.array(X_test, dtype=float)
    y_train = np.array(y_train, dtype=float)
    y_test = np.array(y_test, dtype=float)
    # return data
    return {
        'data': X_train,
        'target': y_train,
        'target_names': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
        'test_data': X_test,
        'test_target': y_test,
    }

def load_squad(force_download=False):
    path = download_to_cache('https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json', 'squad', force_download=force_download)
    import json
    with open(path, 'r') as f:
        text = json.load(f)
    return text

def load_imdb(force_download=False):
    import os
    path = download_to_cache(r'https://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz', 'imdb', force_download=force_download) / 'aclImdb'
    print(path)
    train_path = path/ 'train'
    test_path = path/'test'
    def _get_review(path):
        X = []
        y = []
        neg = os.listdir(path/'neg')
        pos = os.listdir(path/'pos')
        for f in neg:
            # print(f)
            with open(path/'neg'/f, 'r') as f:
                X.append(f.read())
                y.append(0)

        for f in pos:
            # print(f)
            with open(path/'pos'/f, 'r') as f:
                X.append(f.read())
                y.append(1)
        return X, y
    X_train, y_train = _get_review(train_path)
    X_test, y_test = _get_review(test_path)
    return {
        'data': X_train,
        'target': y_train,
        'target_names': [0, 1],
        'test_data': X_test,
        'test_target': y_test,
    }

    

if __name__ == '__main__':
    pass