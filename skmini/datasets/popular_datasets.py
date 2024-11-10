from downloader import download_to_cache
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
    print('target')
    with open(path_y) as f:
        dataset_y = f.read()
    ds_y = []
    for line in dataset_y.split('\n'):
        # i = line.split(' ')
        
        if len(line) == 24:
            ds_y.append(float(line))



    return {'data':np.array(ds,dtype=float), 
            'target':np.array(ds_y,dtype=float), 
            }

if __name__ == '__main__':
    i = load_diabetes(force_download=True)