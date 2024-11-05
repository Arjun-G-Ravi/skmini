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



        