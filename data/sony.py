
from typing import Tuple
import numpy as np
import pandas as pd


def get_train(label=0, scale=False, *args):
    """Get training dataset"""
    return _get_adapted_dataset("train", scale)


def get_test(label=0, scale=False, *args):
    """Get testing dataset"""
    return _get_adapted_dataset("test", scale)


def get_valid(label=0, scale=False, *args):
    """Get validation dataset"""
    return None


def get_shape_input():
    """Get shape of the dataset"""
    return (None, 70)


def get_shape_label():
    """Get shape of the labels"""
    return (None,)


def get_anomalous_proportion():
    return 0.2


def _get_adapted_dataset(split: str, scale: bool) -> Tuple[np.ndarray, np.ndarray]:
    """ Gets the adapted dataset for the experiments

    Args :
            split (str): train or test
    Returns :
            (tuple): <training, testing> images and labels
    """
    # print("_get_adapted",scale)
    dataset = _get_dataset(scale)
    key_img = 'x_' + split
    key_lbl = 'y_' + split

    if split == 'test':
        dataset[key_img], dataset[key_lbl] = _adapt_ratio(dataset[key_img],
                                                          dataset[key_lbl])

    return (dataset[key_img], dataset[key_lbl])


def _get_dataset(scale):
    df = pd.read_csv('data/SonyAIBORobotSurface1_TEST.txt',
                     delim_whitespace=True,
                     header=None,
                     index_col=None)
    labels = df[0].copy()
    labels[labels != 2] = 0
    labels[labels == 2] = 1

    df[0] = labels
    df_train = df.sample(frac=0.7, random_state=42)
    df_test = df.loc[~df.index.isin(df_train.index)]
    df_valid = df_train.sample(frac=0.1, random_state=42)
    # df_train = df_train.loc[~df_train.index.isin(df_valid.index)]

    x_train, y_train = _to_xy(df_train, target=0)
    x_valid, y_valid = _to_xy(df_valid, target=0)
    x_test, y_test = _to_xy(df_test, target=0)

    y_train = y_train.flatten().astype(int)
    y_valid = y_valid.flatten().astype(int)
    y_test = y_test.flatten().astype(int)
    x_train = x_train[y_train != 1]
    y_train = y_train[y_train != 1]
    x_valid = x_valid[y_valid != 1]
    y_valid = y_valid[y_valid != 1]

    if scale:
        print("Scaling Sony dataset")
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_valid = scaler.transform(x_valid)
        x_test = scaler.transform(x_test)

    dataset = {}
    dataset['x_train'] = x_train.astype(np.float32)
    dataset['y_train'] = y_train.astype(np.float32)
    dataset['x_valid'] = x_valid.astype(np.float32)
    dataset['y_valid'] = y_valid.astype(np.float32)
    dataset['x_test'] = x_test.astype(np.float32)
    dataset['y_test'] = y_test.astype(np.float32)

    return dataset


def _to_xy(df: pd.DataFrame, target: str):
    """Converts a Pandas dataframe to the x,y inputs that TensorFlow needs"""
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)
    dummies = df[target]
    return df.as_matrix(result).astype(np.float32), dummies.as_matrix().flatten().astype(int)


def _adapt_ratio(x, y, rho=0.2):
    """Adapt the ratio of normal/anomalous data"""

    # Normal data: label =0, anomalous data: label =1

    rng = np.random.RandomState(42)  # seed shuffling

    inliersx = x[y == 0]
    inliersy = y[y == 0]
    outliersx = x[y == 1]
    outliersy = y[y == 1]

    size_outliers = outliersx.shape[0]
    inds = rng.permutation(size_outliers)
    outliersx, outliersy = outliersx[inds], outliersy[inds]

    size_x = inliersx.shape[0]
    out_size_x = int(size_x*rho/(1-rho))

    out_sample_x = outliersx[:out_size_x]
    out_sample_y = outliersy[:out_size_x]

    x_adapted = np.concatenate((inliersx, out_sample_x), axis=0)
    y_adapted = np.concatenate((inliersy, out_sample_y), axis=0)

    size_x = x_adapted.shape[0]
    inds = rng.permutation(size_x)
    x_adapted, y_adapted = x_adapted[inds], y_adapted[inds]

    return x_adapted, y_adapted
