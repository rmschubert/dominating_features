import torch
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import TensorDataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor


class Flatten:
    def __call__(self, img):
        img = img.view(-1)
        return img


def load_MNIST():

    transform = Compose([ToTensor(), Flatten()])

    # load the data
    print("loading data")
    train_dataset = MNIST(root="./data", train=True, download=True, transform=transform)
    print("done loading_data")
    x_train_r, y_train_r = zip(*[t for t in train_dataset])
    x_train = torch.stack(x_train_r)
    y_train = torch.tensor(y_train_r)

    test_dataset = MNIST(root="./data", train=False, transform=transform)
    x_test_r, y_test_r = zip(*[t for t in test_dataset])
    x_test = torch.stack(x_test_r)
    y_test = torch.tensor(y_test_r)

    X = torch.cat((x_train, x_test), dim=0)
    y = torch.cat((y_train, y_test))

    dataset = TensorDataset(X, y)

    return dataset


def load_torch_iris():
    X, y = load_iris(return_X_y=True)
    y_t = torch.from_numpy(y).long()

    s = MinMaxScaler().fit(X)
    X_t = torch.from_numpy(s.transform(X))
    return TensorDataset(X_t, y_t)


def load_torch_breastcancer():
    X, y = load_breast_cancer(return_X_y=True)
    y_t = torch.from_numpy(y).long()

    pipeline = make_pipeline(RobustScaler(), MinMaxScaler())
    X_t = torch.from_numpy(pipeline.fit_transform(X))
    return TensorDataset(X_t, y_t)
