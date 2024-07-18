import torch
from abc import abstractmethod, ABCMeta
from common.interfaces import D
from pdes import PDE


class DatasetInterface(metaclass=ABCMeta):
    @property
    @abstractmethod
    def data_interface(self) -> D:
        """The enum defining the dataset's interface
        """
        raise NotImplementedError("data_interface not set!")

    @property
    @abstractmethod
    def train(self) -> torch.utils.data.Dataset:
        raise NotImplementedError("train dataset not set!")

    @property
    @abstractmethod
    def valid(self) -> torch.utils.data.Dataset:
        raise NotImplementedError("valid dataset not set!")

    @property
    @abstractmethod
    def test(self) -> torch.utils.data.Dataset:
        raise NotImplementedError("test dataset not set!")

    @property
    @abstractmethod
    def pde(self) -> PDE:
        raise NotImplementedError("no PDE provided!")
