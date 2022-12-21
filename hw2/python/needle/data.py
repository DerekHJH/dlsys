import numpy as np
from .autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import gzip
import struct


class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p = 0.5):
        """
        Args:
            p (*float*): The probability of flipping the input image.
        """
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        if flip_img:
            img = img[:, ::-1, :]
        return img


class RandomCrop(Transform):
    def __init__(self, padding=3):
        """
        Args:
            padding (*int*): The padding on each border of the image.
        """
        self.padding = padding

    def __call__(self, img):
        """ Zero pad and then randomly crop an image.
        Args:
            img: H x W x C NDArray of an image
        Return 
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(low=-self.padding, high=self.padding+1, size=2)
        shift_x += self.padding
        shift_y += self.padding
        original_shape = img.shape
        img = np.pad(img, pad_width=((self.padding, self.padding), (self.padding, self.padding), (0, 0)))
        img = img[shift_x:shift_x+original_shape[0], shift_y:shift_y+original_shape[1], :]
        return img


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):
        """
        Args:
            dataset: `needle.data.Dataset` - a dataset 
            batch_size: `int` - what batch size to serve the data in 
            shuffle: `bool` - set to ``True`` to have the data reshuffle at every epoch, default ``False``.
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                           range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        self.start = 0
        if self.shuffle:
            self.ordering = np.random.permutation(len(self.dataset))
            self.ordering = np.array_split(self.ordering, 
                                           range(self.batch_size, len(self.ordering), self.batch_size))
        return self

    def __next__(self):
        if self.start >= len(self.ordering):
            raise StopIteration
        batch = self.dataset[self.ordering[self.start]]
        self.start += 1
        batch = list(batch)
        for i in range(len(batch)):
            batch[i] = Tensor(batch[i])
        return batch
        


class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        """ Read an images and labels file in MNIST format.  See this page:
        http://yann.lecun.com/exdb/mnist/ for a description of the file format.

        Args:
            image_filename (str): name of gzipped images file in MNIST format
            label_filename (str): name of gzipped labels file in MNIST format
            transforms: an optional list of transforms to apply to data

        Get:
            self.X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            self.y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
        """
        super().__init__(transforms)
        with gzip.open(image_filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            self.X = np.frombuffer(f.read(), dtype=np.uint8).astype(np.float32)
            self.X = self.X.reshape(num, rows, cols, 1)
            self.X = self.X / 255.0
        with gzip.open(label_filename, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            self.y = np.frombuffer(f.read(), dtype=np.uint8).astype(np.uint8)

    def __getitem__(self, index) -> object:
        x = self.X[index]
        y = self.y[index]
        return self.apply_transforms(x), y

    def __len__(self) -> int:
        return self.y.shape[0]

class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])
