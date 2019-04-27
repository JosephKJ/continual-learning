from __future__ import print_function
import torch.utils.data as data
from PIL import Image
from torchvision import datasets
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import matplotlib.image
from random import randint

class SoundImageMNIST(data.Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``processed/training.pt``
            and  ``processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    urls = [
        'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz',
        'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz',
    ]
    raw_folder = 'raw'
    processed_folder = 'processed'
    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if download:
            self.download()

        self.read_audio_data()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            self.train_data, self.train_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.training_file))
        else:
            self.test_data, self.test_labels = torch.load(
                os.path.join(self.root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
            cls = target.item()
            rand_index = randint(0, len(self.audio_train[cls]) - 1)
            audio = self.audio_train[cls][rand_index]

        else:
            img, target = self.test_data[index], self.test_labels[index]
            cls = target.item()
            rand_index = randint(0, len(self.audio_test[cls]) - 1)
            audio = self.audio_test[cls][rand_index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')
        audio = Image.fromarray(audio.numpy(), mode='L')    # because it is a spectrogram (image)

        if self.transform is not None:
            img = self.transform(img)
            audio = self.transform(audio)

        if self.target_transform is not None:
            target = self.target_transform(target)

        data = torch.cat((img, audio), 0)
        return data, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(self.root, self.processed_folder, self.test_file))

    def read_audio_data(self):
        path = os.path.join(self.root, '../sound_mnist')

        if os.path.exists(os.path.join(path, self.processed_folder, self.training_file)) and \
            os.path.exists(os.path.join(path, self.processed_folder, self.test_file)):
            self.audio_train = torch.load(os.path.join(path, self.processed_folder, self.training_file))
            self.audio_test = torch.load(os.path.join(path, self.processed_folder, self.test_file))
            return

        train_spectrograms = [[],[],[],[],[],[],[],[],[],[]]
        test_spectrograms = [[],[],[],[],[],[],[],[],[],[]]

        for root, _, fnames in sorted(os.walk(path)):
            for fname in fnames:
                if fname.split('.')[1] == 'png':
                    cls, _, id = fname.split('.')[0].split('_')
                    id = int(id)
                    cls = int(cls)
                    path_temp = os.path.join(root, fname)
                    img = torch.from_numpy(self.read_image(path_temp))

                    if id < 10:     # Test image
                        test_spectrograms[cls].append(img)
                    else:           # Train image
                        train_spectrograms[cls].append(img)

        self.audio_train = train_spectrograms
        self.audio_test = test_spectrograms

        try:
            os.makedirs(os.path.join(path, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        with open(os.path.join(path, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(train_spectrograms, f)
        with open(os.path.join(path, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_spectrograms, f)

    def read_image(self, path):
        with open(path, 'rb') as f:
            img = matplotlib.image.imread(path)
            img = np.mean(img, axis=2)
            return img

    def download(self):
        """Download the MNIST data if it doesn't exist in processed_folder already."""
        from six.moves import urllib
        import gzip

        if self._check_exists():
            return

        # download files
        try:
            os.makedirs(os.path.join(self.root, self.raw_folder))
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        for url in self.urls:
            print('Downloading ' + url)
            data = urllib.request.urlopen(url)
            filename = url.rpartition('/')[2]
            file_path = os.path.join(self.root, self.raw_folder, filename)
            with open(file_path, 'wb') as f:
                f.write(data.read())
            with open(file_path.replace('.gz', ''), 'wb') as out_f, \
                    gzip.GzipFile(file_path) as zip_f:
                out_f.write(zip_f.read())
            os.unlink(file_path)

        # process and save as torch files
        print('Processing...')

        training_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 'train-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 'train-labels-idx1-ubyte'))
        )
        test_set = (
            read_image_file(os.path.join(self.root, self.raw_folder, 't10k-images-idx3-ubyte')),
            read_label_file(os.path.join(self.root, self.raw_folder, 't10k-labels-idx1-ubyte'))
        )
        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_label_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8)
        return torch.from_numpy(parsed).view(length).long()


def read_image_file(path):
    with open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        images = []
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16)
        return torch.from_numpy(parsed).view(length, num_rows, num_cols)


class SoundMNIST(datasets.MNIST):
    """
    SoundMNIST: https://github.com/Jakobovski/free-spoken-digit-dataset

    This Dataset object loads only the audio.
    """
    def __init__(self, root, **kwargs):
        super(SoundMNIST, self).__init__(root, **kwargs)

    def download(self):
        try:
            os.makedirs(os.path.join(self.root, self.processed_folder))
        except OSError as e:
            if e.errno == errno.EEXIST:
                pass
            else:
                raise

        train_images = []
        train_labels = []
        test_images = []
        test_labels = []

        for root, _, fnames in sorted(os.walk(self.root)):
            for fname in fnames:
                if fname.split('.')[1] == 'png':
                    cls, _, id = fname.split('.')[0].split('_')
                    id = int(id)
                    cls = int(cls)
                    path = os.path.join(root, fname)
                    img = torch.from_numpy(self.read_image(path))

                    if id < 10:     # Test image
                        test_images.append(img)
                        test_labels.append(cls)
                    else:           # Train image
                        train_images.append(img)
                        train_labels.append(cls)

        training_set = (train_images, train_labels)
        test_set = (test_images, test_labels)

        with open(os.path.join(self.root, self.processed_folder, self.training_file), 'wb') as f:
            torch.save(training_set, f)
        with open(os.path.join(self.root, self.processed_folder, self.test_file), 'wb') as f:
            torch.save(test_set, f)

        print('Done!')

    def read_image(self, path):
        with open(path, 'rb') as f:
            img = matplotlib.image.imread(path)
            img = np.mean(img, axis=2)
            return img
