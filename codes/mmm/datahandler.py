import json
import numpy as np
import os
import pickle
import torch
import torch.utils.data as data
from PIL import Image


class DatasetFlickr(data.Dataset):
    def __init__(self, *, filenames: dict, transform=None, image_path=''):
        '''
        コンストラクタ

        Arguments:
            filenames: 'Annotation'，'Category_to_Index'
                       をkeyにそれぞれのpathをvalueとして持つ辞書
            transform: torchvision.transform.Composeについて指定している辞書
            image_path: 画像データセットのpath
        '''
        self._transform = transform
        self._imagelist = json.load(open(filenames['Annotation'], 'r'))
        self._category2index = json.load(
            open(filenames['Category_to_Index'], 'r')
        )
        self._imagepath = image_path

    def __len__(self):
        return len(self._imagelist)

    def __getitem__(self, index):
        item = self._imagelist[index]
        filename = item['file_name']
        labels = sorted(item['labels'])

        image = Image.open(
            os.path.join(self._imagepath, filename)
        ).convert('RGB')
        image = image if self._transform is None else self._transform(image)

        target = np.zeros(len(self._category2index), np.float32)
        target[labels] = 1

        return image, target, filename

    def num_category(self):
        return len(self._category2index)


class DatasetGeotag(data.Dataset):
    def __init__(self, *, class_num, transform=None, data_path=''):
        '''
        コンストラクタ

        Arguments:
            filenames: 'Annotation'，'Category_to_Index'
                       をkeyにそれぞれのpathをvalueとして持つ辞書
            transform: torchvision.transform.Composeについて指定している辞書
            data_path: 位置情報データセットのpath
        '''
        self._transform = transform
        self._data = pickle.load(open(data_path, 'rb'))
        self._class_num = class_num

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index):
        item = self._data[index]
        locate = item['locate'] if self._transform is None \
            else self._transform(item['locate'], dtype=torch.float32)

        target = np.zeros(self._class_num, np.float32)
        target[sorted(item['labels'])] = 1

        # return locate, target, item['file_name']
        return locate, target, 0

    def num_category(self):
        return len(self._class_num)


class DataHandler:
    @staticmethod
    def savePickle(savedata, filename, directory=None, mode='wb'):
        filename = str(filename)
        filename = os.path.splitext(filename)[0] + '.pickle'
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
            filename = os.path.join(directory, filename)

        pickle.dump(savedata, open(filename, mode=mode))

    @staticmethod
    def loadPickle(filename, directory=None, mode='rb'):
        filename = str(filename)
        filename = os.path.splitext(filename)[0] + '.pickle'
        if directory is not None:
            filename = os.path.join(directory, filename)

        if os.path.isfile(filename):
            return pickle.load(open(filename, mode=mode))

    @staticmethod
    def saveNpy(savedata, filename, directory=None):
        filename = str(filename)
        filename = os.path.splitext(filename)[0] + '.npy'
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
            filename = os.path.join(directory, filename)

        np.save(filename, savedata, allow_pickle=True)

    @staticmethod
    def loadNpy(filename, directory=None):
        filename = str(filename)
        filename = os.path.splitext(filename)[0] + '.npy'
        if directory is not None:
            filename = os.path.join(directory, filename)

        if os.path.isfile(filename):
            return np.load(filename, allow_pickle=True)

    @staticmethod
    def saveJson(savedata, filename, directory=None, mode='w'):
        filename = str(filename)
        filename = os.path.splitext(filename)[0] + '.json'
        if directory is not None:
            os.makedirs(directory, exist_ok=True)
            filename = os.path.join(directory, filename)

        json.dump(savedata, open(filename, mode=mode))

    @staticmethod
    def loadJson(filename, directory=None, mode='rb'):
        filename = str(filename)
        filename = os.path.splitext(filename)[0] + '.json'
        if directory is not None:
            filename = os.path.join(directory, filename)

        if os.path.isfile(filename):
            return json.load(open(filename, mode=mode))
