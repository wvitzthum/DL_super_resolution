from .image_operations import ImageTransformer
import torch
from torch.utils.data import Dataset
import json
from PIL import Image


class ImageDataset(Dataset):
    """
    A implementation of the Pytorch dataset for super resolution images
    """

    def __init__(self, data_file, data_type, crop_size, scaling_factor):
        """
        :param data_file: location of data file
        :param data_type: 'train' or 'test'
        :param crop_size: crop size of HR images
        :param scaling_factor: factor by which the HR will be downsampled to
        :param lr_img_type: the format for the LR image for the model
        :param hr_img_type: the format for the HR image for the model
        """

        self.data_file = data_file
        self.data_type = data_type
        self.crop_size = int(crop_size)
        self.scaling_factor = int(scaling_factor)

        assert self.data_type in ['train', 'test']

        if self.data_type == 'train':
            assert self.crop_size % self.scaling_factor == 0, "Crop/dimen != 0"


        with open(self.data_file, 'r') as j:
            self.images = json.load(j)

        self.transform = ImageTransformer(split=self.data_type,
                                         crop_size=self.crop_size,
                                         scaling_factor=self.scaling_factor)

    def __getitem__(self, i):
        """
        Method to iteratively retrieve items based on the index

        :param i: index
        :return: LR and HR images based on index
        """
        img = Image.open(self.images[i], mode='r')
        img = img.convert('RGB')
        lr_img, hr_img = self.transform(img)

        return lr_img, hr_img

    def __len__(self):
        """
        Retrieve length of dataset used by pytorch

        :return: size of this dataset
        """
        return len(self.images)
