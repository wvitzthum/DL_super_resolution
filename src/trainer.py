import os

import time
import logging
from datetime import datetime
from pathlib import Path

from PIL import Image
import torch

from src.utils.image_operations import convert_image

from .utils.dataloader import ImageDataset
from .utils.utils import *


class Trainer(object):
    model = None
    log_init = False

    checkpoint = None  # path to checkpoint
    batch_size = 20  # batch size
    start_epoch = 0  # start at this epoch
    iters = 100  # number of training iterations
    workers = 2  # amount of DataLoader workers
    vgg19_i = 5  # VGG loss index i
    vgg19_j = 4  # VGG loss index j
    beta = 1e-3  # coefficient weight of adversarial loss with perceptual loss
    print_freq = 100  # print frequency à batches
    lr = 1e-4  # learning rate
    early_stopping = 10 # Count value for early stopping implementation

    # training device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # training 
    batch_size: int
    training_folder = './data_lists/train_images.json'  # folder with JSON data files
    
    # image details
    crop_size = 96
    scale_factor = 4

    # properites for saving images from training iterations
    save_images = False
    img_folder = None

    def __init__(self, **kwargs): 
        self.__dict__.update(kwargs)

        # empty cuda cache
        torch.cuda.empty_cache()

        self.load_data()

        self.epochs = int(self.iters // len(self.training_loader) + 1)
        
        device = torch.cuda.get_device_properties(0)

        self.log(f"##### Training new {self.model} model #####")
        self.log(
            f"Params: {self.device} [LR {self.lr}] [batch_size {self.batch_size}] [iterations {int(self.iters)}] " + \
                f"[workers {self.workers}] [stopping {self.early_stopping}] [Images {self.save_images}]")
        self.log(f"Device: {device.name} Memory: {(device.total_memory/1e9):.2f}GB")
        if self.save_images:
            self.set_save_folder()

    def load_data(self):
        train_dataset = ImageDataset(
            self.training_folder, data_type='train', crop_size=self.crop_size,
            scaling_factor=self.scale_factor)
            
        self.training_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=True)

    def log(self, msg):
        if not self.log_init:
            logging.basicConfig(
                filename=f'./logs/{self.model}.log',
                format='%(asctime)s %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S',
                level=logging.INFO
            )
            self.log_init = True
        logging.info(msg)

    def log_loss_msg(self, i, epoch, loss_msg, batch_time, timer):
        pred_time = self.epochs * (len(self.training_loader)/(i+1)) * int(timer)
        self.log(f'[E {epoch}/{self.epochs}] [B {i}/{len(self.training_loader)}] complete with {loss_msg} in {int(batch_time)}s, {int(timer)//60}m/~{int(pred_time//60)}m')
        memory_used = torch.cuda.max_memory_allocated()
        memory_available = torch.cuda.get_device_properties(0).total_memory
        self.log(f"Memory usage: {int(memory_used/memory_available*100)}% {(memory_used/1e9):.2f} / {(memory_available/1e9):.2f} GB")

    def log_end_msg(self, timer):
        self.log(f'## Training has ended with a total time of {timer//60/60:.2f}h')

    def save_model(self, epoch, model, optimizer, loss, start_time, identifier=''):
        
        save_time = datetime.now().strftime("[%m-%d]%H%M")
        self.log(f'{save_time} Saved {self.model} checkpoints at epoch {epoch} with loss {loss:.2f}')

        loss = f'{loss:.2f}'.replace('.', '')

        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss},
                    f'./checkpoints/{save_time}_CP_{self.model}_{identifier}_L_{loss}.pth.tar')

        self.log_end_msg(time.time()-start_time)

    def set_save_folder(self):
        save_time = datetime.now().strftime("%m-%d_%H%M")
        dir = Path(f'./output/{save_time}_{self.model}')
        os.mkdir(dir)
        self.log(f'Directory {dir} created')
        self.img_folder = dir

    def save_img(self, model, epoch, iteration):
        hr_img = Image.open("./test_data/8000_4000.tif", mode="r").convert('RGB')
        lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
                        Image.BICUBIC)
        lr_img = convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(self.device)
        sr_img = model(lr_img)
        sr_img = sr_img.squeeze(0).detach().to('cpu')
        sr_img = convert_image(sr_img, source='[-1, 1]', target='pil')
        sr_img.save(f'{str(self.img_folder)}//{epoch}_{iteration}.jpg')