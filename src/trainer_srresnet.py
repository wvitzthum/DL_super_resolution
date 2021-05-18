from datetime import datetime
import time

import copy

import torch
from torch import nn

from .models import SRResNet
from .trainer import Trainer
from .utils.utils import AverageMeter
from .utils.image_operations import convert_image

class SRResNetTrainer(Trainer):
    model = 'SRResNet'
    large_kernel_size = 9  # kernel size of the first and last convolutions which transform the inputs and outputs
    small_kernel_size = 3  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
    channels = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
    n_blocks = 16  # number of residual blocks

    def train(self):
        model = SRResNet(
            large_kernel_size=self.large_kernel_size,
            small_kernel_size=self.small_kernel_size,
            n_channels=self.channels, n_blocks=self.n_blocks,
            scaling_factor=self.scale_factor)
        
        model.to(self.device)
        model.train()


        # Initialize the optimizer
        optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, model.parameters()),
                lr=self.lr
            )

        # loss criterions
        content_loss_criterion = nn.MSELoss()


        batch_time = AverageMeter()
        data_time = AverageMeter() 
        losses = AverageMeter()

        # variables for early stopping
        best_model = None
        best_epoch = None
        best_optimizer = None
        best_loss = None

        loss_counter = 0

        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            batch_start = time.time()
            for i, (lr_imgs, hr_imgs) in enumerate(self.training_loader):
                data_time.update(time.time() - batch_start)

                # Move to default device
                lr_imgs = lr_imgs.to(self.device)  # (batch_size (N), 3, 24, 24), imagenet-normed
                hr_imgs = hr_imgs.to(self.device)  # (batch_size (N), 3, 96, 96), in [-1, 1]

                # Forward prop.
                sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

                # Loss
                loss = content_loss_criterion(sr_imgs, hr_imgs)  # scalar

                # Backward prop.
                optimizer.zero_grad()
                loss.backward()

                # Update model
                optimizer.step()


                if loss.item() < losses.min:
                    self.log(f"# New best model selected for loss {loss.item():.4f} last {losses.min:.4f}")
                    best_model = copy.deepcopy(model)
                    best_epoch = epoch
                    best_loss = loss.item()
                    best_optimizer = copy.deepcopy(optimizer)
                    loss_counter = 0
                elif loss_counter == self.early_stopping:
                    self.log("Early stopping condition has been reached, selected model from epoch %s" % (epoch))
                    self.save_model(
                        epoch=best_epoch, model=best_model, optimizer=best_optimizer,
                        loss=best_loss, start_time=start_time
                    )
                    return
                else: loss_counter += 1
                # Keep track of loss
                losses.update(loss.item(), lr_imgs.size(0))

                # Keep track of batch time
                batch_time.update(time.time() - batch_start)

                # Reset start time
                if self.save_images:
                    self.save_img(model, epoch, i)
                batch_start = time.time()
                if i % self.print_freq == 0:
                    self.log_loss_msg(i, epoch, f'[LOSS {losses.val:.4f}] C[{loss_counter}]', batch_time.val, time.time()-start_time)
        self.save_model(
            epoch=best_epoch, model=best_model, optimizer=best_optimizer,
            loss=best_loss, start_time=start_time
        )


