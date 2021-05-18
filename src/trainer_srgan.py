from datetime import datetime
import time
import copy
from pathlib import Path

import torch
from torch import nn

from .models import SRGANGenerator, SRGANDiscriminator, TruncatedVGG19
from .trainer import Trainer
from .utils.utils import AverageMeter
from .utils.image_operations import convert_image

class SRGANTrainer(Trainer):
    ## SRGAN specific params
    model = 'SRGAN'

    # generator 
    kernel_l_g = 9  # first and last convolution k size for inputs and outputs
    kernel_s_g = 3  # residual and subpixel convolutional blocks kernel size
    channels_g = 64  # input and output channels for the residual and subpixel conv blocks
    blocks_g = 16

    # discriminator
    kernel_d = 3  # kernel size in all convolutional blocks
    channels_d = 64  # num of output channels in the first convolutional block
    blocks_d = 8  # num of convolutional blocks
    fc_d = 1024  # size of the first fully connected layer

    def train(self):
        # Generator
        generator = SRGANGenerator(
            large_kernel_size=self.kernel_l_g,
            small_kernel_size=self.kernel_s_g,
            n_channels=self.channels_g,
            n_blocks=self.blocks_g,
            scaling_factor=self.scale_factor
        ).to(self.device)

        srresnet_cp = Path(self.srresnet_cp)
        if srresnet_cp.is_file():
            generator.initialize_with_srresnet(
                srresnet_checkpoint=self.srresnet_cp
            )
        optimizer_g = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, generator.parameters()),
            lr=self.lr
        )

        # Discriminator
        discriminator = SRGANDiscriminator(
            kernel_size=self.kernel_d,
            n_channels=self.channels_d,
            n_blocks=self.blocks_d,
            fc_size=self.fc_d
        ).to(self.device)

        truncated_vgg19 = TruncatedVGG19(
            i=self.vgg19_i, j=self.vgg19_j
        ).to(self.device)
        truncated_vgg19.eval()

        optimizer_d = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, discriminator.parameters()),
            lr=self.lr
        )

        # loss criterions
        con_loss_crit = nn.MSELoss().to(self.device)
        adv_loss_crit = nn.BCEWithLogitsLoss().to(self.device)



        batch_time = AverageMeter()
        data_time = AverageMeter() 
        losses_g = AverageMeter()  
        losses_d = AverageMeter()

        # variables for early stopping
        best_model_d, best_model_g = None, None
        best_epoch = None
        best_optimizer_d, best_optimizer_g = None, None
        best_loss = None

        loss_counter = 0

        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()

            generator.train()
            discriminator.train()


            for i, (lr_imgs, hr_imgs) in enumerate(self.training_loader):
                data_time.update(time.time() - epoch_start)

                # Move to default device
                lr_imgs = lr_imgs.to(self.device)
                hr_imgs = hr_imgs.to(self.device)

                # GENERATOR UPDATE
                optimizer_g.zero_grad()
                # Generate
                sr_img = generator(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]
                sr_imgs = convert_image(sr_img, source='[-1, 1]', target='imagenet-norm')  # (N, 3, 96, 96), imagenet-normed

                # VGG feature maps for sr and hr images
                sr_imgs_in_vgg_space = truncated_vgg19(sr_imgs)
                hr_imgs_in_vgg_space = truncated_vgg19(hr_imgs).detach()

                sr_disc = discriminator(sr_imgs)

                # Calculate losses
                content_loss = con_loss_crit(sr_imgs_in_vgg_space, hr_imgs_in_vgg_space)
                adversarial_loss = adv_loss_crit(sr_disc, torch.ones_like(sr_disc))
                perceptual_loss = content_loss + self.beta * adversarial_loss

                # Back-prop.

                perceptual_loss.backward()

                # Update generator
                optimizer_g.step()


                # DISCRIMINATOR UPDATE
                optimizer_d.zero_grad()

                hr_disc = discriminator(hr_imgs)
                sr_disc = discriminator(sr_imgs.detach())

                adversarial_loss = adv_loss_crit(sr_disc, torch.zeros_like(sr_disc)) + \
                                adv_loss_crit(hr_disc, torch.ones_like(hr_disc))

                # Back-prop.
                adversarial_loss.backward()

                # Update discriminator
                optimizer_d.step()

                if perceptual_loss.item() < losses_g.min:
                    self.log(f"# New best model selected for loss {perceptual_loss.item():.4f} last {losses_g.min:.4f}")
                    best_model_g = copy.deepcopy(generator)
                    best_model_d = copy.deepcopy(discriminator)
                    best_epoch = epoch
                    best_loss = adversarial_loss.item()
                    best_optimizer_d = copy.deepcopy(optimizer_d)
                    best_optimizer_g = copy.deepcopy(optimizer_g)
                    loss_counter = 0
                elif loss_counter == self.early_stopping:
                    self.log("Early stopping condition has been reached, selected model from epoch %s" % (epoch))
                    self.save_model(
                        epoch=best_epoch, model=best_model_d, optimizer=best_optimizer_d,
                        loss=best_loss, start_time=start_time, identifier='d'
                    )
                    self.save_model(
                        epoch=best_epoch, model=best_model_g, optimizer=best_optimizer_g,
                        loss=best_loss, start_time=start_time, identifier='g'
                    )
                    return
                else: loss_counter += 1

                # Keep track of loss
                losses_g.update(perceptual_loss.item(), lr_imgs.size(0))
                losses_d.update(adversarial_loss.item(), lr_imgs.size(0))
                # track batch time
                batch_time.update(time.time() - epoch_start)

                if self.save_images:
                    self.save_img(generator, epoch, i)

                # reset epoch time and log results of iteration
                epoch_start = time.time()
                if i % self.print_freq == 0:
                    loss_msg = f'[G {losses_g.val:.4f}/{losses_g.avg:.4f}] [D {losses_d.val:.4f}] C[{loss_counter}]'
                    self.log_loss_msg(i, epoch, loss_msg, batch_time.val,  time.time()-start_time)

        # Save the model
        save_time = datetime.now().strftime("[%m-%d]%H%M")

        # save generator model
        torch.save({'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer_g.state_dict(),
            'loss': losses_g.val},
            f'./checkpoints/{save_time}_CP_srgan_g_{epoch}.pth.tar')

        # save discriminator model
        torch.save({'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_d.state_dict(),
            'loss': losses_g.val},
            f'./checkpoints/{save_time}_CP_srgan_d_{epoch}.pth.tar')

        self.log(f'{save_time} Saved SRGAN checkpoints at epoch {epoch}')
        self.log_end_msg(time.time()-start_time)