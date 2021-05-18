from datetime import datetime
import time
import copy
import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import MultiStepLR

from .models import ESRGANDiscriminator, ESRGANGenerator
from .trainer import Trainer
from .utils import AverageMeter, PerceptualLoss
from .utils.image_operations import convert_image


class ESRGANTrainer(Trainer):
    ## ESRGAN specific params
    model = 'ESRGAN'

    # generator
    in_channels = 3 
    out_channels = 3
    channels_g = 64
    blocks_g = 23
    gc = 32

    # discriminator
    blocks_d = 4

    # image size
    img_size = 1000

    # optimizer
    weight_decay = 1e-2
    b1 = 0.9
    b2 = 0.999

    # lr
    decay_iter = [50000, 100000, 200000, 300000]

    # loss weights
    adv_loss_weight = 1
    cont_loss_weight = 1
    perc_loss_weight = 1


    def train(self):
        adversarial_loss = nn.BCEWithLogitsLoss().to(self.device)
        content_loss = nn.L1Loss().to(self.device)
        perception_loss = PerceptualLoss().to(self.device)
        # Generator
        generator = ESRGANGenerator().to(self.device)

        # Discriminator
        discriminator = ESRGANDiscriminator().to(self.device)

        # Initialize discriminator's optimizer
        optimizer_g = Adam(
            generator.parameters(),
            lr=self.lr, betas=(self.b1, self.b2),
            weight_decay=self.weight_decay)
        optimizer_d = Adam(
            discriminator.parameters(),
             lr=self.lr, betas=(self.b1, self.b2),
             weight_decay=self.weight_decay)
    
        lr_scheduler_g = MultiStepLR(
            optimizer_g, self.decay_iter
        )
        lr_scheduler_d = MultiStepLR(
            optimizer_d, self.decay_iter
        )

        generator.train()
        discriminator.train()

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

            for i, (lr_img, hr_img) in enumerate(self.training_loader):
                data_time.update(time.time() - epoch_start)

                # Move to default device
                lr_img = lr_img.to(self.device)  # (batch_size (N), 3, 24, 24), imagenet-normed
                hr_img = hr_img.to(self.device)  # (batch_size (N), 3, 96, 96), imagenet-normed

                ###############################
                # Generator update
                ###############################
                optimizer_g.zero_grad()

                # Generate
                sr_img = generator(lr_img)  # (N, 3, 96, 96), in [-1, 1]
                sr_img = convert_image(sr_img, source='[-1, 1]', target='imagenet-norm')  # (N, 3, 96, 96), imagenet-normed

                # discriminate sr images
                hr_disc = discriminator(hr_img)
                sr_disc = discriminator(sr_img)

                disc_rf = hr_disc - sr_disc.mean()
                disc_fr = sr_disc - hr_disc.mean()

                # Calculate the losses
                cont_loss = content_loss(sr_img, hr_img)
                adv_loss = (
                    adversarial_loss(disc_rf, torch.ones_like(disc_rf)) +\
                        adversarial_loss(disc_fr, torch.zeros_like(disc_rf))
                ) / 2
                perc_loss = perception_loss(hr_img, sr_img)


                generator_loss = perc_loss * self.perc_loss_weight + \
                                    adv_loss * self.adv_loss_weight + \
                                    cont_loss * self.cont_loss_weight

                generator_loss.backward()

                # Generator step
                optimizer_g.step()

                ###############################
                # Discriminator
                ###############################

                optimizer_d.zero_grad()

                # Discriminate both hr and sr image
                hr_disc = discriminator(hr_img)
                sr_disc = discriminator(sr_img.detach())

                disc_rf = hr_disc - sr_disc.mean()
                disc_fr = sr_disc - hr_disc.mean()

                disc_adv_loss = (
                    adversarial_loss(disc_rf, torch.ones_like(disc_rf)) +\
                        adversarial_loss(disc_fr, torch.zeros_like(disc_fr))
                ) / 2

                # Backpropagate
                disc_adv_loss.backward()
                optimizer_d.step()

                losses_d.update(disc_adv_loss.item())

                ## Step learning rate schedulers
                lr_scheduler_g.step()
                lr_scheduler_d.step()

                if generator_loss.item() < losses_g.min:
                    self.log(f"# New best model selected for loss {generator_loss.item():.4f} last {losses_g.min:.4f}")
                    best_model_g = copy.deepcopy(generator)
                    best_model_d = copy.deepcopy(discriminator)
                    best_epoch = epoch
                    best_loss = generator_loss.item()
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

                losses_g.update(generator_loss.item())

                # track batch time
                batch_time.update(time.time() - epoch_start)

                if self.save_images:
                    self.save_img(generator, epoch, i)

                # reset epoch time and log results of iteration
                epoch_start = time.time()
                if i % self.print_freq == 0:
                    loss_msg = f'[G {losses_g.val:.4f}/{losses_g.avg:.4f}] [D {losses_d.val:.4f}/{losses_d.avg:.4f}] [C {loss_counter}]'
                    self.log_loss_msg(i, epoch, loss_msg, batch_time.val,  time.time()-start_time)



        # Save the model
        save_time = datetime.now().strftime("[%m-%d]%H%M")

        # save generator model
        torch.save({'epoch': epoch,
            'model_state_dict': generator.state_dict(),
            'optimizer_state_dict': optimizer_g.state_dict(),
            'loss': generator_loss.item()},
            f'./checkpoints/{save_time}_CP_esrgan_g_{epoch}.pth.tar')

        # save discriminator model
        torch.save({'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'optimizer_state_dict': optimizer_d.state_dict(),
            'loss': disc_adv_loss.item()},
            f'./checkpoints/{save_time}_CP_esrgan_d_{epoch}.pth.tar')

        self.log(f'{save_time} Saved ESRGAN checkpoints at epoch {epoch}')
        self.log_end_msg(time.time()-start_time)
