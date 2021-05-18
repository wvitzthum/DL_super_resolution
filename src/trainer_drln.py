from datetime import datetime
import time
import copy
import torch

from .models import DRLN
from .trainer import Trainer
from .utils.utils import AverageMeter
from .utils.image_operations import convert_image

class DRLNTrainer(Trainer):
    ## DRLN specific params

    channels = 64

    model = 'DRLN'

    def train(self):        
        # Model
        drln = DRLN(
            channels=self.channels,
            scale=self.scale_factor
        )

        drln.train()

        drln.to(self.device)

        optimizer = torch.optim.Adam(
            params=filter(lambda p: p.requires_grad, drln.parameters()),
            lr=self.lr
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


        loss = torch.nn.L1Loss()


        loss_l1 = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()  

        loss_counter = 0

        start_time = time.time()
        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()
            for i, (lr_img, hr_img) in enumerate(self.training_loader):
                data_time.update(time.time() - epoch_start)

                optimizer.zero_grad()

                # Move images to device
                lr_img = lr_img.to(self.device)  # (batch_size (N), 3, 24, 24), imagenet-normed
                hr_img = hr_img.to(self.device)  # (batch_size (N), 3, 96, 96), imagenet-normed

                sr_img = drln(lr_img)

                
                sr_img = convert_image(sr_img, source='[-1, 1]', target='imagenet-norm')  # (N, 3, 96, 96), imagenet-normed

                loss_result = loss(sr_img, hr_img)

                # Back-prop.
                optimizer.zero_grad()
                loss_result.backward()

                # Update optimizer
                optimizer.step()

                if loss_result.item() < loss_l1.min:
                    self.log(f"# New best model selected for loss {loss_result.item():.4f} last {loss_l1.min:.4f}")
                    best_model = copy.deepcopy(drln)
                    best_epoch = epoch
                    best_loss = loss_result.item()
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
                loss_l1.update(loss_result.item(), lr_img.size(0))

                # track batch time
                batch_time.update(time.time() - epoch_start)

                if self.save_images:
                    self.save_img(drln, epoch, i)

                # reset epoch time and log results of iteration
                epoch_start = time.time()
                if i % self.print_freq == 0:
                    loss_msg = f'[LOSS {loss_l1.val:.4f}/{loss_l1.avg:.4f}] C[{loss_counter}]'
                    self.log_loss_msg(i, epoch, loss_msg, batch_time.val,  time.time()-start_time)
            scheduler.step()


        # Save the model
        save_time = datetime.now().strftime("[%m-%d]%H%M")

        torch.save({'epoch': epoch,
            'model_state_dict': drln.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_l1.val},
            f'./checkpoints/{save_time}_CP_drln_{epoch}.pth.tar')

        self.log(f'{save_time} Saved DRLN checkpoints at epoch {epoch}')
        self.log_end_msg(time.time()-start_time)
