import argparse
from functools import wraps
from src import DRLNTrainer, SRGANTrainer, SRResNetTrainer, ESRGANTrainer


global_params = {
    'device': 'cuda', 'batch_size': 1, 'iters':1e6, 'workers': 6,
    'save_images': False, 'early_stopping': 1000,
    'training_folder': './data_lists/train_images_R1000.json',
}


def train_srresnet():
    ssresnet_trainer = SRResNetTrainer(**global_params)
    ssresnet_trainer.train()

def train_srgan():
    srgan_trainer = SRGANTrainer(**global_params)
    srgan_trainer.train()

def train_esrgan():
    esrgan_trainer = ESRGANTrainer(**global_params)
    esrgan_trainer.train()

def train_drln():
    drln_trainer = DRLNTrainer(**global_params)
    drln_trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", default='all', type=str, help="choose model to train 'srresnet', 'srgan', 'esrgan' or 'drln'")
    args = parser.parse_args()

    if(args.model == 'all'):
        train_srresnet()
        train_srgan()
        train_esrgan()
        train_drln()
    elif(args.model == 'srresnet'):
        train_srresnet()
    elif(args.model == 'srgan'):
        train_srgan()
    elif(args.model == 'esrgan'):
        train_esrgan()
    elif(args.model == 'drln'):
        train_drln()
