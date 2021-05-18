import torch
from datetime import datetime
from pathlib import Path
from PIL import Image
from torchvision import transforms
from src.models import DRLN
from src.utils.image_operations import convert_image

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
drln_cp = "./checkpoints/[08-02]2058_CP_DRLN__L_001.pth.tar"


def load_model():
    """ 
    Load an check endpoint
    
    Returns
        DRLN model
    """
    drln_checkpoint = torch.load(drln_cp)
    drln = DRLN()
    drln.load_state_dict(drln_checkpoint['model_state_dict'])
    drln.to(device)
    drln.eval()
    return drln


def generate_sr(img, out):
    file_name = img.split('/')[-1].split('.')[0]
    ext = img.split('/')[-1].split('.')[1]
    save_time = datetime.now().strftime("[%m-%d]%H")
    directory = Path(f'{out}{save_time}_{file_name}')
    directory.mkdir(parents=True, exist_ok=True)


    drln = load_model()


    img = Image.open(img, mode="r").convert('RGB')
    img = convert_image(img, source='pil', target='imagenet-norm').unsqueeze(0).to(device)
    sr_img_drln = drln(img)
    sr_img_drln = sr_img_drln.squeeze(0).detach()
    sr_img_drln = convert_image(sr_img_drln, source='[-1, 1]', target='pil')
    sr_img_drln.save(f'{str(directory)}\\drln.{ext}')

    return sr_img_drln

if __name__ == "__main__":
    generate_sr("./test_data/R044a_i_hg_hg0032.png", './output/')
