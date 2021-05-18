import torch
from datetime import datetime
from pathlib import Path

from src.models import SRResNet, DRLN, ESRGANGenerator, SRGANGenerator

from src.utils.image_operations import convert_image

from PIL import Image, ImageDraw, ImageFont




#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'
# Model checkpoints
srresnet_cp = "./output/CSF_100_100/[08-22]1503_CP_SRResNet__L_019.pth.tar"
srgan_cp = "./output/CSF_100_100/[08-22]1550_CP_SRGAN_g_L_094.pth.tar"
esrgan_cp = "./output/CSF_100_100/[08-22]1635_CP_ESRGAN_g_L_112.pth.tar"
drln_cp = "./output/CSF_100_100/[08-22]1728_CP_DRLN__L_003.pth.tar"

def srresnet_sr_img(img, directory, ext):
    srresnet_checkpoint = torch.load(srresnet_cp)
    srresnet = SRResNet()
    srresnet.load_state_dict(srresnet_checkpoint['model_state_dict'])
    srresnet.to(device)
    srresnet.eval()

    # Super-resolution SRResNet
    sr_img_srresnet = srresnet(img)
    sr_img_srresnet = sr_img_srresnet.squeeze(0).detach()
    sr_img_srresnet = convert_image(sr_img_srresnet, source='[-1, 1]', target='pil')
    sr_img_srresnet.save(f'{str(directory)}\\srresnet.{ext}')
    del srresnet
    return sr_img_srresnet

def srgan_sr_img(img, directory, ext):
    srgan_checkpoint = torch.load(srgan_cp)
    srgan = SRGANGenerator()
    srgan.load_state_dict(srgan_checkpoint['model_state_dict'])
    srgan.to(device)
    srgan.eval()

    sr_img_srgan = srgan(img)
    sr_img_srgan = sr_img_srgan.squeeze(0).detach()
    sr_img_srgan = convert_image(sr_img_srgan, source='[-1, 1]', target='pil')
    sr_img_srgan.save(f'{str(directory)}\\srgan.{ext}')
    del srgan
    return sr_img_srgan

def esrgan_sr_img(img, directory, ext):
    esrgan_checkpoint = torch.load(esrgan_cp)
    esrgan = ESRGANGenerator()
    esrgan.load_state_dict(esrgan_checkpoint['model_state_dict'])
    esrgan.to(device)
    esrgan.eval()
    # ESRGAN
    sr_img_esrgan = esrgan(img)
    sr_img_esrgan = sr_img_esrgan.squeeze(0).detach()
    sr_img_esrgan = convert_image(sr_img_esrgan, source='[-1, 1]', target='pil')
    sr_img_esrgan.save(f'{str(directory)}\\esrgan.{ext}')
    del esrgan
    return sr_img_esrgan

def drln_sr_img(img, directory, ext):
    drln_checkpoint = torch.load(drln_cp)
    drln = DRLN()
    drln.load_state_dict(drln_checkpoint['model_state_dict'])
    drln.to(device)
    drln.eval()

    # DRLN
    sr_img_drln = drln(img)
    sr_img_drln = sr_img_drln.squeeze(0).detach()
    sr_img_drln = convert_image(sr_img_drln, source='[-1, 1]', target='pil')
    sr_img_drln.save(f'{str(directory)}\\drln.{ext}')
    del drln
    return sr_img_drln
    

def visualize_sr(img, out):


    file_name = img.split('/')[-1].split('.')[0]
    ext = img.split('/')[-1].split('.')[1]
    save_time = datetime.now().strftime("[%m-%d]%H")
    directory = Path(f'{out}{save_time}_{file_name}')
    directory.mkdir(parents=True, exist_ok=True)

    hr_img = Image.open(img, mode="r")
    hr_img = hr_img.convert('RGB')
    hr_img.save(f'{str(directory)}\\hr.{ext}')

    lr_img = hr_img.resize((int(hr_img.width / 4), int(hr_img.height / 4)),
                           Image.BICUBIC)
                           
    blank = Image.new('RGB', (hr_img.width, hr_img.height), (255, 255, 255))

    # Bicubic
    bicubic_img = lr_img.resize((hr_img.width, hr_img.height), Image.BICUBIC)
    bicubic_img.save(f'{str(directory)}\\bicubic.{ext}')

    # Bilinear
    bilinear_img = lr_img.resize((hr_img.width, hr_img.height), Image.BILINEAR)
    bilinear_img.save(f'{str(directory)}\\bilinear.{ext}')

    # Lanczos
    lanczos_img = lr_img.resize((hr_img.width, hr_img.height), Image.LANCZOS)
    lanczos_img.save(f'{str(directory)}\\laczos.{ext}')

    img = convert_image(lr_img, source='pil', target='imagenet-norm').unsqueeze(0).to(device)

    # Super-resolution (SR) with SRResNet
    sr_img_srresnet = srresnet_sr_img(img, directory, ext)
    
    # Super-resolution (SR) with SRGAN
    sr_img_srgan = srgan_sr_img(img, directory, ext)

    # ESRGAN
    sr_img_esrgan = esrgan_sr_img(img, directory, ext)

    # DRLN
    sr_img_drln = drln_sr_img(img, directory, ext)
    

    srimgs = [sr_img_srresnet, sr_img_srgan, sr_img_esrgan, sr_img_drln]
    imgs = [lr_img, hr_img, blank, blank, bicubic_img, bilinear_img, lanczos_img, blank]
    imgs.extend(srimgs)
    captions = [
        'low resolution', 'high resolution', '', '',
        'bicubic', 'bilinear', 'lanczos', '',
        'srresnet', 'SRGAN', 'ESRGAN', 'DRLN'
    ]
    grid = [4,4]
    margin = 90

    grid_img = Image.new('RGB',
        (grid[0] * hr_img.width + grid[0] * margin+margin, grid[1] * hr_img.height + grid[1] * margin+margin),
        (255, 255, 255)
    )

    draw = ImageDraw.Draw(grid_img)

    font = ImageFont.truetype("calibril.ttf", size=72)

    h = sr_img_srresnet.height
    w = sr_img_srresnet.width

    for i, img in enumerate(imgs):
        col, row = (i%grid[0], i//grid[1])
        grid_img.paste(img, box=(col*w+col*margin+margin, row*h+row*margin+margin))
        text_size = font.getsize(captions[i])
        text_width = ((col*w + img.width/2) + (col*margin+margin) - text_size[0]/2)
        text_height = row*h+row*margin + margin - text_size[1] - 1
        draw.text(xy=[
            text_width if text_width >= 0 else 0,
            text_height
            ], text=captions[i], font=font, fill='black')



    print(f"Sucessfully processed {str(directory)}")

    grid_img.save(f'{str(directory)}\\grid.{ext}')
    #grid_img.show()
    return grid_img


if __name__ == '__main__':

    #visualize_sr("./test_data/4000_4000.jpg", './output/')
    
    #visualize_sr("./test_data/8000_4000.tif", './output/')
    #visualize_sr("./test_data/8000_4000_gaussian.tif", './output/')
    visualize_sr("./test_data/poisson.tif", './output/')
    #visualize_sr("./test_data/8000_4000_s&p.tif", './output/')
    #visualize_sr("./test_data/8000_4000_speckle.tif", './output/')

    #visualize_sr("./test_data/butterfly.png", './output/')
    #visualize_sr("./test_data/R227_i_hg_hg0442-0.png", './output/')

    #visualize_sr("./test_data/R100.tif", './output/')