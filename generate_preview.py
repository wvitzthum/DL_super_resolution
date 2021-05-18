from PIL import Image, ImageDraw, ImageFont
from skimage.util.dtype import img_as_float

from torchvision.transforms.functional import to_tensor
import cv2
from src.metrics import psnr
import numpy as np
from numpy import asarray
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float

colors = ('red', 'yellow')

pos_crops = [(790, 90), (400, 500)]

crop_size = 100

crop_resize = 250

#images = ['hr.tif', 'srgan.tif', 'esrgan.tif', 'drln.tif']
#captions = ['Ground Truth', 'SRGAN', 'ESRGAN', 'DRLN']

images = ['hr.tif', 'bicubic.tif', 'bilinear.tif', 'laczos.tif']
captions = ['Ground Truth', 'Bicubic', 'Bilinear', 'Lanczos']
# horizontal, vertical
margin = (50, 100)

font = ImageFont.truetype("calibril.ttf", size=72)


def generate_preview(directory):
    imgs = [Image.open(directory+'/'+image) for image in images]

    hr_arr = np.asfarray(Image.open(directory+'/'+images[0]).convert('L'))
    size = imgs[0].size

    canvas = Image.new('RGB',
        ( size[1]*len(imgs)+150, size[0]+620),
        (255, 255, 255)
    )

    for i, image in enumerate(imgs):
        ## metrics
        psnr_value = psnr(to_tensor(image), to_tensor(imgs[0]))
        ssim_value = ssim(hr_arr, np.asfarray(Image.open(directory+'/'+images[i]).convert('L')))

        canvas_pos = (margin[0]*i+size[0]*i, 500)
        canvas.paste(image, canvas_pos)
        draw = ImageDraw.Draw(canvas)
        draw.text((canvas_pos[0], canvas_pos[1]+size[0]), captions[i], font=font, fill='black')
        draw.text((canvas_pos[0], canvas_pos[1]-60), f"PSNR/SSIM: {psnr_value:.2f}/{ssim_value:.2f}", font=font, fill='black')

        for j, (top, left) in enumerate(pos_crops):
            crop_pos = (left, top, left+crop_size, top+crop_size)
            crop = image.crop(crop_pos)
            crop = crop.resize((crop.size[0]+crop_resize, crop.size[1]+crop_resize))
            crop_canv_pos = (margin[0]*i+size[0]*i+(crop.size[0]*j)+margin[0]*j, 50)
            canvas.paste(crop, crop_canv_pos)
            draw.rectangle(
                [canvas_pos[0]+crop_pos[0], canvas_pos[1]+crop_pos[1], canvas_pos[0]+crop_pos[0]+crop_size,
                canvas_pos[1]+crop_pos[1]+crop_size],
                outline=colors[j], width=5)
            draw.rectangle(
                [crop_canv_pos[0], crop_canv_pos[1], crop_canv_pos[0]+crop.size[0],
                crop_canv_pos[1]+crop.size[1]],
                outline=colors[j], width=5)


    canvas.save(f'{directory}'+'\preview.tif')



generate_preview('.\output\[08-22]17_R100')