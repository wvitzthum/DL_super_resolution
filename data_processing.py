import os
import shutil
from itertools import product
from pathlib import Path
import PIL
from PIL import Image


# To avoid the PIL file size restrictions that are in place to avoid bomb DOS attacks
PIL.Image.MAX_IMAGE_PIXELS = 386973490 

def clean_out_dir(dir_out, name, d):
    path = Path(os.path.join(dir_out, f'{d}/{name}/'))
    if path.is_dir():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    print(f'{path} dir cleaned and created')


def tile(filename, dir_in, dir_out, d):
    name, ext = os.path.splitext(filename)
    clean_out_dir(dir_out, name, d)
    img = Image.open(os.path.join(dir_in, filename))
    w, h = img.size
    grid = list(product(range(0, h-h%d, d), range(0, w-w%d, d)))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(dir_out, f'{d}/{name}/{i}_{j}{ext}')
        img.crop(box).save(out)


def tile_all(origin_dir, output_dir, size):
    files = [
        'R038a_i_hg_hg0031.tif',
        'R040a_i_cvh_cvh2415_PM.tif',
        'R098_i_hg_hg0142_PM.tif',
        'R182_i_cvh_cvh0066.tif',
        'R227_i_hg_hg0442.tif',
        'R293_i_cvh_cvh1060.tif',
        'R787_A_cvh_cvh0938.tif',
    ]

    for file in files:
        tile(
            file, origin_dir, 
            output_dir, size
        )
    
        print(f"successfully processed file {file}")


if __name__ == "__main__":
    origin_dir = 'G:/data/raw/'
    output_dir = './processed_data/'
    size = 1000
    tile_all(origin_dir, output_dir, size)