import os
import json
from PIL import Image
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from itertools import product

def index_folder(folder, images=[]):
    """
    simple multi threaded recusive function to map folder
    Args:
        @param folder: folder str path to folder
        @param images: images list containing absolute paths of directory images
    Returns:
        List with image paths
    """
    print(f'Entering {folder}')
    folders = []
    for i in os.listdir(folder):
        item_path = os.path.join(folder, i)   

        try:
            Image.open(item_path, mode='r')

            images.append(item_path)
        except (PermissionError, IsADirectoryError):
            print(f'found folder {i}')
            print(item_path)
            folders.append(item_path)

    if folders:
        with ThreadPool(cpu_count()) as pool:
            pool.map_async(index_folder, folders).get()
    return images



if __name__ == '__main__':

    print("\nGenerating json list ... \n")

    image_list = index_folder('./processed_data/1000/')

    print(f"There are {len(image_list)} images in the file.\n")

    with open(os.path.join('./data_lists/', 'train_images_R1000.json'), 'w') as j:
        json.dump(image_list, j)

    print(f"Training data json file saved to {'./data_lists/'}\n")
