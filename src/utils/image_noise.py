from PIL import Image
import numpy as np

def add_noise_to_image(img, noise='poisson'):
    """
    Add noise to a single image to create training/test data

    :param img: path to image
    :param noise: string defining noise
    """

    img = np.asarray(Image.open(img))
    if noise == "gaussian":
        row,col,ch= img.shape
        mean = 0
        var = 0.1
        sigma = var**0.5
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = img + gauss
    elif noise == "s&p":
        row,col,ch = img.shape
        s_vs_p = 0.5
        amount = 0.01
        out = np.copy(img)
        # Salt mode
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                for i in img.shape]
        out[coords] = 1

        # Pepper mode
        num_pepper = np.ceil(amount* img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                for i in img.shape]
        out[coords] = 0
        noisy = out
    elif noise == "poisson":
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(img * vals) / float(vals)
    elif noise =="speckle":
        intensity = 0.2
        row,col,ch = img.shape
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = img + img * (gauss * intensity)

    return Image.fromarray(noisy.astype('uint8'), 'RGB')

def save_img(path, file, img, noise):
    file_split = file.split('.')
    img.save(f'{path}{file_split[0]}_{noise}.{file_split[1]}')

if __name__ == "__main__":
    path = 'C:\\Users\wvitz\\Documents\\DL_super_resolution\\test_data\\'
    file = '8000_4000.tif'
    noise = 'gaussian'
    save_img(path, file, add_noise_to_image(path+file, noise), noise)
    noise = 's&p'
    save_img(path, file, add_noise_to_image(path+file, 's&p'), noise)
    noise = 'poisson'
    save_img(path, file, add_noise_to_image(path+file, 'poisson'), noise)
    noise = 'speckle'
    save_img(path, file, add_noise_to_image(path+file, 'speckle'), noise)

