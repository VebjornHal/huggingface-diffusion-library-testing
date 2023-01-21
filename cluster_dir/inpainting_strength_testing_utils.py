from io import BytesIO
import numpy as np
import torch
from torch import autocast
import requests
import PIL
from PIL import Image, ImageDraw, ImageFilter
import ftfy
import random
import argparse
from diffusers import StableDiffusionInpaintPipelineLegacy
from diffusers import LMSDiscreteScheduler
import os
import matplotlib.pyplot as plt
import imagehash
import glob
import pickle
from tqdm import tqdm


def create_pipeline(device):
    # Setting up pipeline
    if device == torch.device('cuda'):
        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            "./stable-diffusion-v1-5",
            revision="fp16",
            torch_dtype=torch.float16).to(device)

    elif device == torch.device('cpu'):
        print('Using cpu')
        pipe = StableDiffusionInpaintPipelineLegacy.from_pretrained(
            "./stable-diffusion-v1-5",
            revision="fp32",
            torch_dtype=torch.float32).to(device)

    # Also removing safety checker
    def dummy(images, **kwargs):
        return images, False

    pipe.safety_checker = dummy

    return pipe


def downscale(image, scale=0.5, dynamic_max=None):
    img_w, img_h = image.size

    def divisible_by_32(x):
        return int(x - (x % 32))

    # Downscaling the image to a maximum size
    if dynamic_max is not None:
        max_side = max(img_h, img_w)
        if max_side > dynamic_max:
            scale = dynamic_max / max_side
            new_w = img_w * scale
            new_h = img_h * scale

        # No downscaling needed
        else:
            new_w = img_w
            new_h = img_h

    else:
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)

    # Resize new_w and new_h to be divisible by 32 by downscaling
    new_w = divisible_by_32(new_w)
    new_h = divisible_by_32(new_h)

    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)


def upscale(image, scale=1):
    img_w, img_h = image.size
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)

    # Resize new_w and new_h to be divisible by 32 by downscaling
    new_w = new_w - (new_w % 32)
    new_h = new_h - (new_h % 32)

    return image.resize((new_w, new_h), Image.Resampling.BICUBIC)


def import_images_from_folder(folder, get_file_names=False, sort_by_name=False):
    images = []
    file_names = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert('RGB')
        if img is not None:
            images.append(img)
        # Removing the file ending
        filename = filename.split('.')[0]
        file_names.append(filename)

    if sort_by_name:
        sorted_filenames = sorted(file_names, key=lambda x: int(x.split('_')[1].split('.')[0]))
        sorted_indexes = [file_names.index(filename) for filename in sorted_filenames]
        sorted_images = [images[i] for i in sorted_indexes]
        images = sorted_images
        file_names = sorted_filenames

    if get_file_names:
        return images, file_names

    return images


def gaussian_blur(image, radius):
    return image.filter(ImageFilter.GaussianBlur(radius))


def check_if_file_has_pkl_ending(file_name):
    if file_name[-4:] != '.pkl':
        return file_name + '.pkl'
    else:
        return file_name


def create_mask(image, radius=None, radius_img_percentage=None, return_radius=False):
    img_w, img_h = image.size

    # Standard numbers
    if radius is None and radius_img_percentage is None:
        radius_scale = 0.5
        if img_w > img_h:
            radius = radius_scale * img_w / 2
        else:
            radius = radius_scale * img_h / 2

    elif radius_img_percentage is not None:

        min_side = min(img_w, img_h)

        radius = radius_img_percentage / 100 * min_side / 2

    radius = np.floor(radius)

    # For retrieving radius later
    if return_radius:
        return radius

    mask = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((img_w / 2 - radius, img_h / 2 - radius, img_w / 2 + radius, img_h / 2 + radius), fill=255)
    return mask


# Create a function for calculating hash difference between two images
def imagehash_diff(img1, img2, phash=False):
    if phash:
        return imagehash.phash(img1) - imagehash.phash(img2)
    hash1 = imagehash.average_hash(img1)
    hash2 = imagehash.average_hash(img2)
    return hash1 - hash2


# Create a function for calculating sum of squared differences between two images
def sum_of_squared_diff(img1, img2):
    return np.sum((np.array(img1) - np.array(img2)) ** 2)


# Create a function for calculating the average mean squared error between two images
def average_mse(img1, img2):
    return np.mean((np.array(img1) - np.array(img2)) ** 2)


def check_and_create_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


# def a function which removes the mask image from the image
def add_white_mask_to_image(image, mask):
    image = np.array(image)
    mask = np.array(mask)

    # Checcking that the mask is the same shape as the image
    if len(mask.shape) != len(image.shape) and len(mask.shape) < 3:
        mask = np.expand_dims(mask, axis=-1)
        mask = np.repeat(mask, 3, axis=-1)

    image = np.where(mask == 255, 255, image)
    return Image.fromarray(image)


def update_version_number_of_file(file_name, save_as_npy=False):
    """

    Args:
        file_name: The name of the file whthout the file ending or version number
        save_as_npy:

    Returns: Filename or saves the file as a new npy file

    """

    # First do a file search in order to find the filename with the highest version number

    file = os.path.basename(file_name)
    file_name = os.path.splitext(file)[0]
    version_number = file_name[-1]

    # Checking if the last character is actually a number
    if not version_number.isdigit():
        raise ValueError("The file name does not end with a number")

    new_version_number = int(version_number) + 1
    new_file_name = images_file_name[:-1] + str(new_version_number)

    if save_as_npy:
        np.save(f'{new_images_file_name}.npy', np.asanyarray(self.inpainted_images, dtype=object))
    else:
        return new_file_name


def save_list(list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(list, f)


def load_list(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


# def add_figures_to_grid(figures):
#     fig, axs = plt.subplots(5, 4, sharey=True)
#     axs = axs.ravel()
#     for i, figure in enumerate(figures):
#         for j, ax in enumerate(figure.axes):
#             axs[i*4+j].plot(ax.lines[0].get_xdata(), ax.lines[0].get_ydata())
#             axs[i*4+j].axis('off')
#     plt.subplots_adjust(hspace=0.4)
#     plt.savefig('big_subplot_figure.png')

def add_figures_to_grid(figures):
    fig, axs = plt.subplots(5, 4, sharey=True)
    axs = axs.ravel()
    for ax, figure in zip(axs, figures):
        ax = figure.axes[0]
        ax.axis('off')

    # for i, figure in enumerate(figures):
    #     for j, ax in enumerate(figure.axes):
    #         for line in ax.get_lines():
    #             axs[i*4+j].plot(line.get_xdata(), line.get_ydata())
    #         axs[i*4+j].axis('off')

    plt.subplots_adjust(hspace=0.4)
    plt.savefig('big_subplot_figure.png')


# Creat if name == main
if __name__ == '__main__':

    def find_mask_radius_percentage_of_smallest_imageside(mask_image):

        # Finding the image_row containing the most white pixels (255)
        highest_sum = 0
        highest_sum_idx = 0
        mask_image_array = np.array(mask_image)
        for idx, row in enumerate(mask_image_array):
            if idx == 0:
                highest_sum = np.sum(row)
                highest_sum_idx = idx
            else:
                if np.sum(row) > highest_sum:
                    highest_sum = np.sum(row)
                    highest_sum_idx = idx

        if len(mask_image_array.shape) == 3:
            row_with_most_white = mask_image_array.transpose(2, 0, 1)[0][highest_sum_idx].flatten()
        else:
            row_with_most_white = mask_image_array[highest_sum_idx].flatten()

        # Counter the number of white pixels in the row
        white_pixels = 0
        for pixel in row_with_most_white:
            if pixel == 255:
                white_pixels += 1

        smallest_mask_side = min(mask_image_array.shape[0:2])
        radius = white_pixels / 2
        radius_percentage = radius / (smallest_mask_side / 2) * 100
        radius_percentage = np.floor(radius_percentage)

        return radius_percentage


    def create_mask_for_images_in_folder(folder, mask_folder, radius=None, radius_img_percentage=None,
                                         mask_images=None):

        images, filenames = import_images_from_folder(folder, get_file_names=True, sort_by_name=True)

        # If mask images are inputted, use those
        if mask_images is not None:
            for image, filename, mask in zip(images, filenames, mask_images):
                mask_filename = filename + '_mask.png'
                check_and_create_folder(mask_folder)
                mask.save(os.path.join(mask_folder, mask_filename))
                print(f'Saved mask for image {filename} to {mask_folder}')
            return

        for image, filename in zip(images, filenames):
            mask = create_mask(image, radius, radius_img_percentage)
            mask_filename = filename + '_mask.png'
            check_and_create_folder(mask_folder)
            mask.save(os.path.join(mask_folder, mask_filename))


    b_create_mask = False
    b_extract_mask_from_pkl = True
    b_find_mrp = False
    b_check_maskcreation = False

    if b_create_mask:
        create_mask_for_images_in_folder('landscape_images', 'landscape_images_masked_70mrp', radius_img_percentage=70)

    if b_extract_mask_from_pkl:
        list_ = load_list('inpainting_strength_testing_files/landscape_images_n-strengths-100_mrp-70.pkl')
        masks = list_[1]
        create_mask_for_images_in_folder(folder='landscape_images', mask_folder='landscape_images_masks',
                                         mask_images=masks)

    if b_find_mrp:
        mask_images = import_images_from_folder('landscape_images_masks', get_file_names=False, sort_by_name=True)
        mrp_list = []
        for image in mask_images:
            mrp_list.append(find_mask_radius_percentage_of_smallest_imageside(image))

    if b_check_maskcreation:
        images = import_images_from_folder('landscape_images', get_file_names=False, sort_by_name=True)
        masked_images = []
        mpr_list = []
        for image in images:
            masked_images.append(create_mask(image, radius_img_percentage=70))

        for image in masked_images:
            mpr_list.append(find_mask_radius_percentage_of_smallest_imageside(image))
