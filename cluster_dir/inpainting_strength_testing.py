# Importing all necesarry libraries
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
from inpainting_strength_testing_utils import *

# Using nicer style
plt.style.use('ggplot')


class InpaintingStrengthExperiment:
    def __init__(self, input_img_folder_path, n_strengths=10, mask_radius_percentage=None,
                 need_for_image_processing=False):

        self.images, self.image_filenames_list = import_images_from_folder(input_img_folder_path,
                                                                           get_file_names=True,
                                                                           sort_by_name=True)

        self.num_of_imgs = len(self.images)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_inference_steps = 200
        self.guidance_scale = 0
        self.n_strengths = n_strengths
        self.strengths = np.round(np.linspace(0, 1, n_strengths), 2)

        self.num_of_imgs = len(self.images)
        self.upscale_var = 1.5
        self.downscale_var = 0.5
        self.blur_radius = 2
        self.prompt = ''
        self.dynamic_max = 650

        self.pipe = create_pipeline(self.device)
        self.generator = torch.Generator(device=self.device)

        self.mask_radius_percentage = mask_radius_percentage
        self.disc_area_image_percentages = None

        # Storage lists
        self.masked_images = []
        self.inpainted_images = []
        self.inpainted_images_diff_disc = []  # List of lists of images where each list is the shape of the number of strength values
        self.downscaled_images = []
        self.dynamic_downscaled_images = []
        self.upscaled_images = []
        self.blurred_images = []
        self.white_mask_added_blurred_images = []
        self.white_mask_added_inpainted_images = []
        self.white_mask_added_upscaled_images = []
        self.white_mask_added_upscaled_disc_diff_images = []
        self.white_mask_added_inpainted_images_diff_disc = []
        self.white_mask_added_pre_inpainted_diff_disc_images = []

        # Running the image processing functions and filling the lists
        self.mask_radius_percentage = mask_radius_percentage

        self.need_for_image_processing = need_for_image_processing
        if need_for_image_processing:
            print('Processing images')
            self.all_image_processing(use_dynamic_ds_for_creating_mask=True)

        self.import_from_file = None
        self.load_pkl_filename = None
        self.out_pkl_filename = None
        self.amount_of_images = None
        self.out_images_folder = None
        self.diff_disc_out_folder_name = None

        # Check if folder exists and if not create it
        check_and_create_folder('inpainting_strength_testing_files')
        self.main_folder_path = 'inpainting_strength_testing_files'
        self.main_wd = os.getcwd()

        # Ensuring same generator seeds each time for reproducibility
        if not os.path.exists(f'./{self.main_folder_path}/generator_seeds.npy'):
            generator_seeds = np.random.randint(0, 1e6, size=self.num_of_imgs)
            np.save(f'./{self.main_folder_path}/generator_seeds.npy', generator_seeds)
        else:
            generator_seeds = np.load(f'./{self.main_folder_path}/generator_seeds.npy')
        self.generator_seeds = generator_seeds

        # Printing some parameters
        print(f'{self.device}: Device')
        print(f'{self.num_inference_steps}: Number of inference steps')
        print(f'{self.guidance_scale}: Guidance scale')
        print(f'{self.strengths}: Strength')
        print(f'{self.num_of_imgs}: Number of images')
        print(f'{self.upscale_var}: Upscale var')
        print(f'{self.downscale_var}: Downscale var')
        print(f'{self.blur_radius}: Blur radius')
        print(f'{self.prompt}: Prompt')
        print(f'{self.import_from_file}: Import from npy file')
        print(f'{self.amount_of_images}: Amount of images')

        # Moving into main folder path
        os.chdir(self.main_folder_path)

    # Creating function for inpainting images
    def inpaint(self, init_image, mask_image, generator=None, strength=0.7, attention_slice=False):
        if attention_slice:
            self.pipe.enable_attention_slicing(8)
        image = self.pipe(prompt=self.prompt,
                          init_image=init_image,
                          mask_image=mask_image,
                          guidance_scale=self.guidance_scale,
                          generator=generator,
                          num_inference_steps=self.num_inference_steps,
                          strength=strength,
                          ).images[0]
        return image

    # Creating function for inpainting all the images

    def all_inpainting(self, amount_of_images='all',
                       out_pkl_filename=None,
                       use_dynamic_downscale=True,
                       testing=False):

        # For testing purposes
        if testing:
            amount_of_images = 2
            self.strengths = [0.3, 0.5, 0.8]
            self.out_pkl_filename = self.out_pkl_filename.replace('.pkl', '_testing.pkl')

        if out_pkl_filename is not None:
            self.out_pkl_filename = out_pkl_filename
        else:
            self.out_pkl_filename = 'inpainting_strength_testing.pkl'

        if amount_of_images == 'all':
            self.amount_of_images = self.num_of_imgs
        else:
            self.amount_of_images = amount_of_images

        # Deciding to use blur or not
        images_to_inpaint = []
        if use_dynamic_downscale:
            images_to_inpaint = self.dynamic_downscaled_images
        elif not use_blur:
            images_to_inpaint = self.upscaled_images

        for idx, (mask_image, image_to_inpaint) in enumerate(
                zip(self.masked_images, images_to_inpaint)):
            inpainted_imgs_list = []
            added_white_mask_imgs_list = []
            for strength in self.strengths:
                inpainted_img = self.inpaint(image_to_inpaint,
                                             mask_image,
                                             self.generator,
                                             strength,
                                             attention_slice=False)
                inpainted_imgs_list.append(inpainted_img)

                ### for error analysis
                print(image_to_inpaint.size)
                print(mask_image.size)
                print(inpainted_img.size)
                print('---')
                print(idx)
                print(amount_of_images)
                print('---')

                # Adding white mask for later comparison with hash difference
                white_mask_added_inpainted_img = add_white_mask_to_image(inpainted_img, mask_image)
                added_white_mask_imgs_list.append(white_mask_added_inpainted_img)

            self.white_mask_added_inpainted_images.append(added_white_mask_imgs_list)
            self.inpainted_images.append(inpainted_imgs_list)

            # Breaking the loop if number of images to inpaint of the total is set by user
            if idx + 1 == amount_of_images:
                break

        self.save_all_lists(out_pkl_filename=self.out_pkl_filename)

    def all_inpainting_diff_disc(self, amount_of_images='all',
                                 out_pkl_filename='inpainted_diff_disc',
                                 testing=False,
                                 dynamic_ds=True,
                                 n_masks=20,
                                 mask_percentage_radius=80):

        self.amount_of_images = amount_of_images

        if self.amount_of_images == 'all':
            self.amount_of_images = self.num_of_imgs

        # For testing purposes
        n_masks = 3 if testing else n_masks
        amount_of_images = 2 if testing else amount_of_images
        self.strengths = [0.1] if testing else [0.1, 0.5, 0.9]

        start_percentage = 20
        end_percentage = mask_percentage_radius

        self.disc_area_image_percentages = np.round(np.linspace(start_percentage, end_percentage, n_masks), 2)



        masks_list_of_lists = []

        if dynamic_ds:
            images_to_inpaint = self.dynamic_downscaled_images
        else:
            images_to_inpaint = self.upscaled_images

        for image in images_to_inpaint:
            masks_list = []
            added_white_mask_imgs_list = []
            for disc_area_image_percentage in self.disc_area_image_percentages:
                mask = create_mask(image, radius_img_percentage=disc_area_image_percentage)
                masks_list.append(mask)
                white_mask_added_mask = add_white_mask_to_image(image, mask)
                added_white_mask_imgs_list.append(white_mask_added_mask)
            masks_list_of_lists.append(masks_list)
            self.white_mask_added_pre_inpainted_diff_disc_images.append(added_white_mask_imgs_list)

        for idx, (masks_list, image_to_inpaint) in enumerate(zip(masks_list_of_lists, images_to_inpaint)):
            inpainted_for_each_strength_list = []
            w_inpainted_for_each_strength_list = []

            for strength in self.strengths:
                inpainted_for_each_mask_list = []
                w_inpainted_for_each_mask_list = []

                for mask in masks_list:
                    # Showing some info about the inpainting process
                    print(f'image {idx} of {self.amount_of_images}')
                    print(f'strength {strength}')
                    print(f'mask number {masks_list.index(mask)} of {len(masks_list)}')
                    print(f'size of image: {image_to_inpaint.size}')
                    print(f'size of mask: {mask.size}')

                    inpainted_img = self.inpaint(image_to_inpaint,
                                                 mask,
                                                 self.generator,
                                                 strength,
                                                 attention_slice=False)
                    inpainted_for_each_mask_list.append(inpainted_img)
                    w_inpainted_for_each_mask_list.append(add_white_mask_to_image(inpainted_img, mask))

                inpainted_for_each_strength_list.append(inpainted_for_each_mask_list)
                w_inpainted_for_each_strength_list.append(w_inpainted_for_each_mask_list)

            self.inpainted_images_diff_disc.append(inpainted_for_each_strength_list)
            self.white_mask_added_inpainted_images_diff_disc.append(w_inpainted_for_each_strength_list)

            # Breaking the loop if number of images to inpaint of the total is set by user, mainly for testng
            # purposes
            if idx + 1 == amount_of_images:
                break

        self.save_all_lists(out_pkl_filename)

    def all_image_processing(self, use_dynamic_ds_for_creating_mask=False):
        for idx, image in enumerate(self.images):
            # First I downscale the image in order to reduce the size of the upscale
            ds_image = downscale(image, scale=self.downscale_var)
            dynamic_ds_image = downscale(image, dynamic_max=self.dynamic_max)
            # Then I upscale the image with bilinear upscaling
            us_image = upscale(ds_image, scale=self.upscale_var)
            # Blurring is added to the upscaled image
            blurred_image = gaussian_blur(us_image, radius=self.blur_radius)
            # Adding a mask to the blurred image

            # In order to use either dynamic downscale or upscaled images to add a mask
            images_to_add_mask = None
            if use_dynamic_ds_for_creating_mask:
                images_to_add_mask = dynamic_ds_image
            else:
                images_to_add_mask = us_image

            if self.mask_radius_percentage is not None:
                image_mask = create_mask(images_to_add_mask, radius_img_percentage=self.mask_radius_percentage)
            else:
                image_mask = create_mask(images_to_add_mask)

            # All the different images are added to the storage lists
            self.masked_images.append(image_mask)
            self.blurred_images.append(blurred_image)
            self.downscaled_images.append(ds_image)
            self.dynamic_downscaled_images.append(dynamic_ds_image)
            self.upscaled_images.append(us_image)

            ##########################################################
            # Doing a cheap trick in order not to change the entire code names from upscaled to dynamically downscaled:

            if use_dynamic_ds_for_creating_mask:
                self.white_mask_added_upscaled_images.append(add_white_mask_to_image(dynamic_ds_image, image_mask))
            else:
                # Adding white mask to upscaled images for later comparison with hash difference
                self.white_mask_added_upscaled_images.append(add_white_mask_to_image(us_image, image_mask))
            ##########################################################

            # ORIGINAL CODE:
            # Adding white mask to upscaled images for later comparison with hash difference
            # self.white_mask_added_upscaled_images.append(add_white_mask_to_image(us_image, image_mask))

            self.white_mask_added_blurred_images.append(
                add_white_mask_to_image(blurred_image, create_mask(blurred_image)))
            print(f'Image {idx + 1} of {self.num_of_imgs} processed')

    def save_all_lists(self, out_pkl_filename='inpainting_landscape_diff_disc_images.pkl'):
        all_lists = [self.inpainted_images,
                     self.inpainted_images_diff_disc,
                     self.masked_images,
                     self.downscaled_images,
                     self.dynamic_downscaled_images,
                     self.upscaled_images,
                     self.blurred_images,
                     self.white_mask_added_blurred_images,
                     self.white_mask_added_inpainted_images,
                     self.white_mask_added_upscaled_images,
                     self.white_mask_added_upscaled_disc_diff_images,
                     self.white_mask_added_inpainted_images_diff_disc,
                     self.white_mask_added_pre_inpainted_diff_disc_images,
                     self.disc_area_image_percentages,
                     self.strengths]

        # # Saving pkl file with correct name
        # if self.out_pkl_filename[-4:] == '.pkl':
        #     self.out_pkl_filename = self.out_pkl_filename[:-4]
        # out_pkl_filename = self.out_pkl_filename + f'_n-strengths-{self.n_strengths}_mrp-{self.mask_radius_percentage}'
        # check_if_file_has_pkl_ending(out_pkl_filename)
        save_list(all_lists, out_pkl_filename)

    def extract_all_lists_from_file(self, file_name, old_list_order=False):
        """
        Extracts all the lists from a npy file
        Args:
            old_list_order: If npy file was created before the implementation of disc diff testing
            file_name: The name of the npy file

        Returns: All the lists

        """
        file = load_list(file_name)

        if len(file) == 9:
            old_list_order = True

        if old_list_order:
            self.inpainted_images = file[0]
            self.masked_images = file[1]
            self.downscaled_images = file[2]
            self.upscaled_images = file[3]
            self.blurred_images = file[4]
            self.white_mask_added_blurred_images = file[5]
            self.white_mask_added_inpainted_images = file[6]
            self.white_mask_added_upscaled_images = file[7]
            self.strengths = file[8]

        else:
            self.inpainted_images = file[0]
            self.inpainted_images_diff_disc = file[1]
            self.masked_images = file[2]
            self.downscaled_images = file[3]
            self.dynamic_downscaled_images = file[4]
            self.upscaled_images = file[5]
            self.blurred_images = file[6]
            self.white_mask_added_blurred_images = file[7]
            self.white_mask_added_inpainted_images = file[8]
            self.white_mask_added_upscaled_images = file[9]
            self.white_mask_added_upscaled_disc_diff_images = file[10]
            self.white_mask_added_inpainted_images_diff_disc = file[11]
            self.white_mask_added_pre_inpainted_diff_disc_images = file[12]
            self.disc_area_image_percentages = file[13]
            self.strengths = file[14]

    def export_images_to_folder(self, strength_test=False, diff_disc=False,
                                out_folder_name='inpainting_landscape_images',
                                import_from_pkl_filname=None, old_list_order=False):

        if import_from_pkl_filname is not None:
            self.extract_all_lists_from_file(import_from_pkl_filname, old_list_order=old_list_order)

        # Creating outmost folder
        check_and_create_folder('inpainted_images')
        os.chdir('inpainted_images')

        if strength_test:

            # Separating the inpainted images for strength and diff disc test
            check_and_create_folder('strength_test')
            os.chdir('strength_test')

            out_folder_name = out_folder_name \
                              + f'_n-strengths-{self.n_strengths}_mrp-{self.mask_radius_percentage}'

            # Creating and moving to folder
            check_and_create_folder(out_folder_name)
            os.chdir(out_folder_name)

            for idx, (img_list, img_name) in enumerate(zip(self.inpainted_images, self.image_filenames_list)):

                # Moving to specific image folder
                check_and_create_folder(f'{img_name}')
                os.chdir(f'{img_name}')

                # Creating correct index for image_filename_list
                n_zeros = len(str(len(self.strengths)))

                for idx2, (image, strength) in enumerate(zip(img_list, self.strengths)):
                    save_name = (f'{img_name}{str(idx2).zfill(n_zeros)}_nis{self.num_inference_steps}'
                                 f'_gs{self.guidance_scale}_s{strength}'
                                 f'_mrp{self.mask_radius_percentage}.png')

                    image.save(save_name)

                # Stepping back out for next image
                os.chdir('..')

        elif diff_disc:

            check_and_create_folder('diff_disc_test')
            os.chdir('diff_disc_test')

            n_mask_rads = len(self.disc_area_image_percentages)
            n_strengths = len(self.strengths)
            out_folder_name = out_folder_name + f'_n-mask-rads-{n_mask_rads}_n_strengths{n_strengths}'
            check_and_create_folder(out_folder_name)
            os.chdir(out_folder_name)

            # Creating correct index for image_filename_list
            n_zeros = len(str(len(self.disc_area_image_percentages)))

            for idx, (img_list, img_name) in enumerate(zip(self.inpainted_images_diff_disc, self.image_filenames_list)):

                print(os.getcwd())
                # Moving to specific image folder
                check_and_create_folder(f'{img_name}')
                os.chdir(f'{img_name}')
                print(os.getcwd())

                for strength_list, strength in zip(img_list, self.strengths):

                    check_and_create_folder(f's_{strength}')
                    os.chdir(f's_{strength}')
                    print(os.getcwd())

                    for idx2, (inpainted_img, mask_perc) in enumerate(zip(strength_list,
                                                                          self.disc_area_image_percentages)):
                        save_name = (f'{img_name}{str(idx2).zfill(n_zeros)}_nis{self.num_inference_steps}'
                                     f'_gs{self.guidance_scale}_s{strength}'
                                     f'_mrp{mask_perc}.png')

                        inpainted_img.save(save_name)

                    # Stepping back out for next image
                    os.chdir('..')
                    print(os.getcwd())
                os.chdir('..')
                print(os.getcwd())

    def save_all(self):

        if self.import_from_file:
            check_if_file_has_pkl_ending(self.load_pkl_filename)
            self.extract_all_lists_from_file(self.load_pkl_filename)

        # For saving the inpainted images as png in separate folders
        # Making sure to cut down image_filename_list in case of user input
        self.image_filenames_list = self.image_filenames_list[:self.amount_of_images]

        for idx, (img_list, img_name) in enumerate(zip(self.inpainted_images, self.image_filenames_list)):
            out_folder_name = self.out_images_folder \
                              + f'_n-strengths-{self.n_strengths}_mrp-{self.mask_radius_percentage}' \
                if self.mask_radius_percentage is not None else self.out_images_folder

            check_and_create_folder(out_folder_name)
            check_and_create_folder(f'{out_folder_name}/{img_name}')
            os.chdir(f'{out_folder_name}/{img_name}')

            # Creating correct index for image_filename_list
            n_zeros = len(str(len(self.strengths)))

            for idx2, (image, strength) in enumerate(zip(img_list, self.strengths)):
                save_name = (f'{img_name}{str(idx2).zfill(n_zeros)}_nis{self.num_inference_steps}'
                             f'_gs{self.guidance_scale}_s{strength}'
                             f'_mrp{self.mask_radius_percentage}.png')

                image.save(save_name)

            if idx + 1 == self.amount_of_images:
                break

            # Stepping back out for next image
            os.chdir('../..')

        print('Saving all_lists.pkl')
        # For some reason still in 'inpainted image directory', have to step out again outside for-loop
        os.chdir('../..')

        # Saving pkl file with correct name
        if self.out_pkl_filename[-4:] == '.pkl':
            self.out_pkl_filename = self.out_pkl_filename[:-4]
        out_pkl_filename = self.out_pkl_filename + f'_n-strengths-{self.n_strengths}_mrp-{self.mask_radius_percentage}'
        check_if_file_has_pkl_ending(out_pkl_filename)

        self.save_all_lists(out_pkl_filename)

    def plot_image_differences_by_strength(self, import_from_file=False,
                                           load_pkl_filename='',
                                           number_of_images=None,
                                           plots_folder_name=None,
                                           big_subplot_figure=False,
                                           use_blur=False,
                                           avg_hash=False,
                                           phash=False,
                                           use_ssd=False,
                                           use_mse=True):

        if import_from_file:
            check_if_file_has_pkl_ending(load_pkl_filename)
            self.extract_all_lists_from_file(load_pkl_filename)
        print(os.getcwd())

        if number_of_images == 'all':
            number_of_images = self.num_of_imgs

        check_and_create_folder('plots')
        os.chdir('plots')

        if plots_folder_name is not None:
            check_and_create_folder(plots_folder_name)
            os.chdir(plots_folder_name)

        # For changing between blurred and only upscaled images and changing plot title
        plot_title = ''
        if use_blur:
            images_to_compare = self.white_mask_added_blurred_images
            if avg_hash:
                plot_title = 'Hash difference between blurred and inpainted images'
            if phash:
                plot_title = 'Perceptual Hash difference between blurred and inpainted images'
            if use_ssd:
                plot_title = 'SSD difference between blurred and inpainted images'
            if use_mse:
                plot_title = 'MSE difference between blurred and inpainted images'

        else:
            images_to_compare = self.white_mask_added_upscaled_images
            if avg_hash:
                plot_title = 'Hash difference between upscaled and inpainted images'
            if phash:
                plot_title = 'Perceptual Hash difference between upscaled and inpainted images'
            if use_ssd:
                plot_title = 'SSD difference between upscaled and inpainted images'
            if use_mse:
                plot_title = 'MSE difference between downscaled and inpainted images'

        # The loop whitch actually plots the differences
        all_figs = []

        # Preparing subplots

        fig, axs = plt.subplots(5, 4, sharex=True, sharey=False)
        axs = axs.ravel()
        for idx, (filename, image_to_compare, inpainted_image_list) \
                in enumerate(zip(self.image_filenames_list,
                                 images_to_compare,
                                 self.white_mask_added_inpainted_images)):

            print('Creating plot for image: ', filename)

            if avg_hash or phash:
                # Calculating hash differences
                all_hash_diffs = []
                for inpainted_image in inpainted_image_list:
                    if phash:
                        all_hash_diffs.append(imagehash_diff(image_to_compare, inpainted_image, phash=phash))
                    else:
                        all_hash_diffs.append(imagehash_diff(image_to_compare, inpainted_image))

                # Plotting
                plt.figure()
                plt.plot(self.strengths, all_hash_diffs)
                plt.xlabel('Strength')
                plt.ylabel('Hash difference')
                plt.title(f'{plot_title} for {filename}', loc='center', wrap=True)
                plt.savefig(f'{filename}_hash_diff.png')

            if use_ssd:
                # Calculating sum of squared differences
                all_ssd = []
                for inpainted_image in inpainted_image_list:
                    all_ssd.append(sum_of_squared_diff(image_to_compare, inpainted_image))

                # Plotting
                plt.figure()
                plt.plot(self.strengths, all_ssd)
                plt.xlabel('Strength')
                plt.ylabel('Sum of squared differences')
                plt.title(f'{plot_title} for {filename}', loc='center', wrap=True)
                plt.savefig(f'{filename}_ssd.png')

            if use_mse:
                # Calculating mean squared error

                # Want to create a list which only contains non inpainted pixels

                masked_image = np.asarray(self.masked_images[idx])
                black_pixel_idxs_from_mask_image = np.where(masked_image == 0, True, False)
                relevant_pixels_image_to_compare = np.asarray(image_to_compare)[black_pixel_idxs_from_mask_image]

                all_mse = []

                for inpainted_image in inpainted_image_list:
                    relevant_pixels_inpainted_image = np.asarray(inpainted_image)[black_pixel_idxs_from_mask_image]
                    all_mse.append(average_mse(relevant_pixels_image_to_compare,
                                               relevant_pixels_inpainted_image))

                    # all_mse.append(average_mse(image_to_compare, inpainted_image))

                if not big_subplot_figure:
                    # Plotting
                    plt.figure()
                    plt.plot(self.strengths, all_mse)
                    plt.xlabel('Strength')
                    plt.ylabel('Mean squared error')
                    plt.title(f'{filename}', loc='center', wrap=True)
                    plt.savefig(f'{filename}_mse.png')
                    plt.close()

                if big_subplot_figure:
                    ax = axs[idx]
                    ax.xaxis.set_tick_params(labelsize=5)
                    ax.plot(self.strengths, all_mse, linewidth=0.5)
                    ax.set_title(f"{filename}", fontsize=7)
                    plt.subplots_adjust(hspace=0.4)


            if isinstance(number_of_images, int):
                if idx + 1 == number_of_images:
                    break

        # Saving the big subplot figure
        if big_subplot_figure:
            fig.text(0.5, 0.05, 'Strength', ha='center', va='center', fontsize=12)
            fig.text(0.04, 0.5, 'Mean squared error', ha='center', va='center', rotation='vertical', fontsize=12)
            fig.savefig('big_subplot_figure.png', dpi=500, bbox_inches='tight')
            fig.show()


    def plot_diff_disc(self, load_pkl_filename='', out_folder_name='', dynamic_ds=False, heatmap=False):
        """
        Plots for each image mse of pixel change outside the mask, for each strength. MSE on y-axis and
        disc_area/image_area on the x-axis
        """

        self.extract_all_lists_from_file(load_pkl_filename)

        # Need to find all images radiuses used

        if dynamic_ds:
            images_to_compare = self.dynamic_downscaled_images
        else:
            images_to_compare = self.upscaled_images

        all_radiuses = []
        for image in images_to_compare:
            temp_rads = []
            for percentage in self.disc_area_image_percentages:
                temp_rads.append(create_mask(image, radius_img_percentage=percentage, return_radius=True))
            all_radiuses.append(temp_rads)

        os.chdir('plots')
        check_and_create_folder('disc_diff_plots')
        os.chdir('disc_diff_plots')

        check_and_create_folder(out_folder_name)
        os.chdir(out_folder_name)

        # In order to pick out the images with the biggest difference
        each_image_most_diff = []

        for list1, w_list, filename in zip(self.white_mask_added_inpainted_images_diff_disc,
                                           self.white_mask_added_pre_inpainted_diff_disc_images,
                                           self.image_filenames_list):
            all_x = []
            all_y = []

            for list2, rad_list in zip(list1, all_radiuses):

                x = []
                y = []
                diff_temp_list = []
                for w_inpainted_image, w_upscaled_image, rad in zip(list2, w_list, rad_list):
                    # For error testing
                    # w_inpainted_image.show()
                    # w_upscaled_image.show()

                    img_w, img_h = w_inpainted_image.size

                    img_area = img_w * img_h
                    disc_area = np.pi * rad ** 2
                    disc_area_image_percentage = disc_area / img_area

                    n_pixels = img_area - disc_area
                    mse_div_npixels = average_mse(w_inpainted_image, w_upscaled_image) / n_pixels
                    diff_temp_list.append(mse_div_npixels)
                    x.append(disc_area_image_percentage)
                    y.append(mse_div_npixels)
                all_x.append(x)
                all_y.append(y)
                each_image_most_diff.append([np.max(diff_temp_list), np.argmax(diff_temp_list)])

            plt.figure()
            for x, y, strength in zip(all_x, all_y, self.strengths):
                plt.plot(x, y, label=f'strength {strength}')
                plt.xlabel('Disc area / image area')
                plt.ylabel('MSE / number of pixels outside disc')
                plt.title(f'Pixel change for {filename}', loc='center', wrap=True)
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
                plt.tight_layout()
                plt.legend()

            plt.savefig(f'{filename}_disc_diff.png')


if __name__ == '__main__':
    n_strengths = 100
    mask_radius_percentage = 80

    c = InpaintingStrengthExperiment(input_img_folder_path='landscape_images',
                                     n_strengths=100,
                                     mask_radius_percentage=80,
                                     need_for_image_processing=True)

    # c.all_inpainting(amount_of_images='all',
    #                  out_pkl_filename='only_landscape_12_inpainted_strength.pkl',
    #                  use_dynamic_downscale=True,
    #                  testing=False)

    c.all_inpainting_diff_disc(amount_of_images='all',
                               out_pkl_filename='landscape_images_diff_disc_FINAL.pkl',
                               testing=False,
                               dynamic_ds=True,
                               n_masks=50)

    # c.export_images_to_folder(out_folder_name='landscape_images_100_strengths_80_mrp_images',
    #                           import_from_pkl_filname='landscape_images_100_strengths_80_mrp.pkl',
    #                           diff_disc=False,
    #                           strength_test=True,
    #                           old_list_order=False
    #                           )

    # c.plot_diff_disc(load_pkl_filename='landscape_v3_images_diff_disc_test_final.pkl',
    #                  out_folder_name='landscape_v3_images_diff_disc_test_final_debug',
    #                  dynamic_ds=True)

    # c.plot_image_differences_by_strength(import_from_file=True,
    #                                      load_pkl_filename='landscape_images_100_strengths_80_mrp_fix.pkl',
    #                                      number_of_images='all',
    #                                      use_blur=False,
    #                                      plots_folder_name=f'landscape_images_mse_diff'
    #                                                        f'_s{n_strengths}_mrp{mask_radius_percentage}_plots',
    #                                      big_subplot_figure=False,
    #                                      use_ssd=False,
    #                                      avg_hash=False,
    #                                      phash=False,
    #                                      use_mse=True)

