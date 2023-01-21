import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from inpainting_strength_testing_utils import *
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
import sys

load_mse = True

pkl_path = 'inpainting_strength_testing_files/landscape_images_100_strengths_80_mrp_fix.pkl'

org_working_dir = os.getcwd()

pkl_list = load_list(pkl_path)
inpainted_images = pkl_list[0]
masked_images = pkl_list[2]
downscaled_images = pkl_list[4]

if not load_mse:
    all_mse = []
    all_top_mse_idx = []

    # Finding all mse and appending to list
    for inpainted_list, mask, downscaled_image in zip(inpainted_images, masked_images, downscaled_images):
        temp_mse_list = []
        for inpainted_image in inpainted_list:
            # ignoring the masked area for accurate mse
            outside_mask_idx = np.where(np.asarray(mask) == 0)
            relevant_inpainted_pixels = np.asarray(inpainted_image)[outside_mask_idx]
            relevant_downscaled_pixels = np.asarray(downscaled_image)[outside_mask_idx]

            # Calculating mse
            mse = np.mean((relevant_inpainted_pixels - relevant_downscaled_pixels) ** 2)
            temp_mse_list.append(mse)
        all_mse.append(temp_mse_list)

    for mse_list in all_mse:
        # Find the indexes of the three highest mse
        top_mse_idx = np.argsort(mse_list)[-3:]
        all_top_mse_idx.append(top_mse_idx[::-1])

    # Saving the mse and top mse idx
    save_list(all_top_mse_idx, 'inpainting_strength_testing_files/all_top_mse_idx.pkl')

if load_mse:
    all_top_mse_idx = load_list('inpainting_strength_testing_files/all_top_mse_idx.pkl')


# %%
def heatmap_diff(image1, image2, mask, savename='', horizontal_colorbar=False):
    # Convert images to grayscale
    image1_gray = rgb2gray(image1)
    image2_gray = rgb2gray(image2)

    mask = np.array(mask)
    mask_three_channels = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    # Use where the mask is white to add the same white to the images
    image1_gray = np.where(mask == 255, 255, image1_gray)
    image2_gray = np.where(mask == 255, 255, image2_gray)
    image2_with_mask = np.where(mask_three_channels == 255, 255, np.array(image2))

    # Calculate pixel differences
    diff = np.abs(image1_gray - image2_gray)

    # Normalize the pixel differences
    diff_norm = (diff - diff.min()) / (diff.max() - diff.min())

    # Create the heatmap
    plt.figure()
    plt.axis('off')
    plt.imshow(image2_with_mask, alpha=1)
    plt.imshow(diff_norm, cmap='hot', alpha=0.5)

    # create the axes for the colorbar
    divider = make_axes_locatable(plt.gca())

    # create the colorbar
    if horizontal_colorbar:
        cax = divider.append_axes("bottom", size="5%", pad=0.05)
        cbar = plt.colorbar(cax=cax, orientation='horizontal')
        ticks = cbar.get_ticks()
        cbar.set_ticks(ticks[1:-1])
    else:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(cax=cax)
        ticks = cbar.get_ticks()
        cbar.set_ticks(ticks[1:-1])

    plt.savefig(f'{savename}', dpi=600, bbox_inches='tight', pad_inches=0)
    plt.close()


# Testing heatmap function with most mse image from the landscape 0
test_ds_img = downscaled_images[0]
test_inpainted_img = inpainted_images[0][all_top_mse_idx[0][2]]
mask = masked_images[0]
heatmap_diff(test_ds_img, test_inpainted_img, mask)

# %%

heatmaps_folder_name = 'inpainting_strength_testing_files/only_heatmaps_horizontal_colorbar'

if os.getcwd() == org_working_dir:
    check_and_create_folder(heatmaps_folder_name)
    os.chdir(heatmaps_folder_name)
else:
    print('Not in the correct directory')
    print('Current directory: ', os.getcwd())
    print('Expected directory: ', org_working_dir)
    sys.exit()

# %%
# Want to create heatmap and store every image
use_subfolders = False
for idx, top_mse_idx_list in enumerate(all_top_mse_idx):

    subfolder_name = f'landscape_{idx}'

    if use_subfolders:
        check_and_create_folder(subfolder_name)
        os.chdir(subfolder_name)

    ds_img = downscaled_images[idx]
    mask = masked_images[idx]
    # ds_img.save(f'{subfolder_name}_downscaled_img.png')
    for sub_idx, top_idx in enumerate(top_mse_idx_list):
        print(os.getcwd())

        inpainted_img = inpainted_images[idx][top_idx]
        # inpainted_img.save(f'{subfolder_name}_inpainted_img_{sub_idx}_strength{top_idx}.png')

        heatmap_diff(ds_img, inpainted_img, mask,
                     savename=f'{subfolder_name}_heatmap_{sub_idx}_strength{top_idx}.png',
                     horizontal_colorbar=True)

        print(f'Finished heatmap for landscape {idx} and image {sub_idx}')
    print(os.getcwd())
