import os
import zipfile
import numpy as np
import math
import cv2
from tqdm import tqdm

from scipy import misc
from skimage import color
from urllib.request import urlretrieve


DATA_PATH = "/Set/To/Data/Path"

class TrainSet:
    def __init__(self, benchmark, batch_size=64, patch_size=41, scaling_factors=(2, 4, 8)):
        self.benchmark = benchmark
        self.batch_size = batch_size
        self.patch_size = patch_size
        self.scaling_factors = scaling_factors
        self.images_completed = 0
        self.epochs_completed = 0
        self.root_path = os.path.join(DATA_PATH, 'TRAIN_SUBSET', self.benchmark)
        self.images = []
        self.targets = []


        for file_name in os.listdir(self.root_path):
            #Read in image
            image = misc.imread(os.path.join(self.root_path, file_name))
            #Crop to an area divisible by 12
            width, height = image.shape[0], image.shape[1]
            width = width - width % 12
            height = height - height % 12
            n_horizontal_patches = width // patch_size
            n_vertical_patches = height // patch_size
            image= image[:width,:height]
            
            #For each level of enhacement
            for scaling_factor in scaling_factors:
                #Conditional blur
                blur_level=scaling_factor/2
                blurred = cv2.GaussianBlur(image, (0, 0), blur_level, blur_level, 0)
                #Pull out the luminance component of ycbcr for the HR and blurred images
                if len(image.shape) == 3:
                    blurred = color.rgb2ycbcr(blurred)[:, :, 0].astype(np.uint8)
                    image = color.rgb2ycbcr(image)[:, :, 0].astype(np.uint8)
                
                
                
                #downscale the blurred component
                downscaled=cv2.resize(blurred, (0,0), fx=float(1 / scaling_factor),fy=float(1 / scaling_factor), interpolation=cv2.INTER_AREA)
                #rescale the blurred component
                rescaled = misc.imresize(downscaled, (image.shape[0],image.shape[1]), 'bicubic', mode='L')
                #Save the luminance component of the original image as an HR target
                high_res_image = image.astype(np.float32) / 255
                #Save the blurred, downscaled/rescaled as a LR target
                low_res_image = np.clip(rescaled.astype(np.float32) / 255, 0.0, 1.0)
                
                #Create patches and data aug for training
                for horizontal_patch in range(n_horizontal_patches):
                    for vertical_patch in range(n_vertical_patches):
                        h_start = horizontal_patch * patch_size
                        v_start = vertical_patch * patch_size
                        high_res_patch = high_res_image[h_start:h_start + patch_size, v_start:v_start + patch_size]
                        low_res_patch = low_res_image[h_start:h_start + patch_size, v_start:v_start + patch_size]

                        for _ in range(4):
                            high_res_patch = np.rot90(high_res_patch)
                            low_res_patch = np.rot90(low_res_patch)

                            self.targets.append(np.expand_dims(high_res_patch, axis=2))
                            self.images.append(np.expand_dims(low_res_patch, axis=2))

                        high_res_patch = np.fliplr(high_res_patch)
                        low_res_patch = np.fliplr(low_res_patch)

                        for _ in range(4):
                            high_res_patch = np.rot90(high_res_patch)
                            low_res_patch = np.rot90(low_res_patch)

                            self.targets.append(np.expand_dims(high_res_patch, axis=2))
                            self.images.append(np.expand_dims(low_res_patch, axis=2))

        self.images = np.array(self.images)
        self.targets = np.array(self.targets)

        self.shuffle()
        self.length = len(self.images)
        self.length = self.length - self.length % batch_size
        self.images = self.images[:self.length]
        self.targets = self.targets[:self.length]

    def batch(self):
        images = self.images[self.images_completed:(self.images_completed + self.batch_size)]
        targets = self.targets[self.images_completed:(self.images_completed + self.batch_size)]

        self.images_completed += self.batch_size

        if self.images_completed >= self.length:
            self.images_completed = 0
            self.epochs_completed += 1
            self.shuffle()

        return images, targets

    def shuffle(self):
        indices = list(range(len(self.images)))
        np.random.shuffle(indices)

        self.images = self.images[indices]
        self.targets = self.targets[indices]



class TestSet:
    def __init__(self, benchmark, scaling_factors=(2, 4, 8)):
        self.benchmark = benchmark
        self.scaling_factors = scaling_factors
        self.images_completed = 0
        self.root_path = os.path.join(DATA_PATH, 'TEST', self.benchmark)
        self.file_names = os.listdir(self.root_path)
        self.images = []
        self.targets = []

        for file_name in tqdm(os.listdir(self.root_path)):
            image = misc.imread(os.path.join(self.root_path, file_name))

            #For each enhancement level...
            for scaling_factor in self.scaling_factors:
                #Conditional Blur
                blur_level=scaling_factor/2
                blurred = cv2.GaussianBlur(image, (0, 0), blur_level, blur_level, 0)             
                
                if len(image.shape) == 3:
                    #Pull out all the original ycbcr components
                    ycbcr = color.rgb2ycbcr(blurred)
                    y = ycbcr[:, :, 0].astype(np.uint8)
                    b = ycbcr[:, :, 1].astype(np.uint8)
                    r = ycbcr[:, :, 2].astype(np.uint8)
                else:
                    y = blurred
                    
                #Downscale them   
                downscaled=cv2.resize(y, (0,0), fx=float(1 / scaling_factor),fy=float(1 / scaling_factor), interpolation=cv2.INTER_AREA)
                d_b=cv2.resize(b, (0,0), fx=float(1 / scaling_factor),fy=float(1 / scaling_factor), interpolation=cv2.INTER_AREA)
                d_r=cv2.resize(r, (0,0), fx=float(1 / scaling_factor),fy=float(1 / scaling_factor), interpolation=cv2.INTER_AREA)
                
                #rescale them
                rescaled = misc.imresize(downscaled, (y.shape[0],y.shape[1]), 'bicubic', mode='L')
                r_b = misc.imresize(d_b, (y.shape[0],y.shape[1]), 'bicubic', mode='L')
                d_r = misc.imresize(d_r, (y.shape[0],y.shape[1]), 'bicubic', mode='L')

                #Create the LR image to convert to HR
                if len(image.shape) == 3:
                    low_res_image = ycbcr
                    low_res_image[:, :, 0] = rescaled
                    low_res_image[:, :, 1] = r_b
                    low_res_image[:, :, 2] = d_r
                    low_res_image = color.ycbcr2rgb(low_res_image)
                    low_res_image = (np.clip(low_res_image, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    low_res_image = rescaled

                self.images.append(low_res_image)
                self.targets.append(image)

        self.length = len(self.images)

    def fetch(self):
        if self.images_completed >= self.length:
            return None
        else:
            self.images_completed += 1

            return self.images[self.images_completed - 1], self.targets[self.images_completed - 1]


class SR_Run:
    def __init__(self, benchmark, scaling_factors=(2, 4, 8)):
        self.benchmark = benchmark
        self.scaling_factors = scaling_factors
        self.images_completed = 0
        self.root_path = os.path.join(DATA_PATH, self.benchmark)
        self.file_names = os.listdir(self.root_path)
        self.images = []
        self.targets = []

        for file_name in tqdm(os.listdir(self.root_path)):
            image = misc.imread(os.path.join(self.root_path, file_name))

            for scaling_factor in self.scaling_factors:
                if len(image.shape) == 3:
                    ycbcr = color.rgb2ycbcr(image)
                    downscaled = ycbcr[:, :, 0].astype(np.uint8)
                    d_b = ycbcr[:, :, 1].astype(np.uint8)
                    d_r = ycbcr[:, :, 2].astype(np.uint8)
                else:
                    y = image

                rescaled = misc.imresize(downscaled, float(scaling_factor), 'bicubic', mode='L')
                r_b = misc.imresize(d_b, float(scaling_factor), 'bicubic', mode='L')
                d_r = misc.imresize(d_r, float(scaling_factor), 'bicubic', mode='L')


                if len(image.shape) == 3:
                    low_res_image = np.stack([rescaled,r_b,d_r],axis=2)
                    low_res_image=low_res_image.astype(np.float64)
                    low_res_image = color.ycbcr2rgb(low_res_image)
                    low_res_image = (np.clip(low_res_image, 0.0, 1.0) * 255).astype(np.uint8)
                else:
                    low_res_image = rescaled

                self.images.append(low_res_image)
                self.targets.append(image)

        self.length = len(self.images)

    def fetch(self):
        if self.images_completed >= self.length:
            return None
        else:
            self.images_completed += 1

            return self.images[self.images_completed - 1], self.targets[self.images_completed - 1]
