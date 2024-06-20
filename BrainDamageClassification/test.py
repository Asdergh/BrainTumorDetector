import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import json as js
import os 


class WaterSheedAlgorithm:

    def __init__(self, image_data_path=None, image_data_tensor=None):

        self.image_data_path = image_data_path
        self.image_data_tensor = image_data_tensor
    
    def calculate_segments(self, result_images_path):
        
        self.result_image_collection = []
        if self.image_data_path is None:

            self.image_data_tensor = []
            for image_file in os.listdir(self.image_data_path):

                curent_image_path = os.path.join(self.image_data_path, image_file)
                image = cv2.imread(curent_image_path)
                for pix_x in range(image.shape[0]):
                    for pix_y in range(image.shape[1]):

                        if np.mean(image[pix_x, pix_y]) > np.mean(image):
                            image[pix_x, pix_y][0] += 50.0
                        
                        else:
                            image[pix_x, pix_y][2] += 50.0
                
                self.result_image_collection.append(image)
        
        elif self.image_data_tensor is not None:

            self.image_data_tensor = []
            for image in self.image_data_tensor:

                for pix_x in range(image.shape[0]):
                    for pix_y in range(image.shape[1]):

                        if np.mean(image[pix_x, pix_y]) > np.mean(image):
                            image[pix_x, pix_y][0] += 50.0
                        
                        else:
                            image[pix_x, pix_y][2] += 50.0
                
                self.result_image_collection.append(image)
        
        else:
            raise BaseException("need path to data or datatensor")

        for (image_number, image) in enumerate(self.result_image_collection):

            curent_image_path = os.path.join(result_images_path, 
                                             f"image_number_{image_number}.jpeg")
            cv2.imwrite(curent_image_path, image)






        


            
                            
