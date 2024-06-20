import numpy as np
import os
import json as js
import cv2
import random as rd


class DataHandler():

    def __init__(self, data_dir, samples_count, image_size) -> None:
        

        self.data_dir = data_dir
        self.samples_count = samples_count
        self.class_types = ["glioma", "meningioma"]
        self.image_size = image_size
    
    def _load_data(self):



        self.data_collection = {}
        
        self.train_data_dir = os.path.join(self.data_dir, "Testing")
        self.test_data_dir = os.path.join(self.data_dir, "Training")

        self.train_notumor_data_dir = os.path.join(self.train_data_dir, "notumor")
        self.test_notumor_data_dir = os.path.join(self.test_data_dir, "notumor")

        self.train_glioma_data_dir = os.path.join(self.train_data_dir, "glioma")
        self.test_glioma_data_dir = os.path.join(self.test_data_dir, "glioma")

        self.train_meningioma_data_dir = os.path.join(self.train_data_dir, "meningioma")
        self.test_meningioma_data_dir = os.path.join(self.test_data_dir, "meningioma")

        self.train_collection = [self.train_glioma_data_dir, self.train_meningioma_data_dir]
        self.test_collection = [self.test_glioma_data_dir, self.test_meningioma_data_dir]

        
        

        curent_start_point = 0
        curent_end_point = curent_start_point + self.samples_count
        class_types_dirs = [(train_notumor, test_notumor, trian_glioma, test_glioma, train_meningioma, test_meningioma)
                            for (train_notumor, test_notumor, trian_glioma, test_glioma, train_meningioma, test_meningioma) in
                                zip(
                                    os.listdir(self.train_notumor_data_dir)[curent_start_point: curent_end_point],
                                    os.listdir(self.test_notumor_data_dir)[curent_start_point: curent_end_point],
                                    os.listdir(self.train_glioma_data_dir)[curent_start_point: curent_end_point],
                                    os.listdir(self.test_glioma_data_dir)[curent_start_point: curent_end_point],
                                    os.listdir(self.train_meningioma_data_dir)[curent_start_point: curent_end_point],
                                    os.listdir(self.test_meningioma_data_dir)[curent_start_point: curent_end_point]
                                   
                                )]
        
        for (class_number, class_type) in enumerate(self.class_types):
            
            self.data_collection[f"train_{class_type}_dataset"] = []
            self.data_collection[f"test_{class_type}_dataset"] = []
            
            for image_number in range(self.samples_count):
                
                

                if rd.randint(0, 100) < 50.0:
                        

                    train_tumor_path = os.path.join(self.train_collection[class_number], class_types_dirs[image_number][class_number + 2])
                    test_tumor_path = os.path.join(self.test_collection[class_number], class_types_dirs[image_number][class_number + 3])
                        
                    train_tumor_image = cv2.imread(train_tumor_path)
                    test_tumor_image = cv2.imread(test_tumor_path)
                        
                    if (train_tumor_image is not None) and (test_tumor_image is not None):
                        
                    
                        train_tumor_image = np.asarray(cv2.resize(train_tumor_image, (self.image_size[1], self.image_size[0])), dtype="float")
                        test_tumor_image = np.asarray(cv2.resize(test_tumor_image, (self.image_size[1], self.image_size[0])), dtype="float")

                        train_tumor_image += np.random.normal(0, 6.23, (train_tumor_image.shape[0], train_tumor_image.shape[1], 3))
                        test_tumor_image += np.random.normal(0, 6.23, (test_tumor_image.shape[0], test_tumor_image.shape[1], 3))

                        std_train_tumor_image = (train_tumor_image - np.mean(train_tumor_image)) / np.std(train_tumor_image)
                        std_test_tumor_image = (train_tumor_image - np.mean(test_tumor_image)) / np.std(test_tumor_image)

                        self.data_collection[f"train_{class_type}_dataset"].append((std_train_tumor_image, 0))
                        self.data_collection[f"test_{class_type}_dataset"].append((std_test_tumor_image, 0))
                    
                    else:
                        
                        
                        if len(self.data_collection[f"train_{class_type}_dataset"]) != 0:

                            std_train_tumor_image = (self.data_collection[f"train_{class_type}_dataset"][-1][0] 
                                                     - np.mean(self.data_collection[f"train_{class_type}_dataset"][-1][0])) \
                                                     /np.std(self.data_collection[f"train_{class_type}_dataset"][-1][0])
                            
                            self.data_collection[f"train_{class_type}_dataset"].append((std_train_tumor_image, 1))
                        
                        else:
                            self.data_collection[f"train_{class_type}_dataset"].append((np.random.normal(0, 12.3, (self.image_size[0], self.image_size[1], 3)), 1))
                        
                        if len(self.data_collection[f"test_{class_type}_dataset"]) != 0:
                            
                            std_test_tumor_image = (self.data_collection[f"test_{class_type}_dataset"][-1][0] 
                                                     - np.mean(self.data_collection[f"test_{class_type}_dataset"][-1][0])) \
                                                     /np.std(self.data_collection[f"test_{class_type}_dataset"][-1][0])
                            
                            self.data_collection[f"test_{class_type}_dataset"].append((std_test_tumor_image, 1))

                        else:
                            self.data_collection[f"test_{class_type}_dataset"].append((np.random.normal(0, 12.3, (self.image_size[0], self.image_size[1], 3)), 1))
                       



                        
                else:
                        

                    train_notumor_path = os.path.join(self.train_notumor_data_dir, class_types_dirs[image_number][0])
                    test_notumor_path = os.path.join(self.test_notumor_data_dir, class_types_dirs[image_number][1])

                    train_notumor_image = cv2.imread(train_notumor_path)
                    test_notumor_image = cv2.imread(test_notumor_path)

                    if (train_notumor_image is not None ) and (test_notumor_image is not None):

                        train_notumor_image = np.asarray(cv2.resize(train_notumor_image, (self.image_size[1], self.image_size[0])), dtype="float")
                        test_notumor_image = np.asarray(cv2.resize(test_notumor_image, (self.image_size[1], self.image_size[0])), dtype="float")

                        train_notumor_image += np.random.normal(0, 6.23, (train_notumor_image.shape[0], train_notumor_image.shape[1], 3))
                        test_notumor_image += np.random.normal(0, 6.23, (test_notumor_image.shape[0], test_notumor_image.shape[1], 3))

                        self.data_collection[f"train_{class_type}_dataset"].append((train_notumor_image, 1))
                        self.data_collection[f"test_{class_type}_dataset"].append((test_notumor_image, 1))
                    
                    else:

                        self.data_collection[f"train_{class_type}_dataset"].append((np.random.normal(0, 12.3, (self.image_size[0], self.image_size[1], 3)), 1))
                        self.data_collection[f"test_{class_type}_dataset"].append((np.random.normal(0, 12.3, (self.image_size[0], self.image_size[1], 3)), 1))
                    
                
    def _generate_and_permutate_data(self, class_name):

        result_list = self.data_collection[class_name]
        choosed_indexes = []
        permutated_result_list = []

        while len(permutated_result_list) != len(result_list):

            random_index = rd.randint(0, len(result_list) - 1)
            if random_index in choosed_indexes:
                continue
                
            else:
                permutated_result_list.append(result_list[random_index])
        
        separation_test_data_tensor = np.asarray([sample[0] for sample in permutated_result_list])
        separation_test_data_labels = np.asarray([sample[1] for sample in permutated_result_list])

        return (separation_test_data_tensor, separation_test_data_labels)

    def _formulate_data(self):

        self.train_glioma_dataset = self._generate_and_permutate_data("train_glioma_dataset")
        self.test_glioma_dataset = self._generate_and_permutate_data("test_glioma_dataset")

        self.train_meningioma_dataset = self._generate_and_permutate_data("train_meningioma_dataset")
        self.test_meningioma_dataset = self._generate_and_permutate_data("test_meningioma_dataset")

        return self.train_glioma_dataset, self.test_glioma_dataset, self.train_meningioma_dataset, self.test_meningioma_dataset


    def _trashholed_sample(self, sample_to_trashholed):

        copy_sample = sample_to_trashholed

        copy_sample_spector_one = copy_sample[:, :, 0]
        copy_sample_spector_two = copy_sample[:, :, 1]
        copy_sample_spector_three = copy_sample[:, :, 2]

        copy_sample_spector_one[copy_sample_spector_one < np.mean(copy_sample_spector_one)] = 0.0
        copy_sample_spector_two[copy_sample_spector_one < np.mean(copy_sample_spector_two)] = 0.0
        copy_sample_spector_three[copy_sample_spector_one < np.mean(copy_sample_spector_two)] = 0.0

        copy_sample[:, :, 0] = copy_sample_spector_one
        copy_sample[:, :, 1] = copy_sample_spector_two
        copy_sample[:, :, 2] = copy_sample_spector_three

        return copy_sample
    
    
    def _formulate_data_for_separation(self):

        self.separation_train_data_tensor = []
        self.separation_train_data_labels = []
        self.separation_test_data_tensor = []
        self.separation_test_data_labels = []

        self.train_mean_glioma_mean_value = np.asarray([sample_image for sample_image in self.train_glioma_dataset[0]]).mean()
        self.test_mean_glioma_mean_value = np.asarray([sample_image for sample_image in self.test_glioma_dataset[0]]).mean()
        self.train_mean_meningioma_mean_value = np.asarray([sample_image for sample_image in self.train_meningioma_dataset[0]]).mean()
        self.test_mean_meningioma_mean_value = np.asarray([sample_image for sample_image in self.test_meningioma_dataset[0]]).mean()
        
        self.result_mean_glioma_value = (self.train_mean_glioma_mean_value + self.test_mean_glioma_mean_value) / 2.0
        self.result_mean_meningioma_value = (self.train_mean_meningioma_mean_value + self.test_mean_meningioma_mean_value) / 2.0
        self.result_mean_value = (self.result_mean_glioma_value + self.result_mean_meningioma_value) / 2.0

        for (train_glioma_sample, train_meningioma_sample, 
             test_glioma_sample, test_meningioma_sample) in zip(self.train_glioma_dataset[0], self.train_meningioma_dataset[0],
                                                                self.test_glioma_dataset[0], self.test_meningioma_dataset[0]):

            train_trashholede_glioma = self._trashholed_sample(train_glioma_sample)
            train_trashholede_meningioma = self._trashholed_sample(train_meningioma_sample)
            test_trashholede_glioma = self._trashholed_sample(test_glioma_sample)
            test_trashholede_meningioma = self._trashholed_sample(test_meningioma_sample)

            self.separation_train_data_tensor.append(train_trashholede_glioma)
            self.separation_train_data_tensor.append(train_trashholede_meningioma)
            self.separation_test_data_tensor.append(test_trashholede_glioma)
            self.separation_test_data_tensor.append(test_trashholede_meningioma)
            
            self.separation_train_data_labels.append(np.sum(train_trashholede_glioma))
            self.separation_train_data_labels.append(np.sum(train_trashholede_meningioma))

            self.separation_test_data_labels.append(np.sum(test_trashholede_glioma))
            self.separation_test_data_labels.append(np.sum(test_trashholede_meningioma))


        return (self.separation_train_data_tensor, self.separation_train_data_labels), \
                (self.separation_test_data_tensor, self.separation_test_data_labels)



        
        
    
    
    

        
        

                




        

    


                