import numpy as np
import pandas as pd
import os
import json as js
import tensorflow as tf
import matplotlib.pyplot as plt
import random as rd
import plotly.graph_objects as go

from DataHandler import DataHandler
from ConcatenativeModel import ConcatenativeModel
from ParallelModel import ParallelModel






class ModelKernel(DataHandler):

    def __init__(self, model, data_dir, samples_count, image_size) -> None:
        
        super().__init__(data_dir, samples_count, image_size)
        self.model = model
    
        
    def _extract_data(self, dataset_one=None, dataset_two=None):

        if (dataset_one is None) and (dataset_two is not None):

                (self.train_glioma_data_tensor, self.test_glioma_data_tensor) = (self.train_glioma_dataset[0], self.test_glioma_dataset[0])
                (self.train_glioma_data_labels, self.test_glioma_data_labels) = (self.train_glioma_dataset[1], self.test_glioma_dataset[1])
                (self.train_meningioma_data_tensor, self.test_meningioma_data_tensor) = (dataset_two[0][0], dataset_two[0][1])
                (self.train_meningioma_data_labels, self.test_meningioma_data_labels) = (dataset_two[1][0], dataset_two[1][1])

            
        elif (dataset_one is not None) and (dataset_two is None):

                (self.train_glioma_data_tensor, self.test_glioma_data_tensor) = (dataset_one[0][0], dataset_one[0][1])
                (self.train_glioma_data_labels, self.test_glioma_data_labels) = (dataset_one[1][0], dataset_one[1][1])
                (self.train_meningioma_data_tensor, self.test_meningioma_data_tensor) = (self.train_meningioma_dataset[0], self.test_meningioma_dataset[0])
                (self.train_meningioma_data_labels, self.test_meningioma_data_labels) = (self.train_meningioma_dataset[1], self.test_meningioma_dataset[1])
            
        elif (dataset_one is not None) and (dataset_two is not None):
                
                (self.train_glioma_data_tensor, self.test_glioma_data_tensor) = (dataset_one[0][0], dataset_one[0][1])
                (self.train_glioma_data_labels, self.test_glioma_data_labels) = (dataset_one[1][0], dataset_one[1][1])
                (self.train_meningioma_data_tensor, self.test_meningioma_data_tensor) = (dataset_two[0][0], dataset_two[0][1])
                (self.train_meningioma_data_labels, self.test_meningioma_data_labels) = (dataset_two[1][0], dataset_two[1][1])
            
        else:

                (self.train_glioma_data_tensor, self.test_glioma_data_tensor) = (self.train_glioma_dataset[0], self.test_glioma_dataset[0])
                (self.train_glioma_data_labels, self.test_glioma_data_labels) = (self.train_glioma_dataset[1], self.test_glioma_dataset[1])
                (self.train_meningioma_data_tensor, self.test_meningioma_data_tensor) = (self.train_meningioma_dataset[0], self.test_meningioma_dataset[0])
                (self.train_meningioma_data_labels, self.test_meningioma_data_labels) = (self.train_meningioma_dataset[1], self.test_meningioma_dataset[1])


    def _fit_model(self):
        
        self._extract_data()
        if self.model.model_type == "CONCATINATIVE":
            
            concatinative_labels = []
            for (train_glioma_label, train_meningioma_label) in zip(self.train_glioma_data_labels, self.train_meningioma_data_labels):
                 
                if (train_glioma_label == 0) or (train_meningioma_label == 0):
                    concatinative_labels.append(0)
                
                else:
                    concatinative_labels.append(1)
            
            concatinative_labels = np.asarray(concatinative_labels)
            self.model_history = self.model.model.fit([self.train_glioma_data_tensor, self.train_meningioma_data_tensor],
                           concatinative_labels,
                           batch_size=30,
                           epochs=60)
        
        elif self.model.model_type == "PARALLEL":
             
            self.model_history_1 = self.model.first_model.fit(self.train_glioma_data_tensor, self.train_glioma_data_labels, 
                                                         batch_size=1,
                                                         epochs=60)
            
            self.model_history_2 = self.model.second_model.fit(self.train_meningioma_data_tensor, self.train_meningioma_data_labels, 
                                                         batch_size=1,
                                                         epochs=60)
             
    def _save_model(self, place_path):

        if self.model.model_type == "CONCATENATIVE":
             
             self.model.model.save(place_path)
        
        elif self.model.model_type == "PARALLEL":
             
            first_model_path = os.path.join("first_mode", place_path)
            second_model_path = os.path.join("first_mode", place_path)

            self.model.first_model.save(first_model_path)
            self.model.second_model.save(second_model_path)


if __name__ == "__main__":

    model = ConcatenativeModel(learning_rate=0.1, pooling_size=(2, 2), filters_size=(3, 3))
    model_kernel = ModelKernel(model=model, 
                    data_dir="C:\\Users\\1\\Desktop\\MachineLearningStud\\archive",
                    image_size=(224, 224), samples_count=200)
    model_kernel._load_data()
    train_glioma_data, test_glioma_data, train_meningioma_data, test_meningioma_data = model_kernel._formulate_data()

    plt.style.use("dark_background")
    fig, axis = plt.subplots(ncols=4, nrows=6)
    cmaps = sorted(m for m in plt.cm.datad if not m.endswith("_r"))

    train_glioma_data_tensor = train_glioma_data[0]
    separated_data = model_kernel._formulate_data_for_separation()[0]
    train_separation_tensor = separated_data[0]

    spector_one_cmap = rd.choice(cmaps)
    spector_two_cmap = rd.choice(cmaps)
    spector_three_cmap = rd.choice(cmaps)

    spectors = ["twilight", "copper", "viridis"]

    random_choise_1 = rd.randint(0, train_glioma_data_tensor.shape[0])
    random_choise_2 = rd.randint(0, train_glioma_data_tensor.shape[0])
    random_choise_3 = rd.randint(0, train_glioma_data_tensor.shape[0])

    random_sep_choise_1 = rd.randint(0, len(train_separation_tensor))
    random_sep_choise_2 = rd.randint(0, len(train_separation_tensor))
    random_sep_choise_3 = rd.randint(0, len(train_separation_tensor))

    axis[0, 0].imshow(train_glioma_data_tensor[random_choise_1][:, :, 0], cmap=spectors[0])
    axis[0, 1].imshow(train_glioma_data_tensor[random_choise_1][:, :, 1], cmap=spectors[1])
    axis[0, 2].imshow(train_glioma_data_tensor[random_choise_1][:, :, 2], cmap=spectors[2])
    axis[0, 3].imshow(train_glioma_data_tensor[random_choise_1])
    axis[1, 0].imshow(train_glioma_data_tensor[random_choise_2][:, :, 0], cmap=spectors[0])
    axis[1, 1].imshow(train_glioma_data_tensor[random_choise_2][:, :, 1], cmap=spectors[1])
    axis[1, 2].imshow(train_glioma_data_tensor[random_choise_2][:, :, 2], cmap=spectors[2])
    axis[1, 3].imshow(train_glioma_data_tensor[random_choise_2])
    axis[2, 0].imshow(train_glioma_data_tensor[random_choise_3][:, :, 0], cmap=spectors[0])
    axis[2, 1].imshow(train_glioma_data_tensor[random_choise_3][:, :, 1], cmap=spectors[1])
    axis[2, 2].imshow(train_glioma_data_tensor[random_choise_3][:, :, 2], cmap=spectors[2])
    axis[2, 3].imshow(train_glioma_data_tensor[random_choise_3])

    axis[3, 0].imshow(train_separation_tensor[random_sep_choise_1][:, :, 0], cmap=spectors[0])
    axis[3, 1].imshow(train_separation_tensor[random_sep_choise_1][:, :, 1], cmap=spectors[1])
    axis[3, 2].imshow(train_separation_tensor[random_sep_choise_1][:, :, 2], cmap=spectors[2])
    axis[3, 3].imshow(train_separation_tensor[random_sep_choise_1])
    axis[4, 0].imshow(train_separation_tensor[random_sep_choise_2][:, :, 0], cmap=spectors[0])
    axis[4, 1].imshow(train_separation_tensor[random_sep_choise_2][:, :, 1], cmap=spectors[1])
    axis[4, 2].imshow(train_separation_tensor[random_sep_choise_2][:, :, 2], cmap=spectors[2])
    axis[4, 3].imshow(train_separation_tensor[random_sep_choise_2])
    axis[5, 0].imshow(train_separation_tensor[random_sep_choise_3][:, :, 0], cmap=spectors[0])
    axis[5, 1].imshow(train_separation_tensor[random_sep_choise_3][:, :, 1], cmap=spectors[1])
    axis[5, 2].imshow(train_separation_tensor[random_sep_choise_3][:, :, 2], cmap=spectors[2])
    axis[5, 3].imshow(train_separation_tensor[random_sep_choise_3])

    plt.show()
    fig = plt.figure()
    axis_1 = fig.add_subplot(1, 3, 1, projection="3d")
    axis_2 = fig.add_subplot(1, 3, 2, projection="3d")
    axis_3 = fig.add_subplot(1, 3, 3, projection="3d")
    
    grid_spec_x, grid_spec_y = np.meshgrid(
         np.linspace(0, train_separation_tensor[0].shape[0], train_separation_tensor[0].shape[0]),
         np.linspace(0, train_separation_tensor[0].shape[1], train_separation_tensor[0].shape[1])
    )

    fig = go.Figure(data=[go.Surface(
         x=grid_spec_x,
         y=grid_spec_y,
         z=train_separation_tensor[random_sep_choise_3][:, :, 2] + np.cos(np.random.normal(0, 2.50, (train_separation_tensor[0][:, :, 2].shape[0],
            train_separation_tensor[0][:, :, 2].shape[1]))),
        opacity=0.65,
        colorscale="amp"
    )])

    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                     highlightcolor="limegreen", project_z=True))

    fig.show()

    
    
    
    #plt.show()
    model_kernel._fit_model()



            



    