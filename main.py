from envGenerate import *
import numpy as np
from training_with_hardness import trainHardEnvs
# specify the hardness levels of the environments
hardness_levels = np.array([0, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7 ,0.8, 1.0])
# generate environments from different hardness levels
generateTrainEnvs(hardness_levels)
# Generate a dataset repesenting real world consisting of all noise levels
generateMixedTest(hardness_levels)
# train different models for different accuracies on these environments

# comment out the following two lines to train our models similar to the one given in the ./models_original folder
# for hardness in hardness_levels:
#     trainHardEnvs(hardness)
# this training saves the models into the ./models folder
# since it takes time, we presaved the trained model in the folder called ./models_original folder


""" See the file called generateEnvToy.ipynb to see the results for differentially private noise in MNIST image and histograms"""
""" See the file called optimization.ipynb to see the boxplot results shown in the paper"""
""" See the file called generalImp.ipynb to see the concave utility results shown in the paper"""