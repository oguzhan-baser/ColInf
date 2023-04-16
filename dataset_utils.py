# Import necessary libraries
import torchvision.transforms as trfm
import logging
from MNIST_model import MNISTDataset
from torch.utils.data import Subset
import numpy as np

# Setup for logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('dataset_utils')

def create_MNIST_dataset(X,y):
    """ A function creating dataset from the numpy arrays corresponding to the data samples and labels

    Args:
        X (_type_): data samples in numpy format
        y (_type_): corresponding labels in numpy format

    Returns:
        a trainable dataset via pytorch functions
    """
    # Define a transformation pipeline for preprocessing images
    transform = trfm.Compose([
    trfm.ToTensor(),
    trfm.Normalize((0.1307), (0.3081))])
    
    # Create a dataset object using the input X, y and the defined transformation pipeline
    dataset = MNISTDataset(X,y,transform)
    return dataset

def sample_nonuniform(dataset, sampling_distribution, total_images, params):
    # Extract the number of classes from input parameters
    n_class = params['dataset_info']['n_class']

    # Normalize the sampling distribution so that it sums to 1
    sampling_distribution = sampling_distribution/ np.sum(sampling_distribution)

    # Compute the number of samples to select from each class
    sampling_distribution = np.rint(sampling_distribution * total_images)

    # Initialize an empty list to store the selected sample indices
    sample_inds = []

    # Iterate over each class and select the specified number of samples from that class
    for class_idx in range(n_class):
        i = 0
        while sampling_distribution[class_idx] > 0:
            if dataset[i][1] == class_idx:
                # If the class label of the current image matches the current class index,
                # add the index of the image to the list of selected sample indices
                sample_inds.append(i)
                sampling_distribution[class_idx] -= 1
            i += 1

    # Return a subset of the input dataset containing only the selected samples
    return Subset(dataset, sample_inds)