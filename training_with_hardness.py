import MNIST_model
import torch
import model_utils
import config_utils
import dataset_utils as data_utils
import numpy as np
from sklearn.model_selection import train_test_split
import logging
import os

def trainHardEnvs(hardness_level):
    """ Trains models on different hardness levels provided as input.
    It asks how many input images that user wants to use to train the model
    After training it asks whether user wants to save the model.
    This way we are able to train models with different number of data samples so that we have the models with different accuracies
    Then the model is saved accordingly to the ./models folder
    Since we want to preserve the ones used in the original experiments, previous models are stored in ./models_original folder.
    Args:
        hardness_level : the hardness level of the robot's operating environment
    """
    #Some setup for logging
    logger = logging.getLogger("main")
    logger.setLevel(logging.DEBUG)

    params = config_utils.get_config()

    DATASET_LOCATION = "hardEnvs/"
    SAVE_PARENT_DIR = "models/hardness_"+str(hardness_level)+"/"

    isExist = os.path.exists("models")
    if not isExist:
        os.mkdir("models")

    isExist = os.path.exists(SAVE_PARENT_DIR)
    if not isExist:
        os.mkdir(SAVE_PARENT_DIR)


    while True:
        X_train = np.load(DATASET_LOCATION + "mnist_train_data_env_hardness_"+str(hardness_level)+".npy")
        y_train = np.load(DATASET_LOCATION + "mnist_train_labels_env_hardness_"+str(hardness_level)+".npy")

        total_test_size = params['dataset_info']['val_ratio'] + params['dataset_info']['test_ratio']
        X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,test_size=total_test_size,stratify=y_train,random_state=1)
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size = 0.5, stratify = y_val, random_state = 1)
        print("The hardness level is: "+str(hardness_level))
        num_images = int(input("Number of images for this training cycle: "))
        use_percentage = num_images/X_train.shape[0]
        #X_rem, X_train, y_rem, y_train = train_test_split(X_train, y_train, test_size=use_percentage, stratify=y_train, random_state=1)

        print("\n\n")
        logger.debug(f"X_train/y_train shape: {X_train.shape}, {y_train.shape}")
        #logger.debug(f"X_rem/y_rem shape:   : {X_rem.shape}, {y_rem.shape}")
        logger.debug(f"X_val/y_val shape    : {X_val.shape}, {y_val.shape}")
        logger.debug(f"X_test/y_test shape  : {X_test.shape}, {y_test.shape}")

        train_dataset = data_utils.create_MNIST_dataset(X_train, y_train)
        val_dataset = data_utils.create_MNIST_dataset(X_val, y_val)
        test_dataset = data_utils.create_MNIST_dataset(X_test, y_test)

        sampling_distribution = eval(params['dataset_info']['initial_distribution_generator'])
        train_dataset = data_utils.sample_nonuniform(train_dataset, sampling_distribution, num_images, params)

        model = MNIST_model.MNISTClassifier().to(params['training']['device'])
        #model_utils.train_model2(train_dataset, val_dataset, model, params) #Val accuracy convergent training function
        model_utils.train_model(train_dataset, model, params, silent = False)

        conf_matrix, label_stats = model_utils.test_model(model, test_dataset, params)
        acc = model_utils.accuracy(conf_matrix)
        logger.info(f"Accuracy of model on test dataset: {acc}")

        save_model = str(input("Save this model (y/n)? "))
        if save_model == 'y':
            torch.save(model.state_dict(), SAVE_PARENT_DIR + f"{round(acc*100)}_{num_images}.pt")
        elif save_model == "b":
            break
        else:
            pass

