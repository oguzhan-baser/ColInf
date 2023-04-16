import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
import os

def getGlobalSens(dataset):
    """Calculates the global sensitivity of the dataset. The formula is given in the paper"""
    max_val = np.max(dataset)
    min_val = np.min(dataset)
    return np.abs(max_val - min_val)

def generateLaplacianNoise(epsilon, globalSens, size,loc=10):
    """A laplacian noise is generated to apply to satisfy certain differential privacy. The formula is given in the paper"""
    scale = globalSens/epsilon
    noise = np.random.laplace(loc, scale,size)
    return noise

def getHardEnv(envData, hardnessLevelEnv = 0.5):
    """Gets the original dataset and adds noise to generate the environment having a certain hardness level.
    In other words, a DP noise is generated and added to the data samples to satisfy certain level of hardness as formulated in the paper

    Args:
        envData: the original datasamples to be perturbed to have a certain hardness level
        hardnessLevelEnv (float, optional): The noise level to be injected into the data samples. Defaults to 0.5.

    Returns:
        The dataset with a certain noise level or the environment with a certain hardness level
    """
    epsilonEnv = 1/hardnessLevelEnv
    globalSensEnv = getGlobalSens(envData)
    envNoise = generateLaplacianNoise(epsilonEnv, globalSensEnv, np.shape(envData))
    noisyEnv = envData + envNoise
    # some hard parameters
    image_origin = 0
    image_max = 1
    
    # normalization
    nmin = np.min(np.reshape(noisyEnv, (np.shape(noisyEnv)[0],-1)), axis=1) 
    nmin = np.reshape(np.repeat(nmin, np.shape(noisyEnv)[-1]*np.shape(noisyEnv)[-2]), np.shape(noisyEnv))
    
    
    noisyEnv = (noisyEnv - nmin + image_origin) 

    nmax = np.max(np.reshape(noisyEnv, (np.shape(noisyEnv)[0],-1)), axis=1) 
    nmax = np.reshape(np.repeat(nmax, np.shape(noisyEnv)[-1]*np.shape(noisyEnv)[-2]), np.shape(noisyEnv))
    
    noisyEnv =noisyEnv / (nmax/image_max)
    return noisyEnv


def generateTrainEnvs(hardness_levels_arr):
    """    This function creates environments with different hardness levels for model training.
            Takes the first 50k images to process for all training datasets. Each dataset have the same samples but different noise levels.

    Args:
        hardness_levels_arr: the array storing the hardness levels of the environments that robots is operating on and training its models
    """
    path = "hardEnvs"
    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(path)
    # Model / data parameters
    num_envs = 1
    sample_per_env = 50000

    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    print('Max val: '+str(np.max(x_train)))
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    env_sample_num = num_envs*sample_per_env

    x_train = x_train[-env_sample_num:]
    y_train = y_train[-env_sample_num:]

    x_env = np.squeeze(x_train)
    y_env = y_train

    hardness_levels = hardness_levels_arr

    for hardness in hardness_levels:
        print("The environment is being generated for the hardness level: "+str(hardness))
        env = getHardEnv(x_env,hardness) # apply the function injects noise with a certain privacy budget
        env = env*255
        env = np.uint8(env)
        np.save('hardEnvs/mnist_train_data_env_hardness_'+str(hardness)+'.npy',env)
        np.save('hardEnvs/mnist_train_labels_env_hardness_'+str(hardness)+'.npy',y_env)


def generateMixedTest(hardness_levels):
    """ This function generates a test dataset or test environments contains samples from different noise levels
    These noise levels consisting of the robot's environmental noise levels and it is used to measure the utility of a model.
    In other words, as the model becomes more robust to noise by having chance to train over the noisy dataset, it becomes more valuable or its utility increases.
    This utility is represented as test accuracy on this dataset containing samples from different noise levels and this dataset mimics the real world representation of the data
    since it is hard to find a dataset contining only one noise level in real life.

    Args:
        hardness_levels: the array storing the hardness levels of the environments that robots is operating on and training its models
    """
    # input parameters
    env_path = 'hardEnvs/'
    data_file_name = 'mnist_train_data_env_hardness_'
    label_file_name = 'mnist_train_labels_env_hardness_'

    target_env_name = 'testEnvironment'
    hardness_levels = hardness_levels
    targetEnvSize = 1000

    hardness_levels = np.flip(hardness_levels)
    environments = [[] for _ in range(len(hardness_levels))]
    environments_labels = [[] for _ in range(len(hardness_levels))]

    # for each hardness level get generated environments
    for hardness in range(len(hardness_levels)):
        environments[hardness] = np.load(env_path+data_file_name+str(hardness_levels[hardness])+'.npy')
        environments_labels[hardness] = np.load(env_path+label_file_name+str(hardness_levels[hardness])+'.npy')

    # mix the dataset having different noise levels
    environments = np.asarray(environments)
    environments = np.swapaxes(environments, 0, 1)
    environments = environments.reshape((-1, 28, 28))[:targetEnvSize]

    environments_labels = np.asarray(environments_labels)
    environments_labels = np.swapaxes(environments_labels, 0, 1)
    environments_labels = environments_labels.reshape((-1, 1))[:targetEnvSize]
    # save the dataset
    np.save(env_path+data_file_name+target_env_name, environments)
    np.save(env_path+label_file_name+target_env_name, environments_labels)