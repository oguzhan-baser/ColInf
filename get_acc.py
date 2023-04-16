import MNIST_model
import torch
import model_utils
import dataset_utils as data_utils
import numpy as np
import config_utils

def getAcc(xloc, yloc, modelloc):
    """Calculate the accuracy of a model

    Args:
        xloc: the location of the test dataset
        yloc: the location of the test labels
        modelloc: the location of the model

    Returns:
        the accuracy of the model
    """
    params = config_utils.get_config()

    #X_test = np.load("dataset/data_env_hardness_1-1.npy")
    X_test = np.load(xloc)
    y_test = np.load(yloc)
    #y_test = np.load("dataset/labels_env_hardness_1-1.npy")

    test_dataset = data_utils.create_MNIST_dataset(X_test, y_test)

    model = MNIST_model.MNISTClassifier().to(params['testing']['device'])
    #model.load_state_dict(torch.load("models/70_50.pt"))
    model.load_state_dict(torch.load(modelloc))
    confusion_matrix, label_stats = model_utils.test_model(model, test_dataset, params)
    acc = model_utils.accuracy(confusion_matrix)
    print(acc)
    return acc

def getAccFromData(X_test, y_test, modelloc):
    """Calculate the accuracy of a model

    Args:
        X_test: the test data
        y_test: the test labels
        modelloc: the location of the model

    Returns:
        the accuracy of the model
    """
    params = config_utils.get_config()
    test_dataset = data_utils.create_MNIST_dataset(X_test, y_test)

    model = MNIST_model.MNISTClassifier().to(params['testing']['device'])
    model.load_state_dict(torch.load(modelloc))
    confusion_matrix, label_stats = model_utils.test_model(model, test_dataset, params)
    acc = model_utils.accuracy(confusion_matrix)
    print(acc)
    return acc

def getlabels(X_test, model):
    """Get the prediction labels of dataset for a given model

    Args:
        X_test: the dataset to be predicted
        model: the classifier

    Returns:
        the predicted labels
    """
    params = config_utils.get_config()
    test_b_size = params['testing']['test_b_size']
    y_test = np.ones((np.shape(X_test)[0],1))
    test_dataset = data_utils.create_MNIST_dataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_b_size)
    model.eval()
    preds = np.zeros((X_test.shape[0],1))
    fill_idx = 0
    with torch.no_grad():
        for x, y in test_loader:
            out = model(x.to(params['testing']['device']))
            _, pred = torch.max(out.data, 1)
            if fill_idx+test_b_size<X_test.shape[0]:
                preds[fill_idx:fill_idx+test_b_size] = np.expand_dims(pred.cpu().detach().numpy(),-1)
            else:
                preds[fill_idx:] = np.expand_dims(pred.cpu().detach().numpy(), -1)
            fill_idx += test_b_size
    preds = np.reshape(np.asarray(preds),(X_test.shape[0],-1))
    return preds