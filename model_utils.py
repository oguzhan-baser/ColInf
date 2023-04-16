import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import numpy as np
import yaml

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("model_utils")

def init_weights(m):
    '''
    Initializes weights for MNIST classifier
    '''
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)
    elif type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)
    elif type(m) == nn.BatchNorm1d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.01)


def train_model2(train_dataset, val_dataset, model, params, silent=True):
    """ Train the model to achieve convergent accuracy on the validator dataset.

    Args:
        train_dataset : the training dataset
        val_dataset : the validation dataset
        model : the model
        params : the training hyper-parameters in the configuration file
        silent (bool, optional): Whether the training occur without verbosity. Defaults to True.

    Returns:
        The trained model
    """
    local_logger = logger.getChild("train_model_without_fixed_epochs")
    local_logger.info("Started model training...")
    
    lr = params['training']['lr']
    b_size =params['training']['b_size']
    max_epochs = params['training']['max_epochs']
    convergence_threshold = params['training']['convergence_threshold']

    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.99,last_epoch=-1)
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=b_size, shuffle=True, worker_init_fn=0)

    prev_val_acc = 0
    if not silent:
        pbar = tqdm([i for i in range(max_epochs)], total=max_epochs)
    else:
        pbar = [i for i in range(max_epochs)]
    for epoch in pbar:
        for x,y in dataloader:
            
            model.zero_grad()

            out = model(x.to(params['training']['device']))
            loss = loss_fn(out, y.to(params['training']['device']))

            loss.backward()

            optimizer.step()

        lr_sch.step()

        if (epoch % 25) == 0:
            val_conf_matrix,label_stats = test_model(model, val_dataset, params)
            #local_logger.debug(val_conf_matrix)
            val_acc = accuracy(val_conf_matrix)
            local_logger.debug(f"Validation accuracy after {epoch} epochs is {val_acc}")
            if np.abs(val_acc - prev_val_acc) <= convergence_threshold:
                local_logger.debug(f"Training converged after {epoch} epochs")
                break
            prev_val_acc = val_acc

        if not silent:
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
            pbar.set_description(s)
    
    return model


def train_model(train_dataset,model,params,silent=True):
    """ Train the model for a certain number of epochs which is specified in the configuration file.

    Args:
        train_dataset : the training dataset
        model : the model
        params : the training hyper-parameters in the configuration file
        silent (bool, optional): Whether the training occur without verbosity. Defaults to True.

    Returns:
        The trained model
    """
    local_logger = logger.getChild("train_model")
    local_logger.info("Started model training...")
    
    lr = params['training']['lr']
    b_size =params['training']['b_size']
    n_epoch = params['training']['n_epoch']
    
    model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    lr_sch = lr_scheduler.ExponentialLR(optimizer,gamma=0.99,last_epoch=-1)
    
    dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=b_size, shuffle=True, worker_init_fn=0)

    if not silent:
        pbar = tqdm([i for i in range(n_epoch)], total=n_epoch)
    else:
        pbar = [i for i in range(n_epoch)]
    for epoch in pbar:
        for x,y in dataloader:
            
            model.zero_grad()

            out = model(x.to(params['training']['device']))
            loss = loss_fn(out, y.to(params['training']['device']))

            loss.backward()

            optimizer.step()

        lr_sch.step()

        if not silent:
            mem = "%.3gG" % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)
            s = ("%10s" * 1 + "%10.4g" * 1) % (mem, loss)
            pbar.set_description(s)
    
    return model


def test_model(model, test_dataset, params):
    """ A function tests the model for a given dataset with a certain hyper-parameters specified in the params dictionary

    Args:
        test_dataset : the test dataset
        model : the model to be evaluated
        params : the training hyper-parameters in the configuration file

    Returns:
        The trained model
    """
    test_b_size = params['testing']['test_b_size']
    n_class = params['dataset_info']['n_class']
    device = params['testing']['device']

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_b_size)
    confusion_matrix = np.zeros((n_class, n_class), dtype=int)
    label_stats = np.zeros(n_class,dtype=int)
    top5correct = 0

    model.eval()
    with torch.no_grad():
        for x,y in test_loader:
                
            out = model(x.to(device))
            _, pred = torch.max(out.data, 1)
            for i in range(n_class):

                filt_i = (y == i)
                label_stats[i] += sum(filt_i)
                pred_i = pred[filt_i]

                for j in range(n_class):
                    filt_j = (pred_i == j)
                    nnum = sum(filt_j)
                    confusion_matrix[i, j] += nnum

    return confusion_matrix, label_stats

def classify(model, dataset, params):
    """ A function to perform inference

    Args:
        model (_type_): the model performing the inference
        dataset (_type_): the dataset to be predicted
        params (_type_): the hyperparameters for the experiment

    Returns:
        _type_: the predicted labels
    """
    batch_size = params['classification']['batch_size']
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    model.eval()
    y_pred = np.zeros(len(dataset),int)
    i =0
    with torch.no_grad():
        ind = 0
        for x,_ in test_loader:

            out = model(x.to(params['classification']['device']))

            _, pred = torch.max(out.data, 1)
            next_ind = ind+len(x)
            y_pred[ind:next_ind] = pred.cpu().numpy()
            ind = next_ind

    return y_pred, get_dist_from_observances(y_pred, params)

def get_dist_from_observances(y_pred, params):
    dist = np.zeros(params['dataset_info']['n_class'], dtype=int)
    for pred in y_pred:
        dist[pred] += 1
    return dist

def accuracy(conf_matrix):
    '''
    calculates the accuracy given the confusion matrix
    '''
    return np.trace(conf_matrix)/np.sum(conf_matrix)