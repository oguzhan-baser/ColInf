import numpy as np

def xoffload2accAlpha(xoffload, coeff=0.01):
    """a function converting offloading rate to the final accuracy

    Args:
        xoffload: offloading rate of a robot

    Returns:
        the accuracy of a robot for a given offloading rate
    """
    return np.log(np.exp(1)-1/(coeff*xoffload+1/(np.exp(1)-1)))

def accAlpha2xoffload(accAlpha,coeff=0.01):
    """converts accuracy of a robot to offloading rate corresponding to that accuracy

    Args:
        accAlpha : accuracy of a robot

    Returns:
        offloading rate to achieve that accuracy
    """
    xoffload = 1/(np.exp(1)-np.exp(accAlpha))-1/(np.exp(1)-1)
    xoffload = xoffload/coeff
    return xoffload

def rate2utility(alpha_c, alpha_r, rates, Omega, gamma=0.5, beta=0.5):
    """converts offloading rate to the utility to be maximized

    Args:
        alpha_c : cloud model's accuracy
        alpha_r : robot's local model's accuracy
        rates: offloading rate of the robot
        Omega: the hardness of the environment that the robot is operating on
        gamma (float, optional): the elasticity value for accuracy
        beta (float, optional):  the elasticity value for hardness

    Returns:
        the resulting utility obtained with these parameters
    """
    return np.multiply(np.power((alpha_c - alpha_r)*rates+alpha_r,gamma) , np.power(Omega,beta))

def newrate2utility(alpha_c, alpha_r, rates, Omega, gamma=0.5, beta=0.5):
    """ converts the rates into utility but assumes all the variables are between 0 and 1

    Args:
        alpha_c : cloud model's accuracy
        alpha_r : robot's local model's accuracy
        rates: offloading rate of the robot
        Omega: the hardness of the environment that the robot is operating on
        gamma (float, optional): the elasticity value for accuracy
        beta (float, optional):  the elasticity value for hardness

    Returns:
        the resulting utility obtained with these parameters
    """
    alpha_c = alpha_c/100
    alpha_r = alpha_r/100
    return np.multiply(np.power((alpha_c - alpha_r)*rates+alpha_r,gamma) , np.power(Omega,beta))