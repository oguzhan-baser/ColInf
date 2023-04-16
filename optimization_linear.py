""" This is a script storing differen resource allocation mechanisms including ours for the collaborative inference"""
# import the necessary packages
import numpy as np
import cvxpy as cp

def uniformAssignment(config, num_of_robots=2):
    """ The resource allocation is done equally to all contributors so that all robots utilizes all resources equally.

    Args:
        config (_type_): the information about the constraints and hyperparameters of the problem
        num_of_robots (int, optional): The number of robots. Defaults to 2.

    Returns:
        rates: the decided offloading rates of the robots
        objective_value: the utility of the robots after the allocation
    """
    # The constraint donating the cloud's limited computation capability or network bandwith
    R = config['cloud_resource']
    # The deadlines of different robots
    Tau = config['task_deadlines']
    # The elasticity of the accuracy variable in the utility formulation
    gamma = config['accuracy_elasticity']
    # The elasticity of the hardness variable in the utility formulation
    beta = config['hardness_elasticity']
    # The accuracy of the robot's local model
    alpha_r = config['robot_accuracy']
    # The accuracy of the cloud model
    alpha_c = config['cloud_accuracy']
    # The hardness level of the robot's operation environment
    Omega = config['hardness_levels']

    tau_sum = np.sum(Tau)
    rates = np.ones((num_of_robots,1))*R/tau_sum
    rates[rates>=1]=1
    objective_value = - np.sum(np.multiply(np.power((alpha_c - alpha_r)*rates+alpha_r,gamma) , np.power(Omega,beta)))
    print(rates)
    print(-1*objective_value)
    return rates, objective_value


def randomAssignment(config, num_of_robots=2):
    """The resource allocation is done probabilistically to all contributors so that all robots utilizes all resources randomly.

    Args:
        config (_type_): the information about the constraints and hyperparameters of the problem
        num_of_robots (int, optional): The number of robots. Defaults to 2.

    Returns:
        rates: the decided offloading rates of the robots
        objective_value: the utility of the robots after the allocation
    """
    # The constraint donating the cloud's limited computation capability or network bandwith
    R = config['cloud_resource']
    # The deadlines of different robots
    Tau = config['task_deadlines']
    # The elasticity of the accuracy variable in the utility formulation
    gamma = config['accuracy_elasticity']
    # The elasticity of the hardness variable in the utility formulation
    beta = config['hardness_elasticity']
    # The accuracy of the robot's local model
    alpha_r = config['robot_accuracy']
    # The accuracy of the cloud model
    alpha_c = config['cloud_accuracy']
    # The hardness level of the robot's operation environment
    Omega = config['hardness_levels']

    
    rates = np.ones((num_of_robots,1))*np.random.rand(num_of_robots,1)
    while np.sum(rates*Tau)>R:
        # rates = np.ones((num_of_robots,1))*np.random.rand(num_of_robots,1)
        rates = rates * R / num_of_robots / Tau
    rates[rates>=1]=1
    objective_value = - np.sum(np.multiply(np.power((alpha_c - alpha_r)*rates+alpha_r,gamma) , np.power(Omega,beta)))
    print(rates)
    print(-1*objective_value)
    return rates, objective_value

def utilityMax(config, num_of_robots=2):
    """The resource allocation is done via the algorithm we probided in the paper to all contributors so that all robots utilizes all resources fairly.

    Args:
        config (_type_): the information about the constraints and hyperparameters of the problem
        num_of_robots (int, optional): The number of robots. Defaults to 2.

    Returns:
        rates: the decided offloading rates of the robots
        objective_value: the utility of the robots after the allocation
    """
    # The constraint donating the cloud's limited computation capability or network bandwith
    R = config['cloud_resource']
    # The deadlines of different robots
    Tau = config['task_deadlines']
    # The elasticity of the accuracy variable in the utility formulation
    gamma = config['accuracy_elasticity']
    # The elasticity of the hardness variable in the utility formulation
    beta = config['hardness_elasticity']
    # The accuracy of the robot's local model
    alpha_r = config['robot_accuracy']
    # The accuracy of the cloud model
    alpha_c = config['cloud_accuracy']
    # The hardness level of the robot's operation environment
    Omega = config['hardness_levels']

    
    # Variables 
    x_rate = cp.Variable((num_of_robots,1))
    A_mat = np.concatenate((np.transpose(Tau),-np.eye(num_of_robots),np.eye(num_of_robots)),axis=0)
    b_vec = np.concatenate((np.array([[R]]), np.zeros((num_of_robots,1)),np.ones((num_of_robots,1))), axis=0)
    # Constraints 
    constraint = [A_mat @ x_rate <= b_vec]
    # Objective
    objective = cp.Minimize(-cp.sum(cp.multiply(cp.power(cp.multiply(alpha_c-alpha_r, x_rate)+alpha_r,gamma),cp.power(Omega,beta))))
    # Solve the problem
    problemm= cp.Problem(objective, constraint)
    problemm.solve()
    # Result
    print('Rates: ')
    print(x_rate.value)
    print('Total Utility: ')
    print(-1*objective.value)
    print('==================')
    return x_rate.value, objective.value
