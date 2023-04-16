import yaml

def get_config():
    '''
    Some parameters in config.yaml might need to be 
    computed or are dependent on others. This function
    gives a way to repair or confirm such parameters before
    running the code.
    '''
    with open('config.yaml', 'r') as yamlfile:
        params = yaml.safe_load(yamlfile)

    with open('config.yaml', 'w') as yamlfile:
        yaml.safe_dump(params, yamlfile, default_flow_style = False)
    
    return params