import configparser

def read_config(section):
    '''
    Read config file settings
    :param str section: Section of the config file to read from
    :return: Key value pairs of configurations
    :rtype: dict
    '''
    config = configparser.ConfigParser()
    config.read('config.ini')

    if section in config:
        return config[section]
    else:
        return {}
    