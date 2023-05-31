import os

import configparser
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

parser = configparser.ConfigParser()
parser.read('config.ini')    

config = parser['MONGODB']
COLLECTION_NEWS = config['COLLECTION_NEWS']
COLLECTION_NODES = config['COLLECTION_NODES']
COLLECTION_RELATIONS = config['COLLECTION_RELATIONS']
COLLECTION_WEBHOOKS = config['COLLECTION_WEBHOOKS']
RAW_DB_NAME = config['RAW_DB_NAME']
RENDERED_DB_NAME = config['RENDERED_DB_NAME']
TEMP_COLLECTION_NEWS = '_' + COLLECTION_NEWS
TEMP_COLLECTION_NODES = '_' + COLLECTION_NODES
TEMP_COLLECTION_RELATIONS = '_' + COLLECTION_RELATIONS

config = parser['PROCESSOR']
SIMILARITY_THRESHOLD = float(config['SIMILARITY_THRESHOLD'])
KEYWORDS_PER_ARTICLE = int(config['KEYWORDS_PER_ARTICLE'])
WINDOW_SIZE = int(config['WINDOW_SIZE'])
SCRAPY_PROJ_PATH = config['SCRAPY_PROJECT_PATH']
ARTICLE_LIMIT = int(config['ARTICLE_LIMIT'])
SCRAPER_MAPPINGS = {
    'CNBC': {
        'save_file': 'articles_cnbc.json',
        'spider': 'cnbc_spider'
    },
    'Straits_Times': {
        'save_file': 'articles_straits_times.json',
        'spider': 'straits_times_spider'
    },
    'Yahoo': {
        'save_file': 'articles_yahoo.json',
        'spider': 'yahoo_spider'
    }
}

config = parser['MISC']
GRAPH_SIMULATION_URL = config['GRAPH_SIMULATION_URL']
HOST_URL = config['HOST_URL']

API_KEY = os.environ.get('API_KEY')
API_URL = os.environ.get('API_URL')
FERNET_KEY = os.environ.get('FERNET_KEY')
FERNET_SECRET = os.environ.get('FERNET_SECRET')
KEYWORD_EXTRACTION_MICROSERVICE_URL = os.environ.get('KEYWORD_EXTRACTION_MICROSERVICE_URL')

