from pymongo import MongoClient
from settings import read_config
import time
import os


config = read_config('MONGODB')
COLLECTION_NEWS = config['COLLECTION_NEWS']
COLLECTION_NODES = config['COLLECTION_NODES']
COLLECTION_RELATIONS = config['COLLECTION_RELATIONS']

client = MongoClient(os.environ.get('MONGODB_URL'))
db = client[config['DB_NAME']]


def insert_many(collection: str, docs: dict):
    '''
    Insert multiple documents

    :param str collection: Collection name
    :param dict docs: Document object to be inserted
    '''
    db[collection].insert_many(docs)

def insert_one(collection: str, doc: dict):
    '''
    Insert a document

    :param str collection: Collection name
    :param dict docs: Document object to be inserted
    '''
    db[collection].insert_one(doc)

def update_one(collection: str, condition: dict, target: dict, upsert: bool = False):
    '''
    Update a document

    :param str collection: Collection name
    :param dict condition: Match condition
    :param dict target: New values to be updated to
    :param bool upsert: If True, update if found, otherwise insert. False by default
    '''
    db[collection].update_one(condition, target, upsert=upsert)

def find_all(collection: str):
    '''
    Get all documents for a specific collection

    :param str collection: Collection name
    :return: List of documents
    '''
    cursor = db[collection].find()

    return list(cursor)

def find_many(collection: str, condition: dict, projection: dict):
    '''
    Find multiple documents

    :param str collection: Collection name
    :param dict condition: Match condition
    :return: List of documents that match condition
    '''
    return db[collection].find(condition, projection)

def delete_many(collection: str, condition):
    '''
    Delete multiple documents

    :param str collection: Collection name
    :param dict condition: Match condition
    :return: Number of documents deleted
    '''
    return db[collection].delete_many(condition).deleted_count