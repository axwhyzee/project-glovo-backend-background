import os

from pymongo import (
    DESCENDING,
    MongoClient
)
from pymongo.errors import OperationFailure

from settings import read_config


config = read_config('MONGODB')

COLLECTION_NEWS = config['COLLECTION_NEWS']
COLLECTION_NODES = config['COLLECTION_NODES']
COLLECTION_RELATIONS = config['COLLECTION_RELATIONS']
COLLECTION_WEBHOOKS = config['COLLECTION_WEBHOOKS']

RAW_DB_NAME = config['RAW_DB_NAME']
RENDERED_DB_NAME = config['RENDERED_DB_NAME']

TEMP_COLLECTION_NEWS = '_' + COLLECTION_NEWS
TEMP_COLLECTION_NODES = '_' + COLLECTION_NODES
TEMP_COLLECTION_RELATIONS = '_' + COLLECTION_RELATIONS

client = MongoClient(os.environ.get('MONGODB_URL'))
db = client[RAW_DB_NAME]


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

def find_last(collection: str):
    '''
    Find last inserted document

    :param str collection: Collection name
    :return: Last inserted document
    '''
    return db[collection].find_one({}, sort=[('_id', DESCENDING)])

def delete_many(collection: str, condition):
    '''
    Delete multiple documents

    :param str collection: Collection name
    :param dict condition: Match condition
    :return: Number of documents deleted
    '''
    return db[collection].delete_many(condition).deleted_count

def drop_collection(collection: str):
    '''
    Drop collection

    :param str collection: Name of collection to drop
    '''
    try:
        db.drop_collection(collection)
    except OperationFailure as e:
        print(f'{collection} does not exist')

def rename_collection(src: str, dst: str):
    '''
    Rename a collection

    :param str src: Original name
    :param str dst: New name
    '''
    db[src].rename(dst)