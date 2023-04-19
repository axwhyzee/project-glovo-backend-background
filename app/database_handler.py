from pymongo import MongoClient
from dotenv import load_dotenv, find_dotenv
import time
import os

# load env variable file
load_dotenv(find_dotenv())

PASSWORD = os.environ.get("MONGODB_PWD")
CONNECTION_STRING = f'mongodb+srv://admin:{PASSWORD}@cluster0.xo29eao.mongodb.net/?retryWrites=true&w=majority'
DB_NAME = 'project-glovo'
KEYWORDS_PER_ARTICLE = 5
COLLECTION_NEWS = 'news'
COLLECTION_NODES = 'nodes'
COLLECTION_RELATIONS = 'relations'

client = MongoClient(CONNECTION_STRING)
db = client[DB_NAME]


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


def find_all(collection: str) -> list:
    '''
    Get all documents for a specific collection

    :param str collection: Collection name
    :return: List of documents
    '''
    cursor = db[collection].find()

    return list(cursor)


def find_many(collection: str, condition: dict) -> list[dict]:
    '''
    Find multiple documents

    :param str collection: Collection name
    :param dict condition: Match condition
    :return: List of documents that match condition
    '''
    return db[collection].find(condition)


def delete_many(collection: str, condition) -> dict:
    '''
    Delete multiple documents

    :param str collection: Collection name
    :param dict condition: Match condition
    :return: Number of documents deleted
    '''
    return db[collection].delete_many(condition).deleted_count


def clean_up_by_days(days: int) -> int:
    '''
    Remove all news documents where publish date is older than specified days old.
    Decrement frequency of nodes that were keywords of the news documents deleted.

    :param int days: Lower limit for publish date
    :return: Object containing number of nodes and news documents deleted
    '''
    lower_limit = time.time() - days * 24 * 60 * 60
    decrement_keywords = {}
    key = ''
    count = 0

    for doc in find_many(COLLECTION_NEWS, {'datetime': {"$lt": lower_limit}}):
        print(doc)
        for i in range(1, KEYWORDS_PER_ARTICLE + 1):
            key = f'key{i}'
            if key in doc:
                keyword = doc[key]
                decrement_keywords[keyword] = decrement_keywords.get(keyword, 0) + 1

    # replace all nodes with newly calculated ones
    nodes = find_all(COLLECTION_NODES)

    for i in range(len(nodes)-1, -1, -1):
        key = nodes[i]['data']
        if key in decrement_keywords:
            count = decrement_keywords[key]

            # key == count means after decrementing, the node will have frequency of 0
            if key == count:
                nodes.pop(i)
            else:
                nodes[i]['freq'] -= count

    delete_many(COLLECTION_NODES, {})

    if nodes:
        insert_many(COLLECTION_NODES, nodes)

    delete_news_count = delete_many(COLLECTION_NEWS, {'datetime': {'$lt': lower_limit}})

    return {'News deleted': delete_news_count, 'Update nodes': len(decrement_keywords)}