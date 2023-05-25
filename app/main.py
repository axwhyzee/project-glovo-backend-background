from dateutil import parser
from dotenv import load_dotenv, find_dotenv
from hashlib import sha256
import json
import os
import requests
import threading
import uuid
load_dotenv(find_dotenv())

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
import numpy as np
from tqdm import tqdm
import uvicorn

from api import extract_keywords
from database_handler import (
    COLLECTION_NEWS,
    COLLECTION_NODES,
    COLLECTION_RELATIONS,
    COLLECTION_WEBHOOKS,
    RAW_DB_NAME,
    RENDERED_DB_NAME,
    TEMP_COLLECTION_NODES,
    TEMP_COLLECTION_NEWS,
    TEMP_COLLECTION_RELATIONS,
    drop_collection,
    find_last,
    insert_one,
    insert_many,
    rename_collection
)
from relation_mapper import map_relations
from settings import read_config
from WQUPC import WeightedQuickUnionPathCompression as WQUPC


config = read_config('WEBSCRAPER')

SIMILARITY_THRESHOLD = float(config['SIMILARITY_THRESHOLD'])

KEYWORDS_PER_ARTICLE = int(config['KEYWORDS_PER_ARTICLE'])

WINDOW_SIZE = int(config['WINDOW_SIZE'])

SCRAPY_PROJ_PATH = config['SCRAPY_PROJECT_PATH']

SHA256_SECRET_KEY = os.environ.get('SHA256_SECRET_KEY')

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

config = read_config('MISC')

GRAPH_SIMULATION_URL = config['GRAPH_SIMULATION_URL']

HOST_URL = config['HOST_URL']

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'], 
    allow_headers=['*'],
)


def verify_origin(secret: str) -> bool:
    '''
    Check if request has the correct API secret key

    :param str secret: Secret key from request
    :return: True if SHA256 hash of secret corresponds to SHA256_SECRET_KEY, else False
    :rtype: bool
    '''
    return SHA256_SECRET_KEY == sha256(secret.encode('utf-8')).hexdigest()

def timestamp_to_epoch(timestamp) -> int:
    '''
    ISO 8601 datestring to unix timestamp
    :param str timestamp: ISO 8601 datestring
    :return: Unix timestamp
    :rtype: int
    '''
    if timestamp:
        return int(parser.parse(timestamp).timestamp())

def get_cosine_similarity(a, b):
    numerator = np.dot(a, b.transpose())
    
    a_norm = np.sqrt(np.sum(a ** 2))
    b_norm = np.sqrt(np.sum(b ** 2))

    denominator = a_norm * b_norm

    cosine_similarity = numerator / denominator

    return cosine_similarity

def merge_adjacency(adjacency_list, src, dst):
    '''
    Inplace merging of 2 adjacency lists (merge src into dst)

    :param dict src: Source adjacency list
    :param dict dst: Destination adjacency list
    '''
    adj_src = adjacency_list[src]
    adj_dst = adjacency_list[dst]

    for neighbour in adj_src:
        if neighbour in adj_dst:
            adj_dst[neighbour] += adj_src[neighbour]
        else:
            adj_dst[neighbour] = adj_src[neighbour]
    
    del adjacency_list[src]

def run_scraper():
    '''
    Run Scrapy spiders & save scraped data as JSON files in {SCRAPY_PROJ_PATH}/{SCRAPY_PROJ_PATH}
    A CRON job will call this endpoint every fixed time interval.

    :param str secret: API secret key. If valid, then scrape, else ignore this GET request
    :return: Object that states the Scrapy spiders that were executed.
    :rtype: dict
    '''
    os.chdir(SCRAPY_PROJ_PATH) # CD to where scrapy.cfg is

    for scraper in SCRAPER_MAPPINGS.values():
        path = scraper['save_file']

        # clear previous scrapings
        if (path in os.listdir()):
            os.remove(path)

        os.system(f'scrapy crawl -o {path} -t json {scraper["spider"]}')

    os.chdir('../')

    run_nlp_processor()

def process_article(article_obj, doc, visited, nodes, relations, embeddings, news_docs):
    # check for null, check for visited
    if not (article_obj['url'] and article_obj['datetime'] and article_obj['content']) or article_obj['url'] in visited:
        return
    
    article_obj['content'] = article_obj['content'].lower()
    article_obj['title'] = article_obj['title'].lower()
    
    # extract keyphrases, word embeddings, doc embedding from content. Title used for seeding
    keyphrases = extract_keywords(article_obj['content'].replace('\n', ' '), article_obj['title'], KEYWORDS_PER_ARTICLE)

    for phrase, embedding in keyphrases:
        # increment frequency of phrase
        nodes[phrase] = nodes.get(phrase, 0) + 1

        doc['keys'].append(phrase)
        phrase = phrase.replace(' ', '__')

        if phrase not in relations:
            relations[phrase] = {}
        
        if phrase not in embeddings:
            embeddings[phrase] = embedding

    news_docs.append(doc)
    visited.add(article_obj['url'])

def process_article_relations(article_obj, visited, relations):
    # check for null, check for visited
    if not (article_obj['url'] and article_obj['datetime'] and article_obj['content']) or article_obj['url'] in visited:
        return

    visited.add(article_obj['url'])
    content = article_obj['title'] + ' ' + article_obj['content']
    content = content.lower()

    for joined_phrase in relations:
        content = content.replace(joined_phrase.replace('__', ' '), joined_phrase)

    map_relations(content, relations, WINDOW_SIZE)

def run_nlp_processor():
    '''
    Remove outdated data.
    Read scraped data outputs and conduct keyword extraction, followed by topic modelling and relation extraction.
    Store node and news data to MongoDB.
    Update list of visited URLs to prevent repeat work.

    :return: Object that states the scraped data that have been processed.
    :rtype: dict
    '''
    print('START NLP PROCESSOR')
    relation_docs = [] # collection of docs to be inserted to relations
    news_docs = [] # collection of docs to be inserted to news
    nodes = {} # collection of docs to be inserted to nodes
    relations = {} # temporary, mutable dict of relations that incrementally gets updated for each article
    embeddings = {}
    visited = set()
    webhook_token = str(uuid.uuid4())
    
    # read scraped data from each of the Scrapy spiders
    for publisher in SCRAPER_MAPPINGS:
        filepath = os.path.join("webscraper", SCRAPER_MAPPINGS[publisher]['save_file'])

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for article_obj in tqdm(data): # url, title, date, content
                doc = {
                    'title': article_obj['title'],
                    'url': article_obj['url'],
                    'content': article_obj['content'],
                    'datetime': timestamp_to_epoch(article_obj['datetime']),
                    'publisher': publisher,
                    'keys': []
                }

                process_article(article_obj, doc, visited, nodes, relations, embeddings, news_docs)

    visited.clear()
    for scraper in SCRAPER_MAPPINGS:
        filepath = os.path.join("webscraper", SCRAPER_MAPPINGS[scraper]['save_file'])
        # relation mapping
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

            for article_obj in tqdm(data): # url, title, date, content
                process_article_relations(article_obj, visited, relations)
    
    # reconciliation using WQUPC
    wqupc = WQUPC(len(embeddings))
    words = list(embeddings.keys())

    for i, word in enumerate(words):
        for j, other in enumerate(words):
            sim = get_cosine_similarity(np.array(embeddings[word]), np.array(embeddings[other]))
            if i != j and sim > SIMILARITY_THRESHOLD:
                wqupc.union(i, j)
                print(word, other, sim)
    
    replacement_map = {}
    for i, id in enumerate(wqupc.get_ids()):
        # belongs to a cluster
        if i != id:
            # transfer relations
            merge_adjacency(relations, words[i], words[id])

            words_i_rep = words[i].replace('__', ' ')
            words_id_rep = words[id].replace('__', ' ')

            # transfer node frequency
            nodes[words_id_rep] += nodes[words_i_rep]
            del nodes[words_i_rep]

            replacement_map[words[i]] = words[id]
            replacement_map[words_i_rep] = words_id_rep

    # replace keywords in news_docs
    for i, doc in enumerate(news_docs):
        new_keys = set()
        for key in doc['keys']:
            new_keys.add(replacement_map.get(key, key))
        
        # replace set with list
        news_docs[i]['keys'] = list(new_keys)

    for central in relations:
        relation = relations[central]
        for to_replace in replacement_map:
            if to_replace in relation:
                replacement = replacement_map[to_replace]
                relation[replacement] = relation.get(replacement, 0) + relation[to_replace]
                del relation[to_replace]
        
    # convert relations from hashmap to list of docs to be inserted
    visited.clear()
    for central in relations:
        adjacency = relations[central]

        for adjacent in adjacency:
            if (adjacent, central) in visited or adjacent in replacement_map:
                continue

            relation_docs.append({
                'src': central.replace('__', ' '),
                'dst': adjacent.replace('__', ' '), 
                'weight': adjacency[adjacent]
            })
            
            visited.add((central, adjacent))
    
    # update database
    if nodes:
        insert_many(TEMP_COLLECTION_NODES, list(map(lambda item: {
            'data': item[0], 
            'freq': item[1]
        }, nodes.items()))) 
    if news_docs:
        insert_many(TEMP_COLLECTION_NEWS, news_docs)
    if relation_docs:
        insert_many(TEMP_COLLECTION_RELATIONS, relation_docs)
    
    # store webhook
    insert_one(COLLECTION_WEBHOOKS, {'token': webhook_token})
    
    requests.get(url=GRAPH_SIMULATION_URL, timeout=5, params={
        'dbraw': RAW_DB_NAME,
        'dbrendered': RENDERED_DB_NAME,
        'webhook': f'{HOST_URL}/webhoook/{webhook_token}/'
    })

@app.get('/')
def read_root():
    return "/scraper/, /processor/"
    
@app.get('/cycle/')
def cycle(request: Request) -> dict:
    if not request.headers.get('API_SECRET_KEY') or not verify_origin(request.headers.get('API_SECRET_KEY')):
        return {'response': 'Invalid or missing secret key'}

    thread = threading.Thread(target=run_scraper)
    thread.start()

    return {'response': 'success'}

@app.get('/webhook/{token}/')
def webhook(token: str):
    if token != find_last(COLLECTION_WEBHOOKS)['token']:
        return {'response': 'Invalid token'}
    
    drop_collection(COLLECTION_NEWS)
    rename_collection(TEMP_COLLECTION_NEWS, COLLECTION_NEWS)
    
    drop_collection(COLLECTION_NODES)
    rename_collection(TEMP_COLLECTION_NODES, COLLECTION_NODES)
    
    drop_collection(COLLECTION_RELATIONS)
    rename_collection(TEMP_COLLECTION_RELATIONS, COLLECTION_RELATIONS)
    
    return {'response': 'Success'}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=False)