import json
import os
import requests
from requests.exceptions import ReadTimeout
import threading
import uuid

from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
import uvicorn

from api import extract_keywords
from database_handler import (
    drop_collection,
    find_last,
    insert_one,
    insert_many,
    rename_collection
)
from settings import (
    API_KEY,
    API_URL,
    ARTICLE_LIMIT,
    COLLECTION_NEWS,
    COLLECTION_NODES,
    COLLECTION_RELATIONS,
    COLLECTION_WEBHOOKS,
    GRAPH_SIMULATION_URL,
    HOST_URL,
    KEYWORDS_PER_ARTICLE,
    RAW_DB_NAME,
    RENDERED_DB_NAME,
    SCRAPER_MAPPINGS,
    SCRAPY_PROJ_PATH,
    SIMILARITY_THRESHOLD,
    TEMP_COLLECTION_NEWS,
    TEMP_COLLECTION_NODES,
    TEMP_COLLECTION_RELATIONS,
    WINDOW_SIZE
)
from utils import (
    verify_origin,
    timestamp_to_epoch,
    map_relations,
    merge_adjacency,
    get_cosine_similarity
)
from WQUPC import WeightedQuickUnionPathCompression as WQUPC


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'], 
    allow_headers=['*'],
)

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
            print(f'({len(data)}) files from {filepath}')
            for i, article_obj in enumerate(data[-1 * ARTICLE_LIMIT:]): # url, title, date, content
                if i % 10 == 0:
                    print(f'{round(100 * i / min(len(data), 2), ARTICLE_LIMIT)}%')
                doc = {
                    'title': article_obj['title'],
                    'url': article_obj['url'],
                    'content': article_obj['content'],
                    'datetime': timestamp_to_epoch(article_obj['datetime']),
                    'publisher': publisher,
                    'keys': []
                }
                process_article(article_obj, doc, visited, nodes, relations, embeddings, news_docs)
                del doc['content']

    visited.clear()
    for scraper in SCRAPER_MAPPINGS:
        filepath = os.path.join("webscraper", SCRAPER_MAPPINGS[scraper]['save_file'])
        # relation mapping
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f'({len(data)}) files from {filepath}')
            for i, article_obj in enumerate(data[-1 * ARTICLE_LIMIT:]): # url, title, date, content
                if i % 10 == 0:
                    print(f'{round(100 * i / min(len(data), 2), ARTICLE_LIMIT)}%')
                process_article_relations(article_obj, visited, relations)
    
    # reconciliation using WQUPC
    wqupc = WQUPC(len(embeddings))
    words = list(embeddings.keys())
    for i, word in enumerate(words):
        for j, other in enumerate(words):
            sim = get_cosine_similarity(embeddings[word], embeddings[other])
            if i != j and sim > SIMILARITY_THRESHOLD:
                wqupc.union(i, j)
                print(word, other, sim)
    
    replacement_map = {}
    for i, id in enumerate(wqupc.get_ids()):
        # belongs to a cluster
        if i != id:
            # transfer relations to parent
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
    drop_collection(TEMP_COLLECTION_NODES)
    drop_collection(TEMP_COLLECTION_NEWS)
    drop_collection(TEMP_COLLECTION_RELATIONS)

    print(f'Insert ({len(nodes)}) records to {TEMP_COLLECTION_NODES}')
    print(f'Insert ({len(news_docs)}) records to {TEMP_COLLECTION_NEWS}')
    print(f'Insert ({len(relation_docs)}) records to {TEMP_COLLECTION_RELATIONS}')

    insert_many(TEMP_COLLECTION_NODES, list(map(lambda item: {
        'data': item[0], 
        'freq': item[1]
    }, nodes.items()))) 
    insert_many(TEMP_COLLECTION_NEWS, news_docs)
    insert_many(TEMP_COLLECTION_RELATIONS, relation_docs) 
    insert_one(COLLECTION_WEBHOOKS, {'token': webhook_token})
    
    print(f'GET / {GRAPH_SIMULATION_URL}')
    print(f'dbraw: {RAW_DB_NAME}')
    print(f'dbrendered: {RENDERED_DB_NAME}')
    print(f'webhook: {HOST_URL}/webhook/{webhook_token}/')

    try:
        res = requests.get(url=GRAPH_SIMULATION_URL, timeout=10, params={
            'dbraw': RAW_DB_NAME,
            'dbrendered': RENDERED_DB_NAME,
            'webhook': f'{HOST_URL}/webhook/{webhook_token}/'
        })
    except ReadTimeout as e:
        pass

@app.get('/')
def read_root():
    return "/scraper/, /processor/"
    
@app.get('/cycle/')
def cycle(request: Request) -> dict:
    if not request.headers.get('API_KEY') or not verify_origin(request.headers.get('API_KEY')):
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
    r = requests.get(f'{API_URL}/flush-cache/?token={API_KEY}') # clear cache
    
    return {'response': r.text}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=False)
