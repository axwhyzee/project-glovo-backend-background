from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from database_handler import *
from WQUPC import WeightedQuickUnionPathCompression as WQUPC
from relation_mapper import map_relations
from api import extract_keywords
from settings import read_config
from dateutil import parser
from hashlib import sha256
import numpy as np
import uvicorn
import json
import os


config = read_config('WEBSCRAPER')

SIMILARITY_THRESHOLD = float(config['SIMILARITY_THRESHOLD'])
KEYWORDS_PER_ARTICLE = int(config['KEYWORDS_PER_ARTICLE'])
WINDOW_SIZE = int(config['WINDOW_SIZE'])
VISITED_URLS_PATH = config['VISITED_URLS_PATH']
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

# if file does not exist, create file
if VISITED_URLS_PATH not in os.listdir():
    f = open(VISITED_URLS_PATH, "w")
    f.close()


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

def run_nlp_processor():
    '''
    Remove outdated data.
    Read scraped data outputs and conduct keyword extraction, followed by topic modelling and relation extraction.
    Store node and news data to MongoDB.
    Update list of visited URLs to prevent repeat work.

    :return: Object that states the scraped data that have been processed.
    :rtype: dict
    '''
    relation_docs = [] # collection of docs to be inserted to relations
    news_docs = [] # collection of docs to be inserted to news
    nodes = {doc['data']: doc['freq'] for doc in find_all(COLLECTION_NODES)} # collection of docs to be inserted to nodes
    relations = {} # temporary, mutable dict of relations that incrementally gets updated for each article
    embeddings = {doc['data']: doc['embedding'] for doc in find_all(COLLECTION_EMBEDDINGS)}

    # get list of URLs from database so to avoid making duplicate entries
    visited = set(map(lambda doc:doc['url'], find_all(COLLECTION_NEWS)))
    visited_clone = visited.copy()
    
    # read scraped data from each of the Scrapy spiders
    for publisher in SCRAPER_MAPPINGS:
        filepath = os.path.join("webscraper", SCRAPER_MAPPINGS[publisher]['save_file'])

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

            for i, article_obj in enumerate(data): # url, title, date, content
                if i % 20 == 0:
                    print(100 * i / len(data))
                # check for null, check for visited
                if not (article_obj['url'] and article_obj['date'] and article_obj['content']) or article_obj['url'] in visited:
                    continue

                doc = {
                    'title': article_obj['title'],
                    'url': article_obj['url'],
                    'datetime': timestamp_to_epoch(article_obj['date']),
                    'publisher': publisher,
                    'keys': []
                }

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
 
                # topic modelling
                # for i, topic in enumerate(topics):
                #    doc[f'topic{i+1}'] = topic

                news_docs.append(doc)
                visited.add(article_obj['url'])

    for scraper in SCRAPER_MAPPINGS:
        filepath = os.path.join("webscraper", SCRAPER_MAPPINGS[scraper]['save_file'])

        # relation mapping
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

            for article_obj in data: # url, title, date, content
                # check for null, check for visited
                if not (article_obj['url'] and article_obj['date'] and article_obj['content']) or article_obj['url'] in visited_clone:
                    continue

                visited_clone.add(article_obj['url'])
                content = article_obj['title'] + ' ' + article_obj['content']
                content = content.lower()

                for joined_phrase in relations:
                    content = content.replace(joined_phrase.replace('__', ' '), joined_phrase)

                map_relations(content, relations, WINDOW_SIZE)
    
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
    visited_clone = set()
    for central in relations:
        adjacency = relations[central]

        for adjacent in adjacency:
            if (adjacent, central) in visited_clone or adjacent in replacement_map:
                continue

            relation_docs.append({
                'src': central.replace('__', ' '),
                'dst': adjacent.replace('__', ' '), 
                'weight': adjacency[adjacent]
            })
            
            visited_clone.add((central, adjacent))
    
    # update database
    delete_many(COLLECTION_NODES, {})

    if nodes:
        insert_many(COLLECTION_NODES, list(map(lambda item: {
            'data': item[0], 
            'freq': item[1]
        }, nodes.items()))) 
    if news_docs:
        insert_many(COLLECTION_NEWS, news_docs)
    if relation_docs:
        insert_many(COLLECTION_RELATIONS, relation_docs)
    
    if embeddings:
        insert_many(COLLECTION_EMBEDDINGS, list(map(lambda item: {
            'data': item[0], 
            'embedding': item[1]
        }, embeddings.items()))) 

    # update visited_urls.txt
    with open(VISITED_URLS_PATH, 'w') as g:
        g.write("\n".join(visited))

@app.get('/')
def read_root():
    return "/scraper/, /processor/"
    
@app.get('/cycle/')
def cycle(request: Request) -> dict:
    if not request.headers.get('API_SECRET_KEY') or not verify_origin(request.headers.get('API_SECRET_KEY')):
        return {'response': 'Invalid or missing secret key'}

    run_scraper() 

    # remove outdated documents (14 days or more)
    # print('Update count:', clean_up_by_days(14))

    # delete all data 
    delete_many(COLLECTION_NODES, {}) 
    delete_many(COLLECTION_NEWS, {}) 
    delete_many(COLLECTION_EMBEDDINGS, {})
    delete_many(COLLECTION_RELATIONS, {}) 
    
    run_nlp_processor()

    return {'response': 'success'}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=10000, reload=False)
    # run_scraper() 

    # remove outdated documents (14 days or more)
    # print('Update count:', clean_up_by_days(14))
    
    #run_nlp_processor()

# to run ...
# pip install -r API_requirements.txt
# uvicorn API:app --host 0.0.0.0 --port 10000
