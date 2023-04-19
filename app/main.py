from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from dateutil import parser
from hashlib import sha256
import uvicorn
import json
import os
from database_handler import *
from keywords_extractor import extract_keywords
from relation_mapper import map_relations
from tqdm import tqdm


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

VISITED_URLS_PATH = 'visited_urls.txt'
SCRAPY_PROJ_PATH = 'webscraper'
SHA256_SECRET_KEY = 'd8b04a8a85e1cf3e4797366aa8d77769963fdbef167ed9e8cd2cb220f5287629'

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


@app.get('/')
def read_root():
    return "/scraper/, /processor/"


def run_scraper():
    '''
    Run Scrapy spiders & save scraped data as JSON files in {SCRAPY_PROJ_PATH}/{SCRAPY_PROJ_PATH}
    A CRON job will call this endpoint every fixed time interval.

    :param str secret: API secret key. If valid, then scrape, else ignore this GET request
    :return: Object that states the Scrapy spiders that were executed.
    :rtype: dict
    '''
    os.chdir(SCRAPY_PROJ_PATH)
    print("[CD]", os.getcwd())

    for scraper in SCRAPER_MAPPINGS.values():
        path = scraper['save_file']

        # clear previous scrapings
        if (path in os.listdir()):
            os.remove(path)

        os.system(f'scrapy crawl -o {path} -t json {scraper["spider"]}')
        print("[CMD]", f'scrapy crawl -o {path} -t json {scraper["spider"]}')

    os.chdir('../')
    print("[CD]", os.getcwd())


def run_nlp_processor():
    '''
    Remove outdated data.
    Read scraped data outputs and conduct keyword extraction, followed by topic modelling and relation extraction.
    Store node and news data to MongoDB.
    Update list of visited URLs to prevent repeat work.

    :return: Object that states the scraped data that have been processed.
    :rtype: dict
    '''
    news_docs = [] # news documents to be inserted
    relation_docs = []
    nodes = {}
    relations = {}

    # get list of URLs from database so to avoid making duplicate entries
    visited = set(map(lambda doc:doc['url'], find_all(COLLECTION_NEWS)))
    visited = set()
    visited_clone = visited.copy()
    
    # read scraped data from each of the Scrapy spiders
    for scraper in SCRAPER_MAPPINGS:
        filepath = os.path.join("webscraper", SCRAPER_MAPPINGS[scraper]['save_file'])

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

            for article_obj in tqdm(data): # url, title, date, content
                # check for null, check for visited
                if not (article_obj['url'] and article_obj['date'] and article_obj['content']) or article_obj['url'] in visited:
                    continue

                doc = {
                    'title': article_obj['title'],
                    'url': article_obj['url'],
                    'datetime': timestamp_to_epoch(article_obj['date'])
                }
                
                # extract keyphrases, word embeddings, doc embedding from content. Title used for seeding
                keyphrases, _, _ = extract_keywords(article_obj['content'].replace('\n', ''), article_obj['title'], KEYWORDS_PER_ARTICLE)
                
                # store keywords in graph
                for i, (phrase, similarity) in enumerate(keyphrases):
                    nodes[phrase] = nodes.get(phrase, 0) + 1
                    doc[f'key{i+1}'] = phrase

                    joined_phrase = phrase.replace(' ', '__')
                    if joined_phrase not in relations:
                        relations[joined_phrase] = {}
        
                '''
                # topic modelling
                for i, topic in enumerate(topics):
                    doc[f'topic{i+1}'] = topic
                '''

                news_docs.append(doc)
                visited.add(article_obj['url'])
                
    # add on to existing nodes in database
    for doc in find_all(COLLECTION_NODES):
        data = doc['data']
        freq = doc['freq']
        nodes[data] = nodes.get(data, 0) + freq

    # relation mapping
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

        for article_obj in tqdm(data): # url, title, date, content
            # check for null, check for visited
            if not (article_obj['url'] and article_obj['date'] and article_obj['content']) or article_obj['url'] in visited_clone:
                continue

            visited_clone.add(article_obj['url'])
            content = article_obj['title'] + ' ' + article_obj['content']

            for joined_phrase in relations:
                content = content.replace(joined_phrase.replace('__', ' '), joined_phrase)

            map_relations(content, relations, 30)
    
    visited_clone = set()
    for central in relations:
        adjacency = relations[central]

        for adjacent in adjacency:
            if (adjacent, central) in visited_clone:
                continue
            
            relation_docs.append({'src': central.replace('__', ' '), 'dest': adjacent.replace('__', ' '), 'weight': adjacency[adjacent]})
            visited_clone.add((central, adjacent))

    # save to database
    delete_many(COLLECTION_NODES, {}) # delete all nodes

    if nodes:
        insert_many(COLLECTION_NODES, list(map(lambda item:{'data': item[0], 'freq': item[1]}, nodes.items()))) # replace with new nodes
    if news_docs:
        insert_many(COLLECTION_NEWS, news_docs)
    if relation_docs:
        insert_many(COLLECTION_RELATIONS, relation_docs)

    # update visited_urls.txt
    with open(VISITED_URLS_PATH, 'w') as g:
        g.write("\n".join(visited))
        

@app.get('/cycle/')
def cycle(request: Request) -> dict:
    if not request.headers.get('API_SECRET_KEY') or not verify_origin(request.headers.get('API_SECRET_KEY')):
        return {'response': 'Invalid or missing secret key'}

    # os.system is blocking so need to sleep
    run_scraper() 

    # remove outdated documents (14 days or more)
    print('Update count:', clean_up_by_days(14))
    
    run_nlp_processor()

    return {'response': '200 OK'}


if __name__ == '__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=10000, reload=False)

# to run ...
# pip install -r API_requirements.txt
# uvicorn API:app --host 0.0.0.0 --port 10000
