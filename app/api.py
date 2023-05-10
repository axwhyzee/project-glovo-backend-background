import requests
import os


KEYWORD_EXTRACTION_MICROSERVICE_URL = os.environ.get('KEYWORD_EXTRACTION_MICROSERVICE_URL')

def extract_keywords(content, heading, top_n):
    '''
    Extract keywords + word embeddings via keyword extraction microservice API
    :param str content: Content of article
    :param str heading: Title of article
    :param int top_n: Number of keywords to extract
    :return: List of [keyword, word_embedding] elements
    :rtype: list[list[str, list[int]]]
    '''
    params = {
        "content": content,
        "heading": heading,
        "top_n": top_n
    }
    
    r = requests.get(KEYWORD_EXTRACTION_MICROSERVICE_URL, params=params)
    
    try:
        return r.json()
    except:
        return []