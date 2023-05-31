from dateutil import parser

from cryptography.fernet import Fernet
import numpy as np

from settings import (
    FERNET_KEY,
    FERNET_SECRET
)

fernet = Fernet(FERNET_KEY)

def verify_origin(secret: str) -> bool:
    '''
    Check if request has the correct API secret key

    :param str secret: Secret key from request
    :return: True if authorized
    :rtype: bool
    '''
    return FERNET_SECRET == fernet.decrypt(secret.encode('utf-8').decode('utf-8'))

def timestamp_to_epoch(timestamp) -> int:
    '''
    ISO 8601 datestring to unix timestamp
    :param str timestamp: ISO 8601 datestring
    :return: Unix timestamp
    :rtype: int
    '''
    if timestamp:
        return int(parser.parse(timestamp).timestamp())
    
def map_relations(content: str, relations: dict, window_size: int):
    '''
    Identify relations between keywords based on window_size
    Modifies keywords in-place
    :param str content: Article content
    :param dict[str, list[str, int]] relations: Adjacency list of relations between key phrases
    :param int window_size: Sliding window size
    '''
    content = content.replace('\n', ' ').split()    
    window: dict[str, int] = {}
    st, end = 0, window_size

    for i in range(min(window_size, len(content))):
        word = content[i]

        # we only care about keywords
        if word in relations:
            window[word] = window.get(word, 0) + 1        
    
    for central in window:
        for peripheral in window:
            if central == peripheral:
                continue

            relations[central][peripheral] = relations[central].get(peripheral, 0) + window[peripheral]
            relations[peripheral][central] = relations[peripheral].get(central, 0) + window[central]

    for i in range(len(content) - window_size):
        if content[st] in relations:
            window[content[st]] -= 1

            if window[content[st]] == 0:
                del window[content[st]]

        if content[end] in relations:
            central = content[end]
            window[central] =  window.get(central, 0) + 1

            for peripheral in window:
                if peripheral != central:
                    relations[central][peripheral] = relations[central].get(peripheral, 0) + 1
                    relations[peripheral][central] = relations[peripheral].get(central, 0) + 1
        
        st += 1
        end += 1

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

def get_cosine_similarity(a, b) -> float:
    '''
    :param List a: Word embedding array of word/phrase A
    :param List b: Word embedding array of word/phrase B
    :return: Similarity score
    :rtype: float
    '''
    a = np.array(a)
    b = np.array(b)
    numerator = np.dot(a, b.transpose())
    a_norm = np.sqrt(np.sum(a ** 2))
    b_norm = np.sqrt(np.sum(b ** 2))
    denominator = a_norm * b_norm
    cosine_similarity = numerator / denominator

    return cosine_similarity