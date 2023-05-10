def map_relations(content: str, relations: dict, window_size: int):
    '''
    Identify relations between keywords based on window_size
    Modifies keywords in-place
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