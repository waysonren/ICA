import json
stop_words = {'!', ',', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', '-', '.', '/', ':', ';', '<', '=', '>', '?',
              '@', '[', ']', '^', '_', '`', '{', '|', '}', '~', 'that', 'our', 'little', 'herself', 'from', 'hadn',
              'those', 'most', 'why', 'further', 'am', 'i', 'I', 'either', 'or', 'mustn', "must'n", "must'", "must' n", 'who', 'Who', 'her', 'everything', 'don', "don'", "don' t", "don't",
              'be', 'me', 'yours', 'same', 'all', 'own', 'and', 'him', 'these', 'for', 'had', 'too', 'we', 'We', 'so', 'hers',
              'down', 'shouldn', "shouldn'", "shouldn't", "shouldn' t", 'could', 'once', 'isn', 'yet', 'but', 'hasn', "hasn'", "hasn't", "hasn' t", 'anybody', 'because', 'couldn', "couldn'", "couldn't", "couldn' t",
              'something', 'were', 'while', 'us', 'another', 'are', 'you', 'here', 'just', 'themselves', 'against',
              'might', 'only', 'itself', 'must', 'their', 'yourself', 'of', 'about', 'his', 'although', 'them',
              'anything', 'someone', 'has', 'now', 'theirs', 'a', 'in', 'up', 'did', 'weren', "weren'", "weren't", "weren' t", 'between', 'as', 'ours',
              'won', "won'", "won't", "won' t", 'other', 'any', 't', "'t", "' t", 'this', 'none', 'under', 'is', 'such', 'during', 'was', 'at', 'several',
              'whom', 'may', 's', 'nobody', 'again', 'if', 'being', 'the', 'neither', 'much', 'nothing', 'to', 'myself',
              'with', 'been', 'both', 'they', 'They', 'wasn', "wasn'", "wasn' t", "wasn't", 'yourselves', 'unless', 'before', 'also', 'some', 'should',
              'your', 'then', 'below', 'through', 'off', 'on', 'over', 'its', 'my', 'an', 'mine', 'even', 'when',
              'after', 'doesn', "doesn'", "doesn't", "doesn' t", 'no one', 'haven', 'since', 'many', 'everyone', 'very', 'more', 'whose', 'didn', "didn'", "didn't", "didn' t",
              'having', 'do', 'there', 'he', 'He', 'anyone', 'above', 'each', 'whether', 'can', 'would', 'aren', 'few', 'it',
              'how', 'into', 'every', 'what', 'she', 'not', 'ourselves', 'shall', 'needn', "needn'", "needn't", "needn' t", 'somebody', 'mightn', "mightn'", "mightn' t", "mightn't",
              'where', 'himself', 'than', 'by', 'doing', 'no', 'which', 'wouldn', "wouldn'", "wouldn't", "wouldn' t", 'until', 'will', 'shan', 'have',
              'does', 'everybody', "each other", "one another", "”", "“"}
new_entity = {}
with open("entity_frequencies.json", "r") as f, open("entity_filter_3more.json", "w") as f_w:
    entity_frequencies = json.load(f)
    for k, v in entity_frequencies.items():
        if v > 3 and k not in stop_words:
            new_entity[k] = v
    f_w.write(json.dumps(new_entity, indent=4, ensure_ascii=False))
