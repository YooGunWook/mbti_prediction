import json
import os
import pprint


def preproc_article(path):
    with open(path, 'r') as f:
        data = json.load(f)
    article_json = {}
    for num in data:
        article = data[num]
        article_json[num] = article['article']
    pprint.pprint(article_json)
    
if __name__ == '__main__':
    list_data = os.listdir('./crawl_data')
    path = './crawl_data/' + 'crawl_res_4.json'
    preproc_article(path)