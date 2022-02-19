import json
import os
import pprint
import re
import pandas as pd
import emoji
import sys
import tqdm
from kss import split_sentences

def remove_emoji(string):
    return emoji.get_emoji_regexp().sub(u' emoji ', string)

def find_writer(writer) :
    cnt = writer.count('x')
    cnt += writer.count('o')
    s = set(writer)
    can = [['i','e'],['s','n'],['f','t'],['p','j']]
    can2 = ['i','e','s','n','f','t','p','j','x','o']
    # 잘못 뽑힌건 제외
    for i in s : 
        if i not in can2 : 
            return 
    ans = set()
    li = [[] for _ in range(4)]
    for i in s : 
        for k in range(4) : 
            if i in can[k] : 
                li[k].append(i)
    null_cnt = 0
    for i in li : 
        if not i :
            null_cnt += 1
    if null_cnt > 2 : 
        return
    for i in range(4) : 
        if not li[i] and ('x' in s or 'o' in s) and cnt != 0 : 
            li[i] += can[i]
            cnt -= 1
    for a in li[0] : 
        for b in li[1] : 
            for c in li[2] : 
                for d in li[3] : 
                    ans.add(a+b+c+d)
    return ans

def preproc_article(article):
    # 글이 없는 것 처리
    if not article :
        return
    # 인터넷 링크 제거
    article = ' '.join(re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$\-@\.&+:/?=]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',"",article).split())
    article = ' '.join(re.sub('(www|m).(?:[a-zA-Z]|[0-9]|[$\-@\.&+:/?=]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',"",article).split())
    article = ' '.join(re.sub('(cafe).(?:[a-zA-Z]|[0-9]|[$\-@\.&+:/?=]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',"",article).split())
    # 괄호 대괄호 제거
    article = re.sub('\([^)]*\)','',article)
    article = re.sub('\[[^)]*\]','',article)
    # 이모지 제거
    article = remove_emoji(article)
    # 띄어쓰기 긴거 하나로 처리
    article = ' '.join(article.split())
    # 20글자 미만 제외
    if len(article) < 20 : 
        return
    try : 
        article = split_sentences(article)
        return article
    except : 
        return
    
def preproc_writer(writer) :
    # 대문자-> 소문자로 
    writer = writer.lower()
    # 괄호 안에 네이버 아이디 삭제
    writer = re.sub('\([^)]*\)','',writer)
    # 애니그램 삭제
    writer = re.sub('[0-9]w[0-9]','',writer)
    with open('/home/harong/mbti_project/data/mbti_kor_to_eng.json','r') as f : 
        kor = json.load(f)
    f.close()
    final_writer = set()
    # 한국말 처리
    for k in kor : 
        if k in writer : 
            final_writer.add(kor[k])
    # enftp, enffp, ixfp 같은 것 처리
    p = re.compile('[e|i|x]\S+[j|p|x]')
    if p.findall(writer) : 
        for i in p.findall(writer) : 
            if find_writer(i) : 
                final_writer |= find_writer(i)
    return list(final_writer)

def preprocess_all(path) : 
    global csv_cnt, preprocess_csv
    path = os.path.abspath('./crawl_data/'+path)
    with open(path, 'r') as f:
        data = json.load(f)
    f.close()
    for articleid in data:
        tmp = data[articleid]
        article = preproc_article(tmp['title'] + ' ' + tmp['article'])
        if not article : 
            continue
        category_in = ["고민, 연애, 진로 상담","궁합○관계 소통","긍정 일기","사진 자랑","신변 잡기"\
            ,"심리 관련 질문","심리 유형 질문","심리테스트 및 설문","에니어그램 소통"\
                ,"유형 감별","음악 자랑","INFJ ♡ ENFJ ","INFP ♧ ENFP","INTJ ◇ ENTJ ","INTP ♧ ENTP"\
                    ,"ISFJ ♡ ESFJ ","ISFP ♤ ESFP ","ISTJ ◇ ESTJ ","ISTP ♤ ESTP","MBTI 일반"]
        if tmp['category'] not in category_in : 
            continue
        category = tmp['category']
        writer = preproc_writer(tmp['writer'])
        if not writer :
            continue
        for i in writer : 
            preprocess_csv.append([category,articleid,article,i])
            if len(preprocess_csv) % 1000 == 0 :
                df = pd.DataFrame(preprocess_csv,columns= ['category','articleid','article','writer'])
                df.to_csv(f'/home/harong/mbti_project/data/preprocessed_{csv_cnt}.csv',encoding='utf-8',index=False)
                preprocess_csv = list()
                csv_cnt += 1

    
if __name__ == '__main__':
    list_data = ['crawl_res_'+str(i)+'.json' for i in range(18,1324)]
    preprocess_csv = list()
    csv_cnt = 1
    for path in tqdm.tqdm(list_data) : 
        preprocess_all(path)