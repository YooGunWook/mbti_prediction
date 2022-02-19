import pandas as pd
import os
import re
import requests
from selenium import webdriver
from bs4 import BeautifulSoup as bs
import time
import tqdm
from pyvirtualdisplay import Display
import json

def crawl_mbti_data() :
    tmp_li = dict()
    cnt = 1222
    for i in tqdm.tqdm(range(101200,0,-1)) : 
        if i == 99776 or i == 99775 or i == 99777: 
            continue
        addr = "https://cafe.naver.com/mbticafe/ArticleRead.nhn?clubid=11856775&boardtype=L&articleid="+str(i)+"&referrerAllArticles=true"
        driver.get(addr)
        try : 
            driver.switch_to.frame('cafe_main')
            html = driver.page_source
            soup = bs(html, "html.parser")
        except : 
            continue
        try : 
            category = soup.findAll("a", {"class": "m-tcol-c"})[3].text
            title = soup.find("span",{"class":"b m-tcol-c"}).text
            writer = soup.find("a",{"m-tcol-c b"}).text
            article = soup.find("div", {"class": "tbody m-tcol-c"}).text.replace('\u200b', ' ').replace('\n','').strip()

            tmp_li[i] = {'writer':writer,'category':category,'title':title,'article':article}
        except : 
            continue
        if len(tmp_li) % 100 == 0 :
            with open(f'/home/harong/mbti_project/crawl_data/crawl_res_{cnt}.json','w',encoding='utf-8') as f:
                json.dump(tmp_li,f,ensure_ascii=False)
            f.close()
            cnt += 1
            tmp_li = dict()
            print('Last articleid : ', i)

if __name__ == '__main__' : 
    display = Display(visible=0, size=(1920,1080))
    display.start()
    path = '/home/harong/chromedriver'
    driver = webdriver.Chrome(path)
    crawl_mbti_data()