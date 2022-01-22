import json
import re
import pandas as pd
pd.options.display.max_rows = 100

path = "/home/harong/mbti_project/crawl_data/crawl_res_5.json"
with open(path,"r") as f : 
    temp = json.load(f)
    end,start = max(temp.keys()), min(temp.keys())
    writer = []
    for k in range(int(start),int(end)+1) : 
        if str(k) in temp.keys() : 
            writer.append(temp[str(k)]['category'])
df = pd.DataFrame(writer,columns = ['writer'])
print(df)
f.close()