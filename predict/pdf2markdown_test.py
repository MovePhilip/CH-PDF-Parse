import requests
import os
import json
from tqdm import tqdm


url = 'http:/localhost:4502/upload'




root_path="/data4/students/zhangguangyin/ch_multidiscipinepdf/results/data"
subfolders = [f.path for f in os.scandir(root_path) if f.is_dir()]

for subfolder in subfolders:
    files = [f.path for f in os.scandir(subfolder) if f.is_file()]
    parsed_pdf_savepath1=os.path.join(subfolder,"parsed_json_own")
    parsed_pdf_savepath2=os.path.join(subfolder,"parsed_md_own")
    if not os.path.isdir(parsed_pdf_savepath1):
         os.mkdir(parsed_pdf_savepath1)
    if not os.path.isdir(parsed_pdf_savepath2):
         os.mkdir(parsed_pdf_savepath2)
    for file in files:
        pdf_path="test.pdf"
        save_folder=""
        data = {'filepath': file}  # 将文件打开并放入一个字典中
        response = requests.post(url=url,json=data)  # 发送POST请求并传递文件data=data, 
        #mdstring=response.json()["md"]
        parsed_json=response.json()["parse_json"]
        mdstring=response.json()["md"]
        parse_json_path=os.path.join(parsed_pdf_savepath1,file.split()[-1][:-4]+".json")
        parse_md_path=os.path.join(parsed_pdf_savepath2,file.split()[-1][:-4]+".md")
        #保存markdown结果
        with open(parse_md_path,"w",encoding="utf-8") as f:
            f.write(mdstring)
        #保存json结果
        with open(parse_json_path,"w",encoding="utf-8") as f:
            json.dump(parsed_json,f,ensure_ascii=False,indent=1)




