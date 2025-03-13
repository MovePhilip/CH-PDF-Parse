# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 00:29:37 2021
@author: 1222
"""
import cv2
import matplotlib.pyplot as plt 
import matplotlib.image as img
import os
#import paddlehub as hub
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from paddleocr import PaddleOCR, draw_ocr
from paddleocr import PPStructure,save_structure_res,draw_structure_result,PaddleOCR

from flask import Flask,request
import json
import numpy as np
import fitz
from PIL import Image,ImageDraw, ImageFont
from flask import Flask,request,jsonify


app = Flask(__name__)
params = {
    'det_model_dir': '/data/zhangguangyin/pdf_parse/ocrv4_server_det/ch_PP-OCRv4_det_server_infer',
    'rec_model_dir': '/data/zhangguangyin/pdf_parse/ocrv4_server_rec/ch_PP-OCRv4_rec_server_infer',
    #'cls_model_dir': '/ppocr_img/ch_ppocr_mobile_v2.0_cls_slim_infer',
    'use_gpu': False,
    #'use_angle_cls': True
}

# 初始化 OCR 实例
ocr_engine = PaddleOCR(det_model_dir="/data/zhangguangyin/pdf_parse/ch_PP-OCRv3_det_infer",#'/data/zhangguangyin/pdf_parse/ocrv4_server_det/ch_PP-OCRv4_det_server_infer',
        rec_model_dir='/data/zhangguangyin/pdf_parse/ch_PP-OCRv4_rec_infer/ch_PP-OCRv4_rec_infer',#'/data/zhangguangyin/pdf_parse/ocrv4_server_rec/ch_PP-OCRv4_rec_server_infer',
        use_gpu=True,
        use_dilation = False,
        rec_image_shape='3, 50, 360',
        det_db_box_thresh=0.3,
        det_db_unclip_ratio=1.7,
        lang="ch",
        show_log=True,
        det_limit_side_len=1200,
        #det_db_score_mode="slow",
        use_mp=False,
        total_process_num=4)

#ocr = PaddleOCR(use_angle_cls=False, lang="ch",use_gpu=True,det_limit_type="max",det_algorithm="ch_PP-OCRv4_server_det",rec_algorithm="ch_PP-OCRv4_server_rec")  # need to run only once to download and load model into memory det_limit_side_len=2200 
layout_engine = PPStructure(
        #det_model_dir="ch_ppocr_server_v2.0_det",
        #rec_model_dir="ch_ppocr_server_v2.0_rec",
        table_model_dir="ch_ppstructure_mobile_v2.0_SLANet",
        layout_model_dir="picodet_lcnet_x1_0_fgd_layout_cdla",
        lang="ch",
        recovery=True,
        show_log=False,
        ocr=False,
        use_gpu=True
            )

# 创建一个路由和视图函数的映射
# 创建的是根路由
# 访问根路由，就会执行hello_world这个函数
#这个函数的主输入是图片不是pdf，pdf转图片在另一个函数中
@app.route('/upload',methods=["GET","POST"])
def hello_world():  # put application's code here
    b64ImgData = request.data
    #print(type(b64ImgData))
    ocr_result = ocr_engine.ocr(b64ImgData, cls=False)
    print(ocr_result)
    return {"ocr_result":ocr_result[0]}

@app.route('/upload2',methods=["GET","POST"])
def hello_world2():  # put application's code here
    image_num=request.json["page_num"]
    image_path=request.json["image_path"]
    table_image_equation_save_path=request.json["table_image_equation_save_path"]

    save_dict=[]
    for i in range(image_num):
            img_path = image_path+"/images_"+str(i)+".png"
            img = cv2.imread(img_path)
            image=Image.open(img_path)

            #开始识别pdf页面中的图片，表格和公式
            layout_result = layout_engine(img)
            temp=[]
            temp2={"tables":[],
                   "images":[],
                   "equations":[]
                   }

            #格式化识别结果，保存识别的图片
            draw=ImageDraw.Draw(image)
            for j,region in enumerate(layout_result):
                if region["type"] in ["table","figure","equation"]:#"figure_caption","table_caption"
                    corr1=region["bbox"]
                    if corr1[0]>corr1[2]:
                        corr1[2]=corr1[0]+5
                    if corr1[1]>corr1[3]:
                        corr1[3]=corr1[1]+5
                    # 切割图像
                    cropped_image = image.crop(region["bbox"])
                    img_save_path=table_image_equation_save_path+"/images_"+str(i)+"_"+region["type"]+"_"+str(j)+".png"
                    # 保存切割后的图像
                    if region["type"]=="table":
                         temp2["tables"].append(img_save_path)
                    elif region["type"]=="figure":
                         temp2["images"].append(img_save_path)
                    elif region["type"]=="equation":
                         temp2["equations"].append(img_save_path)

    
                    cropped_image.save(img_save_path)
                    draw.rectangle(corr1,fill=(255,255,255))
                    temp.append({"bbox":corr1,"region_type":region["type"],"text":"","img_save_path":img_save_path})
            

            #开始识别图片中的文本
            ocr_result = ocr_engine(np.array(image))
            print(ocr_result[1])

            for j,region in enumerate(ocr_result[0]):
                    corr2=region[0].tolist()+region[2].tolist()
                    if corr2[0]>corr2[2]:
                        corr2[2]=corr2[0]+1
                    if corr2[1]>corr2[3]:
                        corr2[3]=corr2[1]+1
                    temp.append({"text":ocr_result[1][j][0],"region_type":"text","bbox":corr2})
                    

            save_dict.append({"image_id":i,"annotate":temp,"image_size":image.size,"table_figures_equations":temp2,"total_page":image_num})
            print("程序拿到了ocr的结果")

    return save_dict


# 运行代码
if __name__ == '__main__':
    app.run(debug=False,port=5008)











