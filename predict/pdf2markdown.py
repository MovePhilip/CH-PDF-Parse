import requests
import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import json
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3Tokenizer,AutoTokenizer,AutoModel,BertTokenizer,XLMRobertaTokenizerFast
from ..train.layoutlmv3_modeling import LayoutLMv3ForTokenClassification_custom
import torch.optim as optim
from collections import defaultdict
import wandb
import fitz
from PIL import Image,ImageDraw, ImageFont
from flask import Flask,request,jsonify,render_template
import copy
import math

app = Flask(__name__)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
#tokenizer = BertTokenizer(vocab_file="/data/zhangguangyin/pdf_parse/read_order_model/vocab.txt")
tokenizer=XLMRobertaTokenizerFast(vocab_file="/data/zhangguangyin/pdf_parse/read_order_model/customized_changed_vocab_model2_binary/sentencepiece.bpe.model")
tokenizer.add_tokens(["Figure__","Table__","Equation__"])
print(f"Custom vocabulary size: {len(tokenizer)}")
print(tokenizer.encode("Figure__"))
print(tokenizer.encode("Table__"))
print(tokenizer.encode("Equation__"))


model=LayoutLMv3ForTokenClassification_custom.from_pretrained("/data4/students/zhangguangyin/pdf_parse/customized_changed_vocab_model2_binary")
#model=LayoutLMv3ForTokenClassification_custom.from_pretrained("/root/private_data/results/checkpoint-9296")
model.resize_token_embeddings(len(tokenizer))
model.load_state_dict(torch.load("/data4/students/zhangguangyin/pdf_parse/save_result/dwa_resized_binary_augmented_detach-4-0.pt",weights_only=True))#,strict=False



model.eval()
model=model.to(device)


def pyMuPDF_fitz(pdfPath, imagePath):
    #print("imagePath=" + imagePath)
    pdfDoc = fitz.open(pdfPath)
    for pg in range(pdfDoc.page_count):
        page = pdfDoc[pg]
        rotate = int(0)
        # 每个尺寸的缩放系数为1.3，这将为我们生成分辨率提高2.6的图像。
        # 此处若是不做设置，默认图片大小为：792X612, dpi=96
        zoom_x = 2 # (1.33333333-->1056x816)   (2-->1584x1224)
        zoom_y = 2
        mat = fitz.Matrix(zoom_x, zoom_y).prerotate(rotate)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        if not os.path.exists(imagePath):  # 判断存放图片的文件夹是否存在
            os.makedirs(imagePath)  # 若图片文件夹不存在就创建
        pix.save(imagePath + '/' + 'images_%s.png' % pg)  # 将图片写入指定的文件夹内
    #endTime_pdf2img = datetime.datetime.now()  # 结束时间
    #print('pdf2img时间=', (endTime_pdf2img - startTime_pdf2img).seconds)
    return pdfDoc.page_count


def match_captions_to_images_tables(images_captions, tables_captions, table_images):
    #images_captions为两层嵌套list结构，最外面一层是以new_line为分割，里面一行是以每个segment分割
    #
    #tables_captions为两层嵌套list结构，最外面一层是以new_line为分割，里面一行是以每个segment分割
    #
    # print(f"images_captions length {len(images_captions)}")
    # print(f"tables_captions length {len(tables_captions)}")
    # print(f"images length {len(table_images['figure'])}")
    # print(f"tables length {len(table_images['table'])}")
    caption_threshold = 50  # 题注距离的阈值
    for item in table_images["table"]:
        item["point"]=[(item["bbox"][0]+item["bbox"][2])/2,item["bbox"][1]]#默认表的标注是在图片的上方，不能排除例外
        item["match_caption"]="未找到题注"

    for item in table_images["figure"]:
        item["point"]=[(item["bbox"][0]+item["bbox"][2])/2,item["bbox"][3]]#默认图片的标注是在图片的下方，不能排除例外
        item["match_caption"]="未找到题注"

    # 创建最终配对结果的字典
    match_result = {}
    
    #下面这两个函数都是用y来判断的
    # 检查题注是否在图片的下方
    def is_caption_below_image(caption_coord, image):
        _,  _,  _,image_bottom, = image
        _, caption_top,_, _ = caption_coord
        return abs(caption_top - image_bottom) < caption_threshold

    # 检查题注是否在表格的上方
    def is_caption_above_table(caption_coord, table):
        _, table_top, _, _ = table
        _, _, _,caption_bottom= caption_coord
        return abs(table_top - caption_bottom) < caption_threshold

    # 匹配图片和题注
    assigedlist=[]
    for i,image in enumerate(table_images["figure"]):
        matched = False
        ranklist=[]
        if len(images_captions)>0 and len(assigedlist)<len(images_captions):
            for j,caption in enumerate(images_captions):
                if j in assigedlist:
                    continue
                distance_agg=0
                for segment in caption:
                    distance_agg+=math.sqrt((image["point"][0] -segment[1][0]) ** 2 + (image["point"][1] - segment[1][1]) ** 2)
                ranklist.append([j,distance_agg])
            ranklist.sort(key=lambda x:x[0])#,reverse=True

            capstring=""
            matched_caption=images_captions[ranklist[0][0]]
            assigedlist.append(ranklist[0][0])
            for segments in matched_caption:
                capstring+=segments[0]

            table_images["figure"][i]["match_caption"]=capstring
            print(capstring)

        # if not matched:
        #     match_result[tuple(image)] = "题注未找到"

    # 匹配表格和题注
    assigedlist=[]
    for i,table in enumerate(table_images["table"]):
        matched = False
        ranklist=[]
        if len(tables_captions)>0 and len(assigedlist)<len(tables_captions):
            for j,caption in enumerate(tables_captions):
                if j in assigedlist:
                    continue
                distance_agg=0
                for segment in caption:
                    distance_agg+=math.sqrt((table["point"][0] -segment[1][0]) ** 2 + (table["point"][1] - segment[1][1]) ** 2)
                ranklist.append([j,distance_agg])
            ranklist.sort(key=lambda x:x[0])#reverse=True
            capstring=""
            matched_caption=tables_captions[ranklist[0][0]]
            assigedlist.append(ranklist[0][0])
            for segments in matched_caption:
                capstring+=segments[0]

            table_images["table"][i]["match_caption"]=capstring
            print(capstring)



    return table_images

# 准备数据集
class TokenClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
        #self.labels = labels

    def __getitem__(self, idx):
        try:
            item={ "p_text_id":torch.tensor([self.inputs["p_text_id"][idx]]),
                "p_x_y":torch.tensor([self.inputs["p_x_y"][idx]]),
                "p_position_id":torch.tensor([self.inputs["p_position_id"][idx]]),
                    "p_mask":torch.tensor([self.inputs["p_mask"][idx]]),
                    "p_page_position_id":torch.tensor([self.inputs["p_page_position_id"][idx]]),
                    "p_sentence_start":torch.tensor([self.inputs["p_sentence_start"][idx]]),
                    "p_text":self.inputs["p_text"][idx],
                        #"p_image_size":b_image_size,
                        #"p_sentence_start_mask":p_sentence_start_mask,
                    "p_original_x_y":self.inputs["p_original_x_y"][idx],
                        }
        except:
            print(self.inputs["p_text_id"][idx])



        # item = {key: torch.tensor(value[idx]) for key, value in self.inputs.items()}
        # # item['labels'] = torch.tensor(self.labels[idx])
        # # item.pop("attention_mask")
        return item

    def __len__(self):
        return len(self.inputs["p_text_id"])


def prepare_input(jsonlist):
    #extract and tokenize the input
    #下面这三个是基于字符个数的
    annotate_table_figure=[]

    b_text_id=[]
    b_x_y=[]
    b_position_id=[]
    b_page_position_id=[]
    b_sentence_start=[]#这个是把所有信息都会包含进去
    b_text=[]
    b_dir=[]
    b_image_id=[]
    b_original_x_y=[]
    b_image_size=[]
    b_mask=[]

    for i in range(len(jsonlist)):
        annotates_table_figure={"table":[],"figure":[]}
        s_text_id=[]
        s_x_y=[]
        s_position_id=[]
        s_sentence_start=[]
        s_order_start=[]
        s_text=[]
        s_original_x_y=[]

        if jsonlist[i]["total_page"]==1:
            pageposition=0
        else:
            pageposition=19*(jsonlist[i]["image_id"])//(jsonlist[i]["total_page"]-1)

        if pageposition>20 or pageposition<0:
            print(jsonlist[i]["image_id"]+1)
            print(jsonlist[i]["total_page"])
            print(jsonlist[i]["dir"])


        current_position=0

        for id2,item in enumerate(jsonlist[i]["annotate"]):
            if "structure_function" not in item:
                item["structure_function"]=item["region_type"]

            # if isinstance(item["text_region"][0],list):
            #     xy_temp=item["text_region"][0]+item["text_region"][2]#(x1,y1,x2,y2)
            # else:
            #     xy_temp=item["text_region"]
            if len(item["bbox"])==4:
                xy_temp=item["bbox"]
            else:
                print(item["bbox"])
                raise
            original_xy=copy.deepcopy(item["bbox"])

            if jsonlist[i]["image_size"][0]>jsonlist[i]["image_size"][1]:
                    max_side=jsonlist[i]["image_size"][0]
            else:
                    max_side=jsonlist[i]["image_size"][1]
            resize_scale=1024/max_side-0.00001
            xy_temp[0]=int(xy_temp[0]*resize_scale)
            xy_temp[1]=int(xy_temp[1]*resize_scale)
            xy_temp[2]=int(xy_temp[2]*resize_scale)
            xy_temp[3]=int(xy_temp[3]*resize_scale)

            # xy_temp[0]=int(xy_temp[0]*(1000/jsonlist[i]["image_size"][0]))
            # xy_temp[1]=int(xy_temp[1]*(1000/jsonlist[i]["image_size"][1]))
            # xy_temp[2]=int(xy_temp[2]*(1000/jsonlist[i]["image_size"][0]))
            # xy_temp[3]=int(xy_temp[3]*(1000/jsonlist[i]["image_size"][1]))
            
            if item["region_type"]=="figure":
                s_text_id.extend([0,25002,25002,25002])
                s_x_y.extend([xy_temp,xy_temp,xy_temp,xy_temp])
                s_position_id.extend([0,1,2,3])
                current_position+=4
                
                annotates_table_figure["figure"].append(item)
            elif item["region_type"]=="table":
                s_text_id.extend([0,25003,25003,25003])
                s_x_y.extend([xy_temp,xy_temp,xy_temp,xy_temp])
                s_position_id.extend([0,1,2,3])
                current_position+=4
                annotates_table_figure["table"].append(item)
            elif item["region_type"]=="equation":
                s_text_id.extend([0,25004])
                s_x_y.extend([xy_temp,xy_temp])
                s_original_x_y.append(original_xy)
                s_position_id.extend([0,1])
                s_sentence_start.append(current_position)
                s_text.append("###equation")
                current_position+=2
            else:
                encoded_info=tokenizer.encode_plus(item["text"],return_offsets_mapping=True,add_special_tokens=False)
                input_id=[0]+encoded_info["input_ids"]
                off_sets=encoded_info["offset_mapping"]
                s_text_id.extend(input_id)
                s_x_y.extend([xy_temp]*len(input_id))
                s_position_id.extend(range(len(input_id)))
                s_sentence_start.append(current_position)
                s_original_x_y.append(original_xy)
                s_text.append(item["text"])
                current_position+=len(input_id)



        if len(s_text_id)==0:#如果这个页面中没有任何文字，就直接跳过这个页面
            continue
        annotate_table_figure.append(annotates_table_figure)
        b_original_x_y.append(s_original_x_y)
        b_image_size.append(jsonlist[i]["image_size"])
        s_page_position_id=[pageposition]*len(s_text_id)
        b_text_id.append(s_text_id)
        b_x_y.append(s_x_y)
        b_position_id.append(s_position_id)
        b_page_position_id.append(s_page_position_id)
        b_text.append(s_text)
        b_sentence_start.append(s_sentence_start)
        b_mask.append([1,]*len(s_text_id))
    
    my_dataset=TokenClassificationDataset(inputs={"p_text_id":b_text_id,
                                            "p_x_y":b_x_y,
                                            "p_position_id":b_position_id,
                                            "p_mask":b_mask,
                                            "p_page_position_id":b_page_position_id,
                                            "p_sentence_start":b_sentence_start,
                                            "p_text":b_text,
                                            #"p_image_size":b_image_size,
                                            #"p_sentence_start_mask":p_sentence_start_mask,
                                            "p_original_x_y":b_original_x_y
                                            })
    return my_dataset,annotate_table_figure


@app.route('/')
def index():
    # 返回前端页面
    return render_template('index2.html')

@app.route('/upload', methods=['GET','POST'])
def upload_file():
    # 检查请求中是否有文件
    # 检查是否有文件在请求中
    if 'pdf' in request.files:
        file = request.files['pdf']
        print(file.filename)
        # 如果用户没有选择文件，浏览器也会提交一个空的文件名
        if file.filename == '':
            return 'No selected file'
        if file and file.filename.endswith('.pdf'):
            # 保存文件
            pdf_save_folder=f"/data/zhangguangyin/pdf_parse/read_order_model/api_test_folder/{file.filename}/"
            if not os.path.exists(pdf_save_folder):
                os.makedirs(pdf_save_folder)
            pdf_save_path=os.path.join(pdf_save_folder,file.filename)
            file.save(pdf_save_path)
        else:
            return 'Invalid file format'
    elif request.is_json:
           data=request.json
           pdf_save_folder=f"/data/zhangguangyin/pdf_parse/read_order_model/api_test_folder/"+data["filepath"].split("/")[-1][:-4]
           pdf_save_path=data["filepath"]
           print("收到json")

    
    page_num=pyMuPDF_fitz(pdf_save_path,pdf_save_folder)
    ocr_layout_result = requests.get("http://124.16.154.97:5008/upload2", json={"page_num":page_num,"image_path":pdf_save_folder,"table_image_equation_save_path":pdf_save_folder})
    if ocr_layout_result.status_code==200:
        annotate_json=ocr_layout_result.json()
        print("pdf版面分析和OCR已完成")
    predict_dataset,annotate_table_figure=prepare_input(annotate_json)
    
    mdstring=""
    parsed_json=[]
    previsou_string=[["",""]]
    matched_table_figure_total={"table":[],"figure":[]}
    for i,input_dict in enumerate(predict_dataset):
        annotate=[]
        try:
            with torch.no_grad():
                segment_type,newline,order_logits,token_type=model(input_ids=input_dict["p_text_id"].to(device),
                        bbox=input_dict["p_x_y"].to(device),
                        position_ids=input_dict["p_position_id"].to(device),
                        attention_mask=input_dict["p_mask"].to(device),
                        page_position_id=input_dict["p_page_position_id"].to(device),
                        sentence_start=input_dict["p_sentence_start"].to(device),
                        )
        except Exception as e:
            print(e)
            print(input_dict["p_position_id"].shape)
            print(input_dict["p_text_id"].shape)
            print(input_dict['p_x_y'].shape)
            print(input_dict["p_mask"].shape)
            # print(f"input_dict['p_x_y']{input_dict['p_x_y']}")
  

        for j in range(len(segment_type[0])):
            if input_dict["p_text"][j]=="###equation":
                annotate.append({"text":input_dict["p_text"][j].split("###_")[-1],
                                    "order":order_logits[0][j],
                                    "need-info":True,
                                    "new_line":True,
                                    "structure_function": "equation",
                                    "text_region":input_dict["p_original_x_y"][j]
                                    })
            else:
                try:
                    if segment_type[0][j]==3:
                        if newline[0][j]==0:
                            newline_=True
                        else:
                            newline_=False
                        annotate.append({"text":input_dict["p_text"][j],
                                                    "order":order_logits[0][j],
                                                    "need-info":True,
                                                    "new_line":newline_,
                                                    "structure_function": "text",
                                                    "text_region":input_dict["p_original_x_y"][j]
                                                    })
                        
                    if segment_type[0][j]==4:
                        if newline[0][j]==0:
                            newline_=True
                        else:
                            newline_=False
                        annotate.append({"text":input_dict["p_text"][j],
                                                    "order":order_logits[0][j],
                                                    "need-info":True,
                                                    "new_line":newline_,
                                                    "structure_function": "title",
                                                    "text_region":input_dict["p_original_x_y"][j]
                                                    })
                        
                    if segment_type[0][j]==1:
                        if newline[0][j]==0:
                            newline_=True
                        else:
                            newline_=False
                        try:
                            annotate.append({"text":input_dict["p_text"][j],
                                                        "order":order_logits[0][j],
                                                        "need-info":True,
                                                        "new_line":newline_,
                                                        "structure_function": "figure_caption",
                                                        "text_region":input_dict["p_original_x_y"][j]
                                                        })
                        except:
                            print(f"input_dict[p_original_x_y]:{input_dict['p_original_x_y']}")

                    if segment_type[0][j]==2:
                        if newline[0][j]==0:
                            newline_=True
                        else:
                            newline_=False
                        annotate.append({"text":input_dict["p_text"][j],
                                                    "order":order_logits[0][j],
                                                    "need-info":True,
                                                    "new_line":newline_,
                                                    "structure_function": "table_caption",
                                                    "text_region":input_dict["p_original_x_y"][j]
                                                    })
                    if segment_type[0][j]==5:
                        if newline[0][j]==0:
                            newline_=True
                        else:
                            newline_=False
                        annotate.append({"text":input_dict["p_text"][j],
                                                    "order":order_logits[0][j],
                                                    "need-info":True,
                                                    "new_line":newline_,
                                                    "structure_function": "author",
                                                    "text_region":input_dict["p_original_x_y"][j]
                                                    })
                    if segment_type[0][j]==6:
                        if newline[0][j]==0:
                            newline_=True
                        else:
                            newline_=False
                        annotate.append({"text":input_dict["p_text"][j],
                                                    "order":order_logits[0][j],
                                                    "need-info":True,
                                                    "new_line":newline_,
                                                    "structure_function": "abstract",
                                                    "text_region":input_dict["p_original_x_y"][j]
                                                    })
                    if segment_type[0][j]==7:
                        if newline[0][j]==0:
                            newline_=True
                        else:
                            newline_=False
                        annotate.append({"text":input_dict["p_text"][j],
                                                    "order":order_logits[0][j],
                                                    "need-info":True,
                                                    "new_line":newline_,
                                                    "structure_function": "institution",
                                                    "text_region":input_dict["p_original_x_y"][j]
                                                    })
                    if segment_type[0][j]==9:
                        if newline[0][j]==0:
                            newline_=True
                        else:
                            newline_=False
                        annotate.append({"text":input_dict["p_text"][j],
                                                    "order":order_logits[0][j],
                                                    "need-info":True,
                                                    "new_line":newline_,
                                                    "structure_function": "reference",
                                                    "text_region":input_dict["p_original_x_y"][j]
                                                    })
                    
                    if segment_type[0][j]==8:
                        if newline[0][j]==0:
                            newline_=True
                        else:
                            newline_=False
                        annotate.append({"text":input_dict["p_text"][j],
                                                    "order":order_logits[0][j],
                                                    "need-info":True,
                                                    "new_line":newline_,
                                                    "structure_function": "keyword",
                                                    "text_region":input_dict["p_original_x_y"][j]
                                                    })
                except:
                    print(len(newline))
                    print(len(newline[0]))
                    print(len(segment_type[0]))
                    raise



        annotate.sort(key=lambda x:x["order"],reverse=True)
        tablestring=[]
        table_caption_list=[]
        figurestring=[]
        figure_caption_list=[]
        newline=""
        for k in range(len(annotate)):
            if annotate[k]["structure_function"]=="table_caption":
                if annotate[k]["new_line"]==True:
                    if len(tablestring)==0:
                        tablestring=[[annotate[k]["text"],[(annotate[k]["text_region"][0]+annotate[k]["text_region"][2])/2,annotate[k]["text_region"][1]]]]                 
                    else:
                        table_caption_list.append(tablestring)
                        tablestring=[[annotate[k]["text"],[(annotate[k]["text_region"][0]+annotate[k]["text_region"][2])/2,annotate[k]["text_region"][1]]]]
                else:
                    tablestring.append([annotate[k]["text"],[(annotate[k]["text_region"][0]+annotate[k]["text_region"][2])/2,annotate[k]["text_region"][1]]])
                    # previoustring[0]+=annotate[k]["text"]




            elif annotate[k]["structure_function"]=="figure_caption":
                if annotate[k]["new_line"]==True:
                    # previoustring[0]=annotate[k]["text"]
                    # previoustring[1]="figure_caption"
                    if len(figurestring)==0:
                        figurestring=[[annotate[k]["text"],[(annotate[k]["text_region"][0]+annotate[k]["text_region"][2])/2,annotate[k]["text_region"][3]]]]                
                    else:
                        figure_caption_list.append(figurestring)
                        figurestring=[[annotate[k]["text"],[(annotate[k]["text_region"][0]+annotate[k]["text_region"][2])/2,annotate[k]["text_region"][3]]]]  
                else:
                    figurestring.append([annotate[k]["text"],[(annotate[k]["text_region"][0]+annotate[k]["text_region"][2])/2,annotate[k]["text_region"][3]]])
                    # previoustring[0]+=annotate[k]["text"]


            elif annotate[k]["structure_function"]=="title":
                if annotate[k]["new_line"]==True:
                     parsed_json.append({"text":annotate[k]["text"],"function_type":"title"})
                     mdstring=mdstring+"   \n\n # "+annotate[k]["text"]
                    #  previoustring=previoustring[1]+annotate[k]["text"]
                    #  previoustring[1]="title"
                else:
                     mdstring=mdstring+annotate[k]["text"]
                    #  previoustring[0]+=annotate[k]["text"]
                     if len(parsed_json)>0:
                         parsed_json[-1]["text"]+=annotate[k]["text"]
                     else:
                         parsed_json.append({"text":annotate[k]["text"],"function_type":"title"})

            

            elif annotate[k]["structure_function"]=="text":
                if annotate[k]["new_line"]==True:
                     parsed_json.append({"text":annotate[k]["text"],"function_type":"text"})
                     mdstring=mdstring+"   \n\n"+annotate[k]["text"]
                    #  previoustring=previoustring[1]+annotate[k]["text"]
                    #  previoustring[1]="text"
                else:
                     mdstring=mdstring+annotate[k]["text"]
                    #  previoustring[0]+=annotate[k]["text"]
                     if len(parsed_json)>0:
                         parsed_json[-1]["text"]+=annotate[k]["text"]
                     else:
                         parsed_json.append({"text":annotate[k]["text"],"function_type":"text"})
            
            
            elif annotate[k]["structure_function"]=="author":
                if annotate[k]["new_line"]==True:
                     parsed_json.append({"text":annotate[k]["text"],"function_type":"author"})
                     mdstring=mdstring+"   \n\n"+annotate[k]["text"]
                    #  previoustring=previoustring[1]+annotate[k]["text"]
                    #  previoustring[1]="text"
                else:
                     mdstring=mdstring+annotate[k]["text"]
                    #  previoustring[0]+=annotate[k]["text"]
                     if len(parsed_json)>0:
                         parsed_json[-1]["text"]+=annotate[k]["text"]
                     else:
                         parsed_json.append({"text":annotate[k]["text"],"function_type":"author"})


            elif annotate[k]["structure_function"]=="reference":
                if annotate[k]["new_line"]==True:
                     parsed_json.append({"text":annotate[k]["text"],"function_type":"reference"})
                     mdstring=mdstring+"   \n\n"+annotate[k]["text"]
                    #  previoustring=previoustring[1]+annotate[k]["text"]
                    #  previoustring[1]="text"
                else:
                     mdstring=mdstring+annotate[k]["text"]
                    #  previoustring[0]+=annotate[k]["text"]
                     if len(parsed_json)>0:
                         parsed_json[-1]["text"]+=annotate[k]["text"]
                     else:
                         parsed_json.append({"text":annotate[k]["text"],"function_type":"reference"})

            elif annotate[k]["structure_function"]=="keyword":
                if annotate[k]["new_line"]==True:
                     parsed_json.append({"text":annotate[k]["text"],"function_type":"keyword"})
                     mdstring=mdstring+"   \n\n"+annotate[k]["text"]
                    #  previoustring=previoustring[1]+annotate[k]["text"]
                    #  previoustring[1]="text"
                else:
                     mdstring=mdstring+annotate[k]["text"]
                    #  previoustring[0]+=annotate[k]["text"]
                     if len(parsed_json)>0:
                         parsed_json[-1]["text"]+=annotate[k]["text"]
                     else:
                         parsed_json.append({"text":annotate[k]["text"],"function_type":"keyword"})
            
            elif annotate[k]["structure_function"]=="institution":
                if annotate[k]["new_line"]==True:
                     parsed_json.append({"text":annotate[k]["text"],"function_type":"institution"})
                     mdstring=mdstring+"   \n\n"+annotate[k]["text"]
                    #  previoustring=previoustring[1]+annotate[k]["text"]
                    #  previoustring[1]="text"
                else:
                     mdstring=mdstring+annotate[k]["text"]
                    #  previoustring[0]+=annotate[k]["text"]
                     if len(parsed_json)>0:
                         parsed_json[-1]["text"]+=annotate[k]["text"]
                     else:
                         parsed_json.append({"text":annotate[k]["text"],"function_type":"institution"})


            elif annotate[k]["structure_function"]=="abstract":
                if annotate[k]["new_line"]==True:
                     parsed_json.append({"text":annotate[k]["text"],"function_type":"abstract"})
                     mdstring=mdstring+"   \n\n"+annotate[k]["text"]
                    #  previoustring=previoustring[1]+annotate[k]["text"]
                    #  previoustring[1]="text"
                else:
                     mdstring=mdstring+annotate[k]["text"]
                    #  previoustring[0]+=annotate[k]["text"]
                     if len(parsed_json)>0:
                         parsed_json[-1]["text"]+=annotate[k]["text"]
                     else:
                         parsed_json.append({"text":annotate[k]["text"],"function_type":"abstract"})


            elif annotate[k]["structure_function"]=="equation":
                    mdstring=mdstring+"![]("+annotate[k]["text"]+")  \n\n"


        if len(figurestring)>0:
            figure_caption_list.append(figurestring)
        if len(tablestring)>0:
            table_caption_list.append(tablestring)
        
        #if data["output_table_image"]==True:
        matched_table_figure=match_captions_to_images_tables(figure_caption_list,table_caption_list,annotate_table_figure[i])
        mdstring=mdstring+"\n\n"
        for item in matched_table_figure["table"]:
            mdstring=mdstring+item["match_caption"]+"  \n\n"
            mdstring=mdstring+"![]("+item["img_save_path"]+")  \n\n"
        for item in matched_table_figure["figure"]:
            mdstring=mdstring+"![]("+item["img_save_path"]+")  \n\n"
            mdstring=mdstring+item["match_caption"]+"  \n\n"
        matched_table_figure_total["table"].extend(matched_table_figure["table"])
        matched_table_figure_total["figure"].extend(matched_table_figure["figure"])
    # with open("","w")as f:
    #     json.dump(parsed_json,f,ensure_ascii=False,indent=1)
    for item in matched_table_figure_total["table"]:
        parsed_json.append({"text":item["match_caption"],"image_path":item["img_save_path"],"function_type":"table_caption"})

        
    for item in matched_table_figure_total["figure"]:
        parsed_json.append({"text":item["match_caption"],"image_path":item["img_save_path"],"function_type":"figure_caption"})

        
    return {"parse_json":parsed_json,"md":mdstring,}

    

            



"""
第一个第二种预测模型
    无关信息0
    图片题注1
    表格题注2
    正常信息3
    标题信息4
"""


if __name__=="__main__":
     app.run('0.0.0.0', port=4502, debug=False)





















