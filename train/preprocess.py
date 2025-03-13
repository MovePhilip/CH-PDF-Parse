import os
import json
import torch
from torch.utils.data import Subset

def preparedataset(json_dir,tokenizer):
    jsonlist=[]
    for path in json_dir:
        try:
            with open(path,"r")as f:
                  textjson=json.load(f)
            #imgjson=json.load(open(tablepic_jsonpath))
        except:
            print(path)
            assert 1==2   
        # if len(textjson)!=len(imgjson): 
        #     print(f"dir:{dir}")

        for i in range(len(textjson)):
            textjson[i]["total_page"]=len(textjson)
            textjson[i]["dir"]=str(path)
        jsonlist.extend(textjson)
        #tablepicjsonlist.extend(imgjson)

    #assert len(jsonlist)==len(tablepicjsonlist)
    print(f"total number of sample is: {len(jsonlist)}")


    #extract and tokenize the input
    #下面这三个是基于字符个数的
    b_text_id=[]
    b_x_y=[]
    b_position_id=[]
    b_reverse_position_id=[]
    #下面这四个的长度是基于ocr的条的数目的，预测时会抽取每个ocr条目的第一个字符来预测
    b_type_id=[]#type_id和sentence_start的长度是相同的，这个事包括false information的
    b_sentence_start=[]#这个是把所有信息都会包含进去
    b_order_id_text=[]#和order_start共用，也会预测caption的顺序信息
    b_order_id_caption=[]
    b_newline_start=[]#newline_start包括caption，但是不包括被省略的信息
    b_newline_id=[]#newline的长度和order_start的长度是相同的，这个是包括caption的
    b_order_start_text=[]#order_start1不包括被省略信息,只包括正常的text，不包括caption
    b_order_start_caption=[]#order_start2不包括被省略信息,不包括正常的text，只包括caption
    b_page_position_id=[]
    #b_pixel_value=[]
    b_citation_start=[]#这个只包括正文的信息，不包括标题和题注，不包括被忽略的信息
    b_citation_id=[]

    b_order_text_mask=[]
    b_order_caption_mask=[]
    b_newline_mask=[]
    b_type_mask=[]

    b_text_citation_start=[]#这个只包括正文的信息，不包括标题和题注，不包括被忽略的信息
    b_text_citation_id=[]
    b_caption_citation_start=[]#这个只包括正文的信息，不包括标题和题注，不包括被忽略的信息
    b_caption_citation_id=[]

    for i in range(len(jsonlist)):
        s_text_id=[]
        s_x_y=[]
        s_position_id=[]
        s_reverse_position_id=[]
        s_type_id=[]
        s_newline_id=[]
        s_sentence_start=[]
        s_order_id_text=[]
        s_order_id_caption=[]
        s_order_start_text=[]
        s_order_start_caption=[]
        s_newline_start=[]

        s_citation_start=[]
        s_citation_id=[]

        s_order_mask_text=[]
        s_order_mask_caption=[]
        s_newline_mask=[]
        s_type_mask=[]

        s_text_citation_start=[]
        s_caption_citation_start=[]
        s_text_citation_id=[]
        s_caption_citation_id=[]
        
        pageposition=19*(jsonlist[i]["image_id"])//(jsonlist[i]["total_page"]-1)
        if pageposition>20 or pageposition<0:
            print(jsonlist[i]["image_id"]+1)
            print(jsonlist[i]["total_page"])
            print(jsonlist[i]["dir"])
        #pixel_value=processor.preprocess(Image.open(image_path),apply_ocr=False,return_tensors="pt")
        current_position=0
        citation_list=[]

        if "position" not in jsonlist[i]:
            jsonlist[i]["position"]=[]
        for k in jsonlist[i]["position"]:
            citation_list.append(k[0])


        
        for id2,item2 in enumerate(jsonlist[i]["annotate"]):
            try:
                if "structure_function" not in item2:
                      item2["structure_function"]=item2["region_type"]


                if isinstance(item2["text_region"][0],list):
                    xy_temp=item2["text_region"][0]+item2["text_region"][2]#(x1,y1,x2,y2)
                else:
                    xy_temp=item2["text_region"]

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


                if item2["structure_function"]=="table_caption":
                    encoded_info=tokenizer.encode_plus(item2["text"],return_offsets_mapping=True,add_special_tokens=False)
                    input_id=[0]+encoded_info["input_ids"]
                    off_sets=encoded_info["offset_mapping"]
                    s_text_id.extend(input_id)
                    s_x_y.extend([xy_temp]*len(input_id))
                    s_position_id.extend(range(len(input_id)))
                    s_reverse_position_id.extend(range(len(input_id)-1,-1,-1))
                    s_sentence_start.append(current_position)
                    s_newline_start.append(current_position)
                    s_order_start_caption.append(current_position)
                    s_caption_citation_start.extend([current_position+l for l in range(1,len(input_id))])
                    s_citation_start.extend([current_position+l for l in range(1,len(input_id))])
                    temp_citation_id=[0]*(len(input_id)-1)#+[-100,]
                    #这里不需要考虑在同一个文本片段是否出现了两次标注
                    if id2 in citation_list:
                        for l in jsonlist[i]["position"]:
                            if id2==l[0]:
                                #确定标注起始位置
                                end=l[2]
                                if l[1]==-1:
                                    start=0
                                else:
                                    start=l[1]
                                #确定标注标签
                                if l[3]==1:
                                    if l[1]==-1:
                                        B=3
                                        I=3
                                    else:
                                        B=1
                                        I=3
                                else:
                                    if l[1]==-1:
                                        B=4
                                        I=4
                                    else:
                                        B=2
                                        I=4
                                #确定标注的token
                                entity_started = False
                                for idx, (offset_start, offset_end) in enumerate(off_sets):
                                    # 检查 token 是否与实体重叠
                                    if offset_start < end and offset_end > start:
                                        if not entity_started:
                                            temp_citation_id[idx] = B
                                            entity_started = True
                                        else:
                                            temp_citation_id[idx] = I
                                # for id3,(token_start, token_end) in enumerate(off_sets):
                                #     if start<token_start and end >=token_start and end <token_end:
                                #         temp_citation_id[id3]=I
                                #     if start>=token_start and start <token_end and end >=token_start and end <token_end:
                                #         temp_citation_id[id3]=B
                                #     if start>=token_start and start<token_end and end >=token_end:
                                #         temp_citation_id[id3]=B
                    #temp_citation_id[-1]=-100
                    #temp_citation_id[0]=-100
                    s_citation_id.extend(temp_citation_id)
                    s_caption_citation_id.extend(temp_citation_id)
                    current_position+=len(input_id)
                    s_type_id.append(2)
                    if item2["new_line"]==True:
                        s_newline_id.append(0)
                    else:
                        s_newline_id.append(1)
                    if len(s_order_id_caption)==0:
                        s_order_id_caption.append(0)
                    else:
                        s_order_id_caption.append(s_order_id_caption[-1]+1)
                    s_order_mask_text.append(0)
                    s_order_mask_caption.append(1)
                    s_newline_mask.append(1)
                    s_type_mask.append(1)
                    
                    


                elif item2["structure_function"]=="figure_caption":
                    encoded_info=tokenizer.encode_plus(item2["text"],return_offsets_mapping=True,add_special_tokens=False)
                    input_id=[0]+encoded_info["input_ids"]
                    off_sets=encoded_info["offset_mapping"]
                    s_text_id.extend(input_id)
                    s_x_y.extend([xy_temp]*len(input_id))
                    s_position_id.extend(range(len(input_id)))
                    s_reverse_position_id.extend(range(len(input_id)-1,-1,-1))
                    s_sentence_start.append(current_position)
                    s_newline_start.append(current_position)
                    s_order_start_caption.append(current_position)
                    s_citation_start.extend([current_position+l for l in range(1,len(input_id))])
                    s_caption_citation_start.extend([current_position+l for l in range(1,len(input_id))])
                    temp_citation_id=[0]*(len(input_id)-1)#+[-100,]

                    if id2 in citation_list:
                        for l in jsonlist[i]["position"]:
                            if id2==l[0]:
                                end=l[2]
                                if l[1]==-1:
                                    start=0
                                else:
                                    start=l[1]
                                if l[3]==1:
                                    if l[1]==-1:
                                        B=3
                                        I=3
                                    else:
                                        B=1
                                        I=3
                                else:
                                    if l[1]==-1:
                                        B=4
                                        I=4
                                    else:
                                        B=2
                                        I=4
                                entity_started = False
                                for idx, (offset_start, offset_end) in enumerate(off_sets):
                                    # 检查 token 是否与实体重叠
                                    if offset_start < end and offset_end > start:
                                        if not entity_started:
                                            temp_citation_id[idx] = B
                                            entity_started = True
                                        else:
                                            temp_citation_id[idx] = I
                                # for id3,(token_start, token_end) in enumerate(off_sets):
                                #     if start<token_start and end >=token_start and end <token_end:
                                #         temp_citation_id[id3]=I
                                #     if start>=token_start and start <token_end and end >=token_start and end <token_end:
                                #         temp_citation_id[id3]=B
                                #     if start>=token_start and start<token_end and end >=token_end:
                                #         temp_citation_id[id3]=B
                    #temp_citation_id[-1]=-100
                    #temp_citation_id[0]=-100
                    s_citation_id.extend(temp_citation_id)
                    s_caption_citation_id.extend(temp_citation_id)
                    current_position+=len(input_id)
                    s_type_id.append(1)
                    if item2["new_line"]==True:
                        s_newline_id.append(0)
                    else:
                        s_newline_id.append(1)
                    if len(s_order_id_caption)==0:
                        s_order_id_caption.append(0)
                    else:
                        s_order_id_caption.append(s_order_id_caption[-1]+1)
                    s_order_mask_text.append(0)
                    s_order_mask_caption.append(1)
                    s_newline_mask.append(1)
                    s_type_mask.append(1)






                elif item2["structure_function"]=="figure":
                        s_text_id.extend([0,25002,25002,25002])#99
                        s_position_id.extend([0,1,2,3])
                        s_reverse_position_id.extend([3,2,1,0])
                        s_x_y.extend([xy_temp,xy_temp,xy_temp,xy_temp])
                        current_position+=4
                elif item2["structure_function"]=="table":
                        s_text_id.extend([0,25003,25003,25003])#100
                        s_position_id.extend([0,1,2,3])
                        s_reverse_position_id.extend([3,2,1,0])
                        s_x_y.extend([xy_temp,xy_temp,xy_temp,xy_temp])
                        current_position+=4
                elif item2["structure_function"]=="equation":
                        s_text_id.extend([0,25004])#98
                        s_position_id.extend([0,1])
                        s_reverse_position_id.extend([1,0])
                        s_order_start_text.append(current_position)#len(s_text_id)-1
                        if len(s_order_id_text)==0:
                            s_order_id_text.append(0)
                        else:
                            s_order_id_text.append(s_order_id_text[-1]+1)
                        s_x_y.extend([xy_temp,xy_temp])
                        current_position+=2
                        s_order_mask_text.append(1)
                        s_order_mask_caption.append(0)
                        s_newline_mask.append(0)
                        s_type_mask.append(0)

                elif item2["structure_function"]=="title" and item2["need-info"]==True:
                        input_id=tokenizer.encode(item2["text"],add_special_tokens=False)
                        input_id=[0]+input_id
                        s_text_id.extend(input_id)
                        s_x_y.extend([xy_temp]*len(input_id))
                        s_position_id.extend(range(len(input_id)))
                        s_reverse_position_id.extend(range(len(input_id)-1,-1,-1))
                        s_sentence_start.append(current_position)
                        s_order_start_text.append(current_position)#len(s_text_id)-1
                        s_newline_start.append(current_position)
                        current_position+=len(input_id)
                        s_type_id.append(4)
                        
                        if item2["new_line"]==True:#这个new_line的设置还是有点问题，所有的token都得加啊，不能只加普通的text啊
                            s_newline_id.append(0)
                        else:
                            s_newline_id.append(1)
                        if len(s_order_id_text)==0:
                            s_order_id_text.append(0)
                        else:
                            s_order_id_text.append(s_order_id_text[-1]+1)
                        s_order_mask_text.append(1)
                        s_order_mask_caption.append(0)
                        s_newline_mask.append(1)
                        s_type_mask.append(1)

                elif item2["structure_function"]=="author" and item2["need-info"]==True:
                        input_id=tokenizer.encode(item2["text"],add_special_tokens=False)
                        input_id=[0]+input_id
                        s_text_id.extend(input_id)
                        s_x_y.extend([xy_temp]*len(input_id))
                        s_position_id.extend(range(len(input_id)))
                        s_reverse_position_id.extend(range(len(input_id)-1,-1,-1))
                        s_sentence_start.append(current_position)
                        s_order_start_text.append(current_position)#len(s_text_id)-1
                        s_newline_start.append(current_position)
                        current_position+=len(input_id)
                        s_type_id.append(5)
                        
                        if item2["new_line"]==True:#这个new_line的设置还是有点问题，所有的token都得加啊，不能只加普通的text啊
                            s_newline_id.append(0)
                        else:
                            s_newline_id.append(1)
                        if len(s_order_id_text)==0:
                            s_order_id_text.append(0)
                        else:
                            s_order_id_text.append(s_order_id_text[-1]+1)
                        s_order_mask_text.append(1)
                        s_order_mask_caption.append(0)
                        s_newline_mask.append(1)
                        s_type_mask.append(1)

                elif item2["structure_function"]=="abstract"  and item2["need-info"]==True:
                        input_id=tokenizer.encode(item2["text"],add_special_tokens=False)
                        input_id=[0]+input_id
                        s_text_id.extend(input_id)
                        s_x_y.extend([xy_temp]*len(input_id))
                        s_position_id.extend(range(len(input_id)))
                        s_reverse_position_id.extend(range(len(input_id)-1,-1,-1))
                        s_sentence_start.append(current_position)
                        s_order_start_text.append(current_position)#len(s_text_id)-1
                        s_newline_start.append(current_position)
                        current_position+=len(input_id)
                        s_type_id.append(6)
                        
                        if item2["new_line"]==True:#这个new_line的设置还是有点问题，所有的token都得加啊，不能只加普通的text啊
                            s_newline_id.append(0)
                        else:
                            s_newline_id.append(1)
                        if len(s_order_id_text)==0:
                            s_order_id_text.append(0)
                        else:
                            s_order_id_text.append(s_order_id_text[-1]+1)
                        s_order_mask_text.append(1)
                        s_order_mask_caption.append(0)
                        s_newline_mask.append(1)
                        s_type_mask.append(1)


                elif item2["structure_function"]=="institution":
                        input_id=tokenizer.encode(item2["text"],add_special_tokens=False)
                        input_id=[0]+input_id
                        s_text_id.extend(input_id)
                        s_x_y.extend([xy_temp]*len(input_id))
                        s_position_id.extend(range(len(input_id)))
                        s_reverse_position_id.extend(range(len(input_id)-1,-1,-1))
                        s_sentence_start.append(current_position)
                        s_order_start_text.append(current_position)#len(s_text_id)-1
                        s_newline_start.append(current_position)
                        current_position+=len(input_id)
                        s_type_id.append(7)
                        
                        if item2["new_line"]==True:#这个new_line的设置还是有点问题，所有的token都得加啊，不能只加普通的text啊
                            s_newline_id.append(0)
                        else:
                            s_newline_id.append(1)
                        if len(s_order_id_text)==0:
                            s_order_id_text.append(0)
                        else:
                            s_order_id_text.append(s_order_id_text[-1]+1)
                        s_order_mask_text.append(1)
                        s_order_mask_caption.append(0)
                        s_newline_mask.append(1)
                        s_type_mask.append(1)


                elif item2["structure_function"]=="keyword"  and item2["need-info"]==True:
                        input_id=tokenizer.encode(item2["text"],add_special_tokens=False)
                        input_id=[0]+input_id
                        s_text_id.extend(input_id)
                        s_x_y.extend([xy_temp]*len(input_id))
                        s_position_id.extend(range(len(input_id)))
                        s_reverse_position_id.extend(range(len(input_id)-1,-1,-1))
                        s_sentence_start.append(current_position)
                        s_order_start_text.append(current_position)#len(s_text_id)-1
                        s_newline_start.append(current_position)
                        current_position+=len(input_id)
                        s_type_id.append(8)
                        
                        if item2["new_line"]==True:#这个new_line的设置还是有点问题，所有的token都得加啊，不能只加普通的text啊
                            s_newline_id.append(0)
                        else:
                            s_newline_id.append(1)
                        if len(s_order_id_text)==0:
                            s_order_id_text.append(0)
                        else:
                            s_order_id_text.append(s_order_id_text[-1]+1)
                        s_order_mask_text.append(1)
                        s_order_mask_caption.append(0)
                        s_newline_mask.append(1)
                        s_type_mask.append(1)

                elif item2["structure_function"]=="reference"  and item2["need-info"]==True:
                        input_id=tokenizer.encode(item2["text"],add_special_tokens=False)
                        input_id=[0]+input_id
                        s_text_id.extend(input_id)
                        s_x_y.extend([xy_temp]*len(input_id))
                        s_position_id.extend(range(len(input_id)))
                        s_reverse_position_id.extend(range(len(input_id)-1,-1,-1))
                        s_sentence_start.append(current_position)
                        s_order_start_text.append(current_position)#len(s_text_id)-1
                        s_newline_start.append(current_position)
                        current_position+=len(input_id)
                        s_type_id.append(9)
                        
                        if item2["new_line"]==True:#这个new_line的设置还是有点问题，所有的token都得加啊，不能只加普通的text啊
                            s_newline_id.append(0)
                        else:
                            s_newline_id.append(1)
                        if len(s_order_id_text)==0:
                            s_order_id_text.append(0)
                        else:
                            s_order_id_text.append(s_order_id_text[-1]+1)
                        s_order_mask_text.append(1)
                        s_order_mask_caption.append(0)
                        s_newline_mask.append(1)
                        s_type_mask.append(1)

                elif item2["need-info"]==True and item2["structure_function"] not in ["figure_caption","table_caption"]:
                        encoded_info=tokenizer.encode_plus(item2["text"],return_offsets_mapping=True,add_special_tokens=False)
                        input_id=[0]+encoded_info["input_ids"]
                        off_sets=encoded_info["offset_mapping"]
                        s_text_id.extend(input_id)
                        s_x_y.extend([xy_temp]*len(input_id))
                        s_position_id.extend(range(len(input_id)))
                        s_reverse_position_id.extend(range(len(input_id)-1,-1,-1))
                        s_sentence_start.append(current_position)
                        s_order_start_text.append(current_position)#len(s_text_id)-1
                        s_newline_start.append(current_position)
                        s_citation_start.extend([current_position+l for l in range(1,len(input_id))])
                        s_text_citation_start.extend([current_position+l for l in range(1,len(input_id))])
                        temp_citation_id=[0]*(len(input_id)-1)#+[-100,]

                        if id2 in citation_list:
                            for l in jsonlist[i]["position"]:
                                if id2==l[0]:
                                    end=l[2]
                                    if l[1]==-1:
                                        start=0
                                    else:
                                        start=l[1]
                                    if l[3]==1:
                                        if l[1]==-1:
                                            B=3
                                            I=3
                                        else:
                                            B=1
                                            I=3
                                    else:
                                        if l[1]==-1:
                                            B=4
                                            I=4
                                        else:
                                            B=2
                                            I=4

                                    entity_started = False
                                    for idx, (offset_start, offset_end) in enumerate(off_sets):
                                        # 检查 token 是否与实体重叠
                                        if offset_start < end and offset_end > start:
                                            if not entity_started:
                                                temp_citation_id[idx] = B
                                                entity_started = True
                                            else:
                                                temp_citation_id[idx] = I
                                    # for id3,(token_start, token_end) in enumerate(off_sets):
                                    #     if start<token_start and end >=token_start and end <token_end:
                                    #         temp_citation_id[id3]=I
                                    #     if start>=token_start and start <token_end and end >=token_start and end <token_end:
                                    #         temp_citation_id[id3]=B
                                    #     if start>=token_start and start<token_end and end >=token_end:
                                    #         temp_citation_id[id3]=B
                        #temp_citation_id[-1]=-100
                        #temp_citation_id[0]=-100

                        s_citation_id.extend(temp_citation_id)
                        s_text_citation_id.extend(temp_citation_id)
                        current_position+=len(input_id)
                        s_type_id.append(3)
                        #len(s_text_id)-1
                        if item2["new_line"]==True:#这个new_line的设置还是有点问题，所有的token都得加啊，不能只加普通的text啊
                            s_newline_id.append(0)
                        else:
                            s_newline_id.append(1)
                        if len(s_order_id_text)==0:
                            s_order_id_text.append(0)
                        else:
                            s_order_id_text.append(s_order_id_text[-1]+1)
                        s_order_mask_text.append(1)
                        s_order_mask_caption.append(0)
                        s_newline_mask.append(1)
                        s_type_mask.append(1)

                        

                elif item2["need-info"]==False:
                        input_id=tokenizer.encode(item2["text"],add_special_tokens=False)
                        input_id=[0]+input_id
                        s_text_id.extend(input_id)
                        s_x_y.extend([xy_temp]*len(input_id))
                        s_position_id.extend(range(len(input_id)))
                        s_reverse_position_id.extend(range(len(input_id)-1,-1,-1))
                        s_sentence_start.append(current_position)
                        current_position+=len(input_id)
                        s_type_id.append(0)
                        s_order_mask_text.append(0)
                        s_order_mask_caption.append(0)
                        s_newline_mask.append(0)
                        s_type_mask.append(1)

            except:
                print(item2)
                raise

        

        if len(s_text_id)>4200:
            print(jsonlist[i]["image_id"])
            print(jsonlist[i]["dir"])
            continue

        #这里要处理下页面没有任何text的情况
        if len(s_type_id)==0:
            continue
        assert len(s_order_id_text)==len(s_order_start_text)
        assert len(s_newline_start)==len(s_newline_id)
        assert len(s_type_id)==len(s_sentence_start)

        s_order_id_text=[300-i for i in s_order_id_text]
        s_order_id_caption=[300-i for i in s_order_id_caption]

        s_page_position_id=[pageposition]*len(s_text_id)

        b_text_id.append(s_text_id)
        b_x_y.append(s_x_y)
        b_position_id.append(s_position_id)
        b_reverse_position_id.append(s_reverse_position_id)
        b_type_id.append(s_type_id)
        
        b_newline_id.append(s_newline_id)
        b_sentence_start.append(s_sentence_start)
        #这里要处理下页面没有任何需要的文字的情况
        b_order_id_text.append(s_order_id_text)
        b_order_id_caption.append(s_order_id_caption)
        b_order_start_text.append(s_order_start_text)
        b_order_start_caption.append(s_order_start_caption)

        b_newline_start.append(s_newline_start)
        b_page_position_id.append(s_page_position_id)
        #b_pixel_value.append(pixel_value["pixel_values"][0])
        b_citation_start.append(s_citation_start)
        b_citation_id.append(s_citation_id)

        b_order_text_mask.append(s_order_mask_text)
        b_order_caption_mask.append(s_order_mask_caption)
        b_newline_mask.append(s_newline_mask)
        b_type_mask.append(s_type_mask)

        b_text_citation_start.append(s_text_citation_start)#如果为0的话这个会是个长度为0的列表
        b_text_citation_id.append(s_text_citation_id)
        b_caption_citation_start.append(s_caption_citation_start)#如果为0的话这个会是个长度为0的列表
        b_caption_citation_id.append(s_caption_citation_id)
    
    
    
    mydataset=TokenClassificationDataset(inputs={"p_text_id":b_text_id,
                                            "p_x_y":b_x_y,
                                            "p_position_id":b_position_id,
                                            "p_reverse_position_id":b_reverse_position_id,
                                            "p_type_id":b_type_id,
                                            "p_order_id_text":b_order_id_text,
                                            "p_order_id_caption":b_order_id_caption,
                                            "p_newline_id":b_newline_id,
                                            "p_sentence_start":b_sentence_start,
                                            "p_order_start_text":b_order_start_text,
                                            "p_order_start_caption":b_order_start_caption,
                                            "p_newline_start":b_newline_start,
                                            "p_page_position_id":b_page_position_id,
                                            "p_citation_start":b_citation_start,
                                            "p_citation_id":b_citation_id,

                                            "p_newline_mask":b_newline_mask,
                                            "p_order_text_mask":b_order_text_mask,
                                            "p_order_caption_mask":b_order_caption_mask,
                                            "p_type_mask":b_type_mask,

                                            "p_text_citation_start":b_text_citation_start,
                                            "p_text_citation_id":b_text_citation_id,
                                            "p_caption_citation_start":b_caption_citation_start,
                                            "p_caption_citation_id":b_caption_citation_id
                                            })
    return mydataset


# 准备数据集
class TokenClassificationDataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs
        #self.labels = labels

    def __getitem__(self, idx):
        item = {key: value[idx] for key, value in self.inputs.items()}
        # item['labels'] = torch.tensor(self.labels[idx])
        # item.pop("attention_mask")
        return item

    def __len__(self):
        return len(self.inputs["p_text_id"])
    


def collate_fn(batch):
    

    maxlength=0
    maxsegmentlength=0
    maxnewlinelength=0
    maxorderlength_text=0
    maxorderlength_caption=0
    maxcitationlength=0

    for item in batch:
        if len(item['p_text_id'])>maxlength:
            maxlength=len(item['p_text_id'])
        
        if len(item['p_type_id'])>maxsegmentlength:
            maxsegmentlength=len(item['p_type_id'])

        if len(item['p_newline_id'])>maxnewlinelength:
            maxnewlinelength=len(item['p_newline_id'])
        
        if len(item['p_order_id_text'])>maxorderlength_text:
            maxorderlength_text=len(item['p_order_id_text'])

        if len(item['p_order_id_caption'])>maxorderlength_caption:
            maxorderlength_caption=len(item['p_order_id_caption'])

        if len(item['p_citation_id'])>maxcitationlength:
            maxcitationlength=len(item['p_citation_id'])


    
    if maxnewlinelength==0:
        maxnewlinelength=10
    if maxorderlength_text==0:
        maxorderlength_text=10
    if maxorderlength_caption==0:
        maxorderlength_caption=10
    if maxcitationlength==0:
        maxcitationlength=10

    p_text_id=[]
    p_x_y=[]
    p_position_id=[]
    p_reverse_position_id=[]
    p_mask=[]
    p_type_id=[]
    p_order_id_text=[]
    p_order_id_caption=[]
    p_newline_id=[]
    p_sentence_start=[]
    p_order_start_text=[]
    p_order_start_caption=[]
    p_order_mask_text=[]
    p_order_mask_caption=[]
    p_newline_start=[]
    p_page_position_id=[]
    #p_pixel_value=[]
    p_citation_start=[]
    p_citation_id=[]

    for item in batch:
        p_text_id.append(item['p_text_id']+[0]*(maxlength-len(item['p_text_id'])))
        p_x_y.append(item['p_x_y']+[[0,0,0,0],]*(maxlength-len(item['p_x_y'])))
        p_position_id.append(item['p_position_id']+[0]*(maxlength-len(item['p_position_id'])))
        p_reverse_position_id.append(item['p_reverse_position_id']+[0]*(maxlength-len(item['p_reverse_position_id'])))
        p_mask.append([1]*len(item['p_text_id'])+[0]*(maxlength-len(item['p_text_id'])))
        p_page_position_id.append(item['p_page_position_id']+[0]*(maxlength-len(item['p_page_position_id'])))

        p_type_id.append(item['p_type_id']+[-100]*(maxsegmentlength-len(item['p_type_id'])))
        p_sentence_start.append(item['p_sentence_start']+[0]*(maxsegmentlength-len(item['p_sentence_start'])))

        p_newline_id.append(item['p_newline_id']+[-100]*(maxnewlinelength-len(item['p_newline_id'])))
        p_newline_start.append(item['p_newline_start']+[0]*(maxnewlinelength-len(item['p_newline_start'])))

        p_order_id_text.append(item['p_order_id_text']+[0]*(maxorderlength_text-len(item['p_order_id_text'])))
        p_order_start_text.append(item['p_order_start_text']+[0]*(maxorderlength_text-len(item['p_order_start_text'])))
        p_order_mask_text.append([1]*len(item['p_order_start_text'])+[0]*(maxorderlength_text-len(item['p_order_start_text'])))

        p_order_id_caption.append(item['p_order_id_caption']+[0]*(maxorderlength_caption-len(item['p_order_id_caption'])))
        p_order_start_caption.append(item['p_order_start_caption']+[0]*(maxorderlength_caption-len(item['p_order_start_caption'])))
        p_order_mask_caption.append([1]*len(item['p_order_start_caption'])+[0]*(maxorderlength_caption-len(item['p_order_start_caption'])))

        p_citation_start.append(item['p_citation_start']+[0]*(maxcitationlength-len(item['p_citation_start'])))
        p_citation_id.append(item['p_citation_id']+[-100]*(maxcitationlength-len(item['p_citation_id'])))

    
        #p_pixel_value.append(item['p_pixel_vlaue'])



    # 使用 torch.stack 来处理张量数据，对于列表数据直接作为列表返回
    return {
        'input_ids': torch.tensor(p_text_id,dtype=torch.int32),
        'bbox': torch.tensor(p_x_y,dtype=torch.int32),
        'position_ids': torch.tensor(p_position_id,dtype=torch.int32),
        'reverse_position_ids': torch.tensor(p_reverse_position_id,dtype=torch.int32),

        'attention_mask': torch.tensor(p_mask,dtype=torch.int32),
        'page_position_id': torch.tensor(p_page_position_id,dtype=torch.int32),
        'sentence_start': torch.tensor(p_sentence_start,dtype=torch.int64),

        'order_id_text':  torch.tensor(p_order_id_text).float(),
        'order_start_text': torch.tensor(p_order_start_text,dtype=torch.int64),
        'order_mask_text': torch.tensor(p_order_mask_text,dtype=torch.int32),

        'order_id_caption':  torch.tensor(p_order_id_caption).float(),
        'order_start_caption': torch.tensor(p_order_start_caption,dtype=torch.int64),
        'order_mask_caption': torch.tensor(p_order_mask_caption,dtype=torch.int32),

        'newline_id': torch.tensor(p_newline_id),#,dtype=torch.int32
        'newline_start': torch.tensor(p_newline_start,dtype=torch.int64),
        'type_ids': torch.tensor(p_type_id),#,dtype=torch.int32

        'citation_start':torch.tensor(p_citation_start),
        'citation_id':torch.tensor(p_citation_id)
    }













def filtered_order_dataset(predataset):
    # 获取符合条件的索引
    valid_indices=[]
    for idx in range(len(predataset)):
        if len(predataset[idx]["p_order_id_text"])>1 or len(predataset[idx]["p_order_id_caption"])>1:
            valid_indices.append(idx)
    return Subset(predataset, valid_indices)


def filtered_citation_dataset(predataset):
    # 获取符合条件的索引
    valid_indices=[]
    for idx in range(len(predataset)):
        if len(predataset[idx]["p_citation_id"])>1 or len(predataset[idx]["p_caption_citation_id"])>1:
         valid_indices.append(idx)
    return Subset(predataset, valid_indices)        

def filtered_newline_dataset(predataset):
    # 获取符合条件的索引
    valid_indices=[]
    for idx in range(len(predataset)):
        if len(predataset[idx]["p_newline_id"])>0:
         valid_indices.append(idx)
    return Subset(predataset, valid_indices)   

