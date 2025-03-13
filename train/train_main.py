import os
os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"
os.environ['NCCL_TIMEOUT'] = '1200'  # 例如设置为 1200 秒
os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
import json
import torch
from torch.utils.data import Dataset
from transformers import LayoutLMv3Tokenizer,AutoTokenizer,AutoModel,BertTokenizer,LayoutLMv3ImageProcessor,XLMRobertaTokenizerFast
from transformers import TrainingArguments, Trainer
from layoutlmv3_modeling import LayoutLMv3ForTokenClassification_custom
import torch.optim as optim
import wandb
from PIL import Image,ImageDraw,ImageFont
from torch.optim.lr_scheduler import CosineAnnealingLR,MultiStepLR
import re
import shutil
from transformers.integrations import WandbCallback
import logging
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist
from contextlib import nullcontext
import numpy as np
import pickle

from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,classification_report
from seqeval.metrics import classification_report as seqeval_classification_report
from seqeval.scheme import IOB2
import shutil
from eval_main_function import eval_model

from preprocess import preparedataset,collate_fn

os.environ["WANDB_PROJECT"]="my_project"









class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler,
        gpu_id: int,
        save_every: int,
        accumulation_steps: int,
        description:str,
        loss_type: str
    ) -> None:
        self.gpu_id = gpu_id

        self.model = model.to(gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.scheduler=scheduler
        self.save_every = save_every
            # Only initialize wandb on the main process
        newconfig=model.config.to_dict()
        # newconfig.update({
        #         "learning_rate": learning_rate,
        #         "epochs": num_epochs,
        #         "accumulation_steps":accumulation_steps
        #     })
        self.world_size=torch.distributed.get_world_size()
        self.accumulation_steps=accumulation_steps
        # if self.gpu_id == 0:

            # 配置基本的日志设置
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='5k_data_manual_weight_1.log',  # 指定日志文件名
            filemode='a'  # 'w'模式会覆盖已存在的文件，'a'模式会追加到文件末尾

        )

        # 创建一个logger
        self.logger = logging.getLogger("my_project")
        self.logger.info(description)
        self.model = DDP(model, device_ids=[gpu_id],find_unused_parameters=True)
        self.train_loss_buffer={"first_loss":[[],[]],           
                                "second_loss":[[],[]],
                                "third_loss":[[],[]],     
                                "fourth_loss":[[],[]],
                                "fifth_loss":[[],[]]   
                 }                  
        self.task_num=5
        self.T=0.7 #terperature for softmax operation
        self.batch_weight = torch.Tensor([1,1,1,1,1]).to(self.gpu_id)     
        self.loss_type=loss_type


    def _run_epoch(self, epoch):
        #b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} |  Steps: {len(self.train_data)}")#Batchsize: {b_sz} |
        self.train_data.sampler.set_epoch(epoch)
        self.model.train()
        if self.gpu_id == 0:
             self.logger.info(f"一共是{len(self.train_data)}step")
        acc_step=0
        for step,inputs in enumerate(self.train_data):
            loss_dict=self.model(input_ids=inputs["input_ids"].to(self.gpu_id),
                    bbox=inputs["bbox"].to(self.gpu_id),
                    position_ids=inputs["position_ids"].to(self.gpu_id),
                    reverse_position_ids=inputs["reverse_position_ids"].to(self.gpu_id),
                    attention_mask=inputs["attention_mask"].to(self.gpu_id),
                    type_ids=inputs["type_ids"].to(self.gpu_id),
                    newline_id=inputs["newline_id"].to(self.gpu_id),
                    sentence_start=inputs["sentence_start"].to(self.gpu_id),

                    order_id_text=inputs["order_id_text"].to(self.gpu_id),
                    order_mask_text=inputs["order_mask_text"].to(self.gpu_id),
                    order_start_text=inputs["order_start_text"].to(self.gpu_id),
                    order_id_caption=inputs["order_id_caption"].to(self.gpu_id),
                    order_mask_caption=inputs["order_mask_caption"].to(self.gpu_id),
                    order_start_caption=inputs["order_start_caption"].to(self.gpu_id),

                    newline_start=inputs["newline_start"].to(self.gpu_id),
                    page_position_id=inputs["page_position_id"].to(self.gpu_id),
                    citation_start=inputs["citation_start"].to(self.gpu_id),
                    citation_id=inputs["citation_id"].to(self.gpu_id)
                    )
            loss=0
            if "first_loss" in loss_dict:
                    loss+=1*loss_dict["first_loss"]
            if "second_loss" in loss_dict:
                    loss+=1*loss_dict["second_loss"]
            if "third_loss" in loss_dict:
                    loss+=10*loss_dict["third_loss"]
            if "fourth_loss" in loss_dict:
                    loss+=5*loss_dict["fourth_loss"]
            if "fifth_loss" in loss_dict:
                    loss+=1*loss_dict["fifth_loss"]
            
            log_loss=loss.item()
            loss=loss/self.accumulation_steps
            loss.backward()

            # if self.gpu_id == 0:
            #         self.logger.info(str(log_loss))


            if (step+1)%self.accumulation_steps:
                self.optimizer.step()
                self.optimizer.zero_grad()

                #self.scheduler.step()

            # Log metrics only on the main process
            if step%20==0:
                # with torch.no_grad():
                #     # Aggregate loss across all GPUs
                #     loss_tensor = torch.tensor(log_loss).to(self.gpu_id)
                #     torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
                #     avg_loss = loss_tensor.item() / self.world_size
                # if self.gpu_id == 0:
                #     wandb.log({"loss": avg_loss})
                self.logger.info(f"rank{self.gpu_id} log_loss:{log_loss}")
                # self.logger.info(f"rank{self.gpu_id} Step:{step} Avg_loss: {avg_loss}")
            
            if self.gpu_id == 0 and (step+1)%5000 == 0:
                self._save_checkpoint(epoch,step=step)



    def _save_checkpoint(self, epoch,step=0):
        ckp = self.model.module.state_dict()
        PATH = f"/data4/students/zhangguangyin/pdf_parse/save_result/5kmaual_weight-{epoch+12}-{step}.pt"
        torch.save(ckp, PATH)
        self.logger.info(f"Epoch {epoch} | Training checkpoint saved at {PATH}")


    def train(self, max_epochs: int):
        self.initialize_wandb()
        if self.gpu_id == 0:
             self.eval()
        self.logger.info(f"rank{self.gpu_id}进入测试评估")
        #self.eval(0)
        # else:
        #      self.logger.info(f"rank{self.gpu_id}进入等待")
        dist.barrier()
        if self.gpu_id == 0:
             self.logger.info(f"rank{self.gpu_id}开始训练")
        else:
             self.logger.info(f"rank{self.gpu_id}开始训练")
        self.optimizer.zero_grad()
        for epoch in range(max_epochs):
            self._run_epoch(epoch)
            if self.gpu_id == 0 and (epoch+1) % self.save_every == 0:
                self._save_checkpoint(epoch)
                torch.cuda.empty_cache()
                self.eval()
            dist.barrier()
            # Finish the wandb run
        
        if self.gpu_id == 0:
            self._save_checkpoint(epoch)

        if self.gpu_id == 0:
            wandb.finish()
    
    def eval(self,):
        self.model.eval()
        with torch.no_grad():
              eval_model(self.model.module,self.logger,self.gpu_id)


    def initialize_wandb(self,):
        # 检测模型参数的梯度信息
        wandb.login(key="cac4443c5caf2e64c71b6f41c73f5212f8c545e4")
        self.run=wandb.init(project="my_project")#,config=newconfig







def load_train_objs(batchsize,learning_rate,loss_type):

    tokenizer=XLMRobertaTokenizerFast(vocab_file="/data4/students/zhangguangyin/pdf_parse/customized_changed_vocab_model2_single_loss/sentencepiece.bpe.model")
    tokenizer.add_tokens(["Figure__","Table__","Equation__"])
    print(f"Custom vocabulary size: {len(tokenizer)}")
    print(tokenizer.encode("Figure__"))
    print(tokenizer.encode("Table__"))
    print(tokenizer.encode("Equation__"))
    model=LayoutLMv3ForTokenClassification_custom.from_pretrained("/data4/students/zhangguangyin/pdf_parse/customized_changed_vocab_model2")
    #model=LayoutLMv3ForTokenClassification_custom.from_pretrained("/root/private_data/results/checkpoint-9296")
    
    model.resize_token_embeddings(len(tokenizer))
    model.load_state_dict(torch.load("/data4/students/zhangguangyin/pdf_parse/save_result/5kmaual_weight-11-4999.pt",weights_only=True),strict=False)

    if os.path.exists("/data4/students/zhangguangyin/pdf_parse/train_dataset.pkl"):
        with open('/data4/students/zhangguangyin/pdf_parse/train_dataset.pkl', 'rb') as f:
            train_set = pickle.load(f)
    else:

        traindir=[]
        for doc in os.listdir("/data4/students/zhangguangyin/pdf_parse/data/zhangguangyin/pdf_parse/annotated_images/draw_image_pure"):
             traindir.append("/data4/students/zhangguangyin/pdf_parse/data/zhangguangyin/pdf_parse/annotated_images/draw_image_pure/"+doc)

        train_set = preparedataset(traindir,tokenizer)  # load your dataset
    
    train_sampler = DistributedSampler(train_set, shuffle=True)
    train_loader=DataLoader(
        train_set,
        batch_size=batchsize,
        collate_fn=collate_fn,
        pin_memory=True,
        shuffle=False,
        sampler=train_sampler
    )
    

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    #scheduler=MultiStepLR(optimizer, milestones=[4000,7000], gamma=0.9)
    return train_loader, model, optimizer,None


def ddp_setup(rank, world_size):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12393"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int, accumulation_steps:int,learning_rate,description,loss_type):
    ddp_setup(rank, world_size)
    train_loader, model, optimizer, scheduler = load_train_objs(batch_size,learning_rate,loss_type)
    trainer = Trainer(model=model, train_data=train_loader, optimizer=optimizer, gpu_id=rank, save_every=save_every,accumulation_steps=accumulation_steps,scheduler=scheduler,description=description,loss_type=loss_type)
    trainer.train(total_epochs)
    # trainer.eval()
    destroy_process_group()












if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--total_epochs', type=int,default=10, help='Total epochs to train the model')
    parser.add_argument('--save_every', type=int,default=1, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=2, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--accumulation_steps', default=3, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--learning_rate', default=9e-6, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--loss_type', default="fifth_loss", type=str, help='single loss type')
    args = parser.parse_args()
    description="train with five task"
    world_size = torch.cuda.device_count()
    print(f"world_size:{world_size}")
    mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size, args.accumulation_steps, args.learning_rate,description,args.loss_type), nprocs=world_size)







