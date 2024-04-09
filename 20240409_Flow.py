PROJECT_PATH = "/workspace/NegativeScreeing/Amesdata/2403_SMARTS"

import sys
sys.path.append(PROJECT_PATH)
import os
import datetime
import argparse
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from sklearn.model_selection import train_test_split
from rdkit import Chem

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src import utils
from src import data_handler as dh
import src.models as ms
from src.plot import *

now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
file = os.path.basename(__file__).split(".")[0]
DIR_NAME = PROJECT_PATH + "/results/" + file + "_" + now
if not os.path.exists(DIR_NAME):
    os.makedirs(DIR_NAME)
LOGGER = utils.init_logger(file,DIR_NAME,now,level_console="debug")
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# argument
parser = argparse.ArgumentParser(description="CLI temprate")
parser.add_argument("--note",type=str,help="short note for this running")
parser.add_argument("--train",type=bool,default=True)
parser.add_argument("--seed",type=str,default=222)
parser.add_argument("--num_epoch",type=int,default=1000000)
parser.add_argument("--batch_size",type=int,default=64)
parser.add_argument("--maxlr",type=float,default=0.0003) 
parser.add_argument("--warmup",type=int,default=10000)
parser.add_argument("--save_dir",type=str,default=DIR_NAME)
parser.add_argument("--patience",type=int,default=10)
parser.add_argument("--beta",type=int,default=1)
parser.add_argument("--train_size",type=int,default=0.9)

args = parser.parse_args()
utils.fix_seed(seed=args.seed,fix_gpu=False)

def randomize_smiles(smiles):
    sm = [None]*len(smiles)
    for i,s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        ans = list(range(m.GetNumAtoms()))
        np.random.shuffle(ans)
        nm = Chem.RenumberAtoms(m,ans)
        sm[i] = Chem.MolToSmiles(nm,canonical=False)
    return sm

#data preparation
def prepare_data():
    df = pd.read_csv("{}/data/0320_smarts_train_data.csv".format(PROJECT_PATH))
    df = df[df["SMARTS"].str.len() < 250]
    df = df[df["SMILES"].str.len() < 148]
    x_train_SMARTS = df["SMARTS"]
    x_train_SMILES = df["SMILES"]
    combined_x_train = pd.concat([x_train_SMARTS, x_train_SMILES], axis=1)

    y_binary_train = df["Y"].values

    df2 = pd.read_csv("{}/data/0320_smarts_test_data.csv".format(PROJECT_PATH))
    df2 = df2[df2["SMARTS"].str.len() < 250]
    df2 = df2[df2["SMILES"].str.len() < 150]
    x_test_SMARTS = df2["SMARTS"]
    x_test_SMILES = df2["SMILES"]
    combined_x_test = pd.concat([x_test_SMARTS, x_test_SMILES], axis=1)
    y_binary_test = df2["Y"].values

    combined_x_train, combined_x_valid, y_binary_train, y_binary_valid = train_test_split(combined_x_train, y_binary_train, train_size=args.train_size, shuffle=True)


    return combined_x_train, combined_x_valid, combined_x_test, y_binary_train, y_binary_valid

class Encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        nx = config.n_embd
        self.wte = nn.Embedding(config.vocab_size,nx)
        self.wpe = ms.PositionalEncoding(nx,config.dropout,max_len=config.n_positions)
        self.drop = nn.Dropout(config.dropout)
        
        self.h = nn.ModuleList([ms.TransformerBlock(config.n_positions,config,scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        self.ln_mem1 = nn.LayerNorm(nx)
        self.ln_mem2 = nn.LayerNorm(nx)
        self.ln_mem3 = nn.LayerNorm(nx)
        self.fc_latent = nn.Linear(3*nx,config.n_embd)

    def create_enc_attention_mask(self,input_ids):
        l, b = input_ids.size()
        pad_array = (input_ids == 0).transpose(0,1).unsqueeze(1).unsqueeze(2)
        return torch.where(pad_array == True, float("-inf"), 0.0) # [B,1,1,L]

    def memory_pool(self,memory):
        mx = torch.max(memory,dim=0)[0]
        ave = torch.mean(memory,dim=0)
        first = memory[0]
        return torch.cat([self.ln_mem1(mx),self.ln_mem2(ave),self.ln_mem3(first)],dim=1)

    def forward(self,x,past=None):
        """
        x: Tensor, [L,B]
        """ 
        input_shape = x.size()
        x = x.view(-1,input_shape[-1])
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        input_embeds = self.wte(x)
        hidden_states = self.wpe(input_embeds)

        attention_mask = self.create_enc_attention_mask(x)
        output_shape = input_shape + (hidden_states.size(-1),)

        for i, (block,layer_past) in enumerate(zip(self.h,past)):
            outputs = block(hidden_states,layer_past=layer_past,attention_mask=attention_mask)
            hidden_states, present = outputs[:2]
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape) # [L,B,D]

        latent_space = self.memory_pool(hidden_states) # [B,3*D]
        latent_space = self.fc_latent(latent_space)
        return  torch.tanh(latent_space) # [B,D]  


class Decoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        nx = config.n_embd
        self.wte = nn.Embedding(config.vocab_size,nx)
        self.wpe = ms.PositionalEncoding(nx,config.dropout,max_len=config.n_positions)
        self.input_proj = nn.Linear(nx,nx,bias=False)
        self.h = nn.ModuleList([ms.TransformerBlock(config.n_positions,config,scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        self.output_fc = nn.Linear(nx,config.vocab_size)

    def create_dec_attention_mask(self,input_ids):
        l, b = input_ids.size()
        pad_array = (input_ids == 0).transpose(0,1).unsqueeze(1).unsqueeze(2) # [B,1,1,L]

        seq_array = torch.triu(torch.full((l,l),True,device=DEVICE),diagonal=1)
        seq_array = seq_array.unsqueeze(0).unsqueeze(1) # [1,1,L,L]
        res = torch.logical_or(pad_array,seq_array)
        return torch.where(res == True, float("-inf"), 0.0) # [B,1,L,L]

    def forward(self,x,latent,layer_past=None):
        # x: [L,B]
        # latent: [B,D]
        input_shape = x.size()
        if layer_past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        
        attention_mask = self.create_dec_attention_mask(x)
        input_embeds = self.wte(x)
        hidden_states = self.wpe(input_embeds)
        hidden_states = hidden_states + latent.unsqueeze(1).transpose(0,1)

        presents = ()
        for i,(block,layer_past) in enumerate(zip(self.h,past)):
            outputs = block(hidden_states,layer_past=layer_past,attention_mask=attention_mask)
            hidden_states, present = outputs[:2]
            presents = presents + (present,)
        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.output_fc(hidden_states)
        return hidden_states # [L,B,V]


class TransformerModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self,src,tgt,past=None):
        latent_space = self.encoder(src)
        outputs = self.decoder(tgt,latent_space,layer_past=past)
        
        return outputs, latent_space

#model preparation
def prepare_model():
    conf = ms.Config(n_embd=256,dropout=0.1)
    conf2 = ms.Config2()
    model = Transformer2Model(conf).to(DEVICE)
    model2 = TransformerModel(conf2).to(DEVICE)
    model2.load_state_dict(torch.load("{}/models2/230415_transformer_93model.pth".format(PROJECT_PATH),torch.device(DEVICE)),strict=False)
        
    criterion = nn.CrossEntropyLoss(ignore_index=0,reduction="mean")
    criterion2 = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(),lr=args.maxlr*np.sqrt(args.warmup))

    lr_schedule = ms.warmup_schedule(args.warmup)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_schedule)
    es = ms.EarlyStopping(patience=args.patience)
    return model, model2,  criterion, criterion2, optimizer, scheduler, es


class Transformer2Model(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
        self.mlp = MLP(config)

    def forward(self,src,latent_smiles, tgt,past=None):
        latent_space = self.encoder(src)
        latent_cat = torch.cat((latent_space,latent_smiles), 1)
        outputs = self.decoder(tgt,latent_space,layer_past=past)
        outputs2 = self.mlp(latent_cat)
        return outputs, outputs2


class MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config  = config
        nx = config.n_embd
        self.classifier = nn.Sequential(
            nn.Linear(nx * 2 ,500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1),
            nn.Sigmoid(),
        )

    def forward(self, x, ):
        x = self.classifier(x)

        return x

def accuracy(y_pred, y_true, mode="accuracy", threshold=0.5):
    
    binary_tensor = torch.where(y_pred < threshold, torch.tensor(0), torch.tensor(1))
    true_positives = torch.sum((binary_tensor == 1) & (y_true == 1)).item()
    false_positives = torch.sum((binary_tensor == 1) & (y_true == 0)).item()
    false_negatives = torch.sum((binary_tensor == 0) & (y_true == 1)).item()
    
    if mode == "accuracy":
        correct_predictions = (binary_tensor == y_true).float()
        accuracy = correct_predictions.mean().item()
        return accuracy, true_positives, false_positives, false_negatives

    if mode == "precision":
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        return precision

    if mode == "recall":
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        return recall

def train_step(model, model2, src, src2, y, criterion, criterion2, optimizer, scheduler):
    
    src = src.to(DEVICE)
    tgt = src.to(DEVICE)
    src2 = src2.to(DEVICE)
    tgt2 = src2.to(DEVICE)
    dec_i = tgt[:-1,:]
    target = tgt[1:,:]
    dec_i_2 = tgt2[:-1,:]
    optimizer.zero_grad()

    _, latent_smiles = model2(src2, dec_i_2)

    array_data = np.array(y)
    tensor_data = torch.tensor(array_data, dtype=torch.float32)
    tensor_data_resized = tensor_data.view(64, 1)
    tensor_data_resized = tensor_data_resized.to(DEVICE)
    out, out2 = model(src, latent_smiles, dec_i)
    l = criterion(out.transpose(-2,-1),target)
    
    tensor_data_resized = tensor_data_resized.to(DEVICE)
    loss_mlp = criterion2(out2,tensor_data_resized)

    with torch.no_grad():
        accurate, true_positives, false_positives, false_negatives = accuracy(out2,tensor_data_resized,mode="accuracy")
        precision = accuracy(out2,tensor_data_resized, mode="precision")
        recall = accuracy(out2,tensor_data_resized, mode="recall")
    
    los = l + args.beta * loss_mlp
    los.backward()
    optimizer.step()
  
    scheduler.step()
    return los.item(), l.item(), loss_mlp.item(), accurate, precision, recall, true_positives, false_positives, false_negatives

def valid_step(model, model2,  src, src2, y, criterion, criterion2):
    src = src.to(DEVICE)
    tgt = src.to(DEVICE)
    src2 = src2.to(DEVICE)
    tgt2 = src2.to(DEVICE)
    dec_i = tgt[:-1,:]
    target = tgt[1:,:]
    dec_i_2 = tgt2[:-1,:]
    
    _, latent_smiles = model2(src2, dec_i_2)

    array_data = np.array(y)
    tensor_data = torch.tensor(array_data, dtype=torch.float32)
    tensor_data_resized = tensor_data.view(64, 1)
    tensor_data_resized = tensor_data_resized.to(DEVICE)
    out, out2 = model(src, latent_smiles, dec_i)
    l = criterion(out.transpose(-2,-1),target)
    
    tensor_data_resized = tensor_data_resized.to(DEVICE)
    loss_mlp = criterion2(out2,tensor_data_resized)

    los = l + args.beta * loss_mlp
    return los.item(), l.item(), loss_mlp.item()

def train(model,model2,train_x,valid_x, binary_train_y, binary_valid_y, criterion, criterion2, optimizer,  scheduler,es):
    train_loader = dh.prep_loader(train_x, binary_train_y,buckets=(20,200,10),batch_size=args.batch_size,
                                    shuffle=True,drop_last=True,device=DEVICE)
    model.train()
    loss = []
    valid_loss = []
    num_iter = 0
    end = False
    
    l_t, l_tt, l_tm, l_v, l_tv, l_mv, acc, pred, reca, t_p, f_p, f_n, num_step = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    with tqdm(total = args.num_epoch) as pbar:
        while num_iter < args.num_epoch:
            for (src_A, src_I), tgt in train_loader:
                l, l_tr, l_m, ac, pre, rec, tp, fp, fn = train_step(model, model2, src_A, src_I, tgt, criterion, criterion2, optimizer, scheduler)
                
                loss.append(l)
                l_t += l
                l_tt += l_tr
                l_tm += l_m

                acc += ac
                pred += pre
                reca += rec 
                t_p += tp
                f_p += fp
                f_n += fn
                num_step += 1

                num_iter += 1
                pbar.update(1)
                
                if num_iter % 1000 == 0:
                    eval_loader = dh.prep_loader(valid_x,binary_valid_y, buckets=(20,200,10),batch_size=args.batch_size,
                                            shuffle=True, drop_last=True, device=DEVICE)
                    model.eval()
                    
                    l_v = 0

                    with torch.no_grad():
                        for (src_A, src_I), tgt in eval_loader:
                                l_v_s, l_tv_s, l_mv_s = valid_step(model, model2, src_A, src_I, tgt, criterion, criterion2)
                                l_v += l_v_s
                                l_tv += l_tv_s
                                l_mv += l_mv_s
                        valid_loss.append(l_v)
                    LOGGER.info(f"Epoch: {num_iter}, train_loss: {l_t:.4f}, trans_loss: {l_tt:.4f}, mlp_loss: {l_tm:.4f},\
        accuracy: {acc/num_step:.3f}, precision: {pred/num_step:.3f}, recall: {reca/num_step:.3f}, \
        TP: {t_p}, FP: {f_p}, FN: {f_n}, \
        valid_loss: {l_v:.4f}, valid_trans_loss: {l_tv:.4f}, valid_mlp_loss: {l_mv:.4f}")
                    end = es.step(l_v)
                    model.train()
                    
                    
                    l_t, l_tt, l_tm, l_v, l_tv, l_mv, acc, pred, reca, t_p, f_p, f_n, num_step = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                if num_iter % 10000 == 0:
                    torch.save(model.state_dict(),args.save_dir+f"/model_trans_{num_iter}.pth")
                
                if num_iter > args.num_epoch:
                    return model, model2, loss, valid_loss
            train_loader = dh.prep_loader(train_x, binary_train_y,buckets=(20,200,10),batch_size=args.batch_size,
                                        shuffle=True,drop_last=True,device=DEVICE)
    return model, model2, loss, valid_loss

def evaluate(model,x,y,maxlen=250):
    eval_loader = dh.prep_loader(x,y,buckets=(20,200,10),batch_size=args.batch_size,
                                 shuffle=True,drop_last=True,device=DEVICE,charset="smarts")
    
    model.eval()
    id2sm = {w:v for v,w in dh.vocab_dict(charset="smarts").items()}
    row = []
    with torch.no_grad():
        for (src, tgt), index in tqdm(eval_loader):
            src = src.to(DEVICE)
            latent = model.encoder(src)
            token_ids = np.zeros((maxlen,src.size(1)))
            token_ids[0:,] = 1
            token_ids = torch.tensor(token_ids,device=DEVICE,dtype=torch.long)
            
            for i in range(1,maxlen):
                token_ids_seq = token_ids[:i,:]              
                out = model.decoder(token_ids_seq,latent)
                _, out_id = out.max(dim=2)
                new_id = out_id[-1,:]
                is_end_token = token_ids[i-1,:] == 2
                is_pad_token = token_ids[i-1,:] == 0
                judge = torch.logical_or(is_end_token,is_pad_token)
                if judge.sum().item() == judge.numel():
                    token_ids = token_ids[:i,:]
                    break
                new_id[judge] = 0
                token_ids[i,:] = new_id
            pred = token_ids[1:,:]
            
            for s,t,v in zip(src.T,tgt.T,pred.T):
                x = [id2sm[j.item()] for j in s]
                y = [id2sm[j.item()] for j in t]
                p = [id2sm[j.item()] for j in v]
                x_str = "".join(x[1:]).split(id2sm[2])[0]
                y_str = "".join(y[1:]).split(id2sm[2])[0]
                p_str = "".join(p).split(id2sm[2])[0]
                judge = True if y_str == p_str else False
                row.append([x_str,y_str,p_str,judge])
        
        pred_df = pd.DataFrame(row,columns=["input","answer","predict","judge"])
        accuracy = len(pred_df.query("judge == True")) / len(pred_df)

    return pred_df, accuracy


if __name__ == "__main__":
    if args.train:
        start = time.time()
        combined_x_train, combined_x_valid, combined_x_test, y_binary_train, y_binary_valid = prepare_data()

        model, model2,  criterion, criterion2, optimizer, scheduler, es = prepare_model()
        LOGGER.info("START")
        model, model2,  loss, valid_loss = train(model,model2,combined_x_train, combined_x_valid, y_binary_train, y_binary_valid, criterion, criterion2, optimizer, scheduler,es)
        
        torch.save(model.state_dict(),args.save_dir+"/model_final.pth")
        plot_loss(args.num_epoch,loss,valid_loss,dir_name=args.save_dir)
        LOGGER.info(f"accuracy: {accuracy:.4f}")

        utils.to_logger(LOGGER,name="argument",obj=args)
        utils.to_logger(LOGGER,name="loss",obj=criterion)
        utils.to_logger(LOGGER,name="optimizer",obj=optimizer,skip_keys={"state","param_groups"})
        utils.to_logger(LOGGER,name="scheduler",obj=scheduler)
        LOGGER.info("elapsed time: {:.2f} min".format((time.time()-start)/60))
    
    else:
        # test_loader = prepare_eval_data()
        pass