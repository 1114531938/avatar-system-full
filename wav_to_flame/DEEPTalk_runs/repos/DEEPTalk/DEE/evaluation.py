import os
import torch
import argparse
from config import get_args_parser
import models
import datasets
from utils.loss import CLIP_loss, RECO_loss, emotion_guided_loss_gt, CLIP_loss_with_expression_guide
from utils.utils import compare_checkpoint_model
from torch.utils.data import DataLoader
import tqdm
import time
import json
import random
import glob
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from transformers import Wav2Vec2Processor

def validation(args, model=None, dataloader=None, visualize=False, device=None):
    # to evaluate during training
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    if model is not None :
        DEE = model
    # only evaluate with validation dataset
    else :
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Validation start using {device}...")

        #Choose number of epoch for checkpoint
        if args.last_ckpt :
            checkpoints = glob.glob(f'{args.save_dir}/*.pt')
            checkpoints = [checkpoint for checkpoint in checkpoints if "best.pt" not in checkpoint]
            if len(checkpoints[0].split('_')) == 2: # checkpoints like model_1.pt
                sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                epoch = os.path.basename(sorted_checkpoints[-1]).split('_')[-1].split('.')[0]
            elif len(checkpoints[0].split('_')) == 3: # checkpoints like model_1_0.011.pt
                sorted_checkpoints = sorted(checkpoints, key=lambda x: int(x.split('_')[-2]))
                epoch = os.path.basename(sorted_checkpoints[-1]).split('_')[-2]
        elif args.best_ckpt :
            epoch = 'best'
 
        else :
            epoch = args.num_ckpt
        
        print("Loading models...")
        if args.use_emo2vec:
            DEE = models.DEE_v2(args)
        else:
            DEE = models.DEE(args)
        DEE = DEE.to(device)
       
        print("Loading validation dataset...")

        dataset = datasets.AudioExpressionDataset(args,dataset=args.dataset ,split = args.split)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        print(f'Evaluate with {args.split} dataset')

        print("Validation starts...")
        # Load checkpoint file
        checkpoint_path = glob.glob(f'{args.save_dir}/model_{epoch}*.pt')[0]
        # checkpoint_path = f"{args.save_dir}/model_{epoch}.pt"
        print(f"Load checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path) 

        DEE.load_state_dict(checkpoint)
        if compare_checkpoint_model(checkpoint, DEE):
            print("Checkpoint and model are the same")
        else:
            print("Checkpoint and model are not the same")
            exit()
        

    DEE.eval()
    expression_embedding_list = []
    audio_embedding_list = []
    emotion_list = []
    intensity_list = []
    gender_list = []
    actor_list = []
    culmulative_loss = 0
    for samples, labels in tqdm.tqdm(dataloader) :
        '''
        samples : [audio_processed,expression_processed]
        labels : [emotion -> int, intensity -> int, gender -> int, actor_id -> int]
        '''
        expression_samples= samples[1].to(device)
        # audio_samples = samples[0].to(device)
        audio_samples = processor(samples[0],sampling_rate=16000, return_tensors="pt").input_values[0].to(device)
        emotion, intensity, gender, actor_id = labels
        
        emotion = torch.tensor(emotion).unsqueeze(1).to(device) #(BS,1)
        intensity = torch.tensor(intensity).unsqueeze(1).to(device) #(BS,1)
        gender = torch.tensor(gender).unsqueeze(1).to(device)
        actor_id = torch.tensor(actor_id).unsqueeze(1).to(device)
        emotion_list.append(emotion)
        intensity_list.append(intensity)
        gender_list.append(gender)
        actor_list.append(actor_id)   

        with torch.no_grad():
            audio_embedding, expression_embedding = DEE(audio_samples, expression_samples)
            
        loss = CLIP_loss(audio_embedding, expression_embedding, DEE.logit_scale, device)
        culmulative_loss += loss.item()
        
        expression_embedding_list.append(expression_embedding)
        audio_embedding_list.append(audio_embedding)
        
    batch_num_per_epoch = len(dataloader) # total batches in valdataset
    val_loss = culmulative_loss / batch_num_per_epoch
    
    expression_embedding = torch.cat(expression_embedding_list, dim=0) # (validation_size,512)

    audio_embedding = torch.cat(audio_embedding_list, dim=0) # (validation_size,512)

    emotion = torch.cat(emotion_list, dim=0) # (validation_size,1)
    intensity = torch.cat(intensity_list, dim=0) # (validation_size,1)
    gender = torch.cat(gender_list, dim = 0) # (validation_size,1)
    actor = torch.cat(actor_list, dim = 0) # (validation_size,1)
    
    DB_emositygenid = torch.cat((emotion, intensity, gender, actor), dim=1) # (val_size,4)
    DB_expression = expression_embedding # (val_size, 512)
    DB_audio = audio_embedding # (val_size,512)
    
    # compute similarity btw audio, expression
    sim_matrix = DB_audio @ DB_expression.T # (val_size,val_size)
    print('sim_matrix shape : ', sim_matrix.shape)
    
    if visualize : # if we want to visualize
        # self-modal retrival (exp from exp, audio from audio)
        # expression_accuracy, audio_accuracy, matched_audio, matched_expression = retrival_accuracy(audio_audio_matrix, DB_emositygenid, visualize=visualize)
        # cross-modal retrival (exp from audio, audio from exp) 
        expression_accuracy, audio_accuracy, matched_audio, matched_expression = cross_retrival_accuracy(sim_matrix, DB_emositygenid, visualize=visualize)
        return DB_expression, DB_audio, DB_emositygenid, expression_accuracy, audio_accuracy, matched_audio, matched_expression, val_loss
        # visualize_validation(matched_audio, matched_expression, args)
    else:
        expression_accuracy, audio_accuracy = cross_retrival_accuracy(sim_matrix, DB_emositygenid, visualize=visualize)
        return expression_accuracy, audio_accuracy, val_loss

# (10-23) added top_k
def cross_retrival_accuracy(sim_matrix, labels, with_self=True, top_k = 1, visualize=True):
    """
    sim_matrix : (N,N)
    labels : (N,3)
    """
    # retrieve topk exp with audio
    if not with_self:
        sim_matrix = sim_matrix - torch.onestorch.eye(sim_matrix.shape[0])
    exp_retrieve_num = 0
    print(f'Sim matrix shape : {sim_matrix.shape}')
    if not torch.is_tensor(labels):
        labels = torch.tensor(labels)
    if not torch.is_tensor(sim_matrix):
        sim_matrix = torch.tensor(sim_matrix)
        
    sim_matrix = sim_matrix.detach().cpu()
    print(f'sim matrix device : {sim_matrix.device}')
    _, retrieved_exp_indices = torch.topk(sim_matrix, k=top_k, dim=1) # given audio retrieve topk exp
    matched_audio = []
    for audio_index, retrieved_exp_index in enumerate(retrieved_exp_indices):
        retrieved_exp_labels = labels[retrieved_exp_index] # (top_k,2)
        
        exp_emotions = retrieved_exp_labels[:,0] # (top_k)
        exp_intensitys = retrieved_exp_labels[:,1] # (top_k)
        
        predicted_label = None
        gt_label = labels[audio_index][0].cpu()
        
        if gt_label in exp_emotions:
            exp_retrieve_num += 1
            predicted_label = gt_label # if GT is in topk, use GT
        else:   
            predicted_label = exp_emotions[0].cpu() # if GT is not in topk, use top1
        # [GT, retrieved]
        if visualize :
            matched_audio.append([int(gt_label), int(predicted_label)])

            
    # retrieve topk audio with exp    
    audio_retrieve_num = 0
    _, retrieved_audio_indices = torch.topk(sim_matrix, k=top_k, dim=0)
    # dim : (N,1) -> (1,N)
    retrieved_audio_indices = retrieved_audio_indices.T
    # for visualize
    matched_expression = []
    for exp_index, retieved_audio_index in enumerate(retrieved_audio_indices):
        retrieved_audio_labels = labels[retieved_audio_index] # (top_k,2)
        audio_emotions = retrieved_audio_labels[:,0]# (top_k)
        audio_intensitys = retrieved_audio_labels[:,1]# (top_k)
        
        predicted_label = None
        gt_label = labels[exp_index][0].cpu()
        
        if gt_label in audio_emotions:
            audio_retrieve_num += 1
            predicted_label = gt_label
        else:
            predicted_label = audio_emotions[0].cpu()
        # [GT, retrieved]
        if visualize :
            matched_expression.append([int(gt_label), int(predicted_label)])
    
    exp_retrieve_accuracy = exp_retrieve_num / sim_matrix.shape[0]
    audio_retrieve_accuracy = audio_retrieve_num / sim_matrix.shape[0]

    if visualize :
        matched_audio = np.array(matched_audio)
        matched_expression = np.array(matched_expression)
        return exp_retrieve_accuracy, audio_retrieve_accuracy, matched_audio, matched_expression
    else :
        return exp_retrieve_accuracy, audio_retrieve_accuracy
    
def classification_accuracy(predictions,labels):
    '''
    get Speaker Identity Detection or Gender classyfication accuracy
    predictions : (BS, numclass)
    labels : (BS,)
    '''
    predictions = torch.argmax(predictions,dim=1)
    correct = torch.sum(predictions == labels)
    accuracy = correct / len(labels)
    return accuracy


def self_retrival_accuracy(self_self_matrix, labels, top_k = 1, visualize=True):
    """
    sim_matrix : (N,N)
    labels : (N,3)
    """
    # slef retrival should always discard self
    self_self_matrix = self_self_matrix - torch.onestorch.eye(self_self_matrix.shape[0])
    retrieve_num = 0
    self_self_matrix = self_self_matrix.detach().cpu()
    
    _, retrieved_indices = torch.topk(self_self_matrix, k=top_k, dim=1) # given audio retrieve topk exp
    matched = []
    for origin_index, retrieved_index in enumerate(retrieved_indices):
        retrieved_labels = labels[retrieved_index] # (top_k,2)
        retrieved_emotions = retrieved_labels[:,0] # (top_k)
        retrieved_intensitys = retrieved_labels[:,1] # (top_k)
        
        predicted_label = None
        gt_label = labels[origin_index][0].cpu()
        
        if gt_label in retrieved_emotions:
            retrieve_num += 1
            predicted_label = gt_label # if GT is in topk, use GT
        else:   
            predicted_label = retrieved_emotions[0].cpu() # if GT is not in topk, use top1
        # [GT, retrieved]
        if visualize :
            matched.append([int(gt_label), int(predicted_label)])
    
    retrieve_accuracy = retrieve_num / self_self_matrix.shape[0]

    if visualize :
        matched = np.array(matched)
        return retrieve_accuracy, matched
    else :
        return retrieve_accuracy
    
    