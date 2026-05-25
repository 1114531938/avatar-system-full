import os
import torch
os.environ["WANDB__SERVICE_WAIT"] = "300"
import argparse
from config import get_args_parser
import models
import datasets
from utils.utils import seed_everything, generate_date_time
from utils.loss import ClosedFormSampledDistanceLoss
from utils.prob_eval import compute_csd_sims,compute_matching_prob_sims
from utils.pcme import sample_gaussian_tensors
from torch.utils.data import DataLoader
import tqdm
import time
import json
import random
import glob
import wandb
import numpy as np
from sklearn.metrics import confusion_matrix
# import seaborn as sns
import matplotlib.pyplot as plt
import evaluation
# import visualize
import utils 
from transformers import Wav2Vec2Processor
from datetime import datetime
torch.cuda.empty_cache()

# this code is for only training probDEE
# speaker normalization is not allowed in this code
def train_one_epoch(args, model, optimizer,scheduler, train_dataloader, device, processor, epoch=None,log_wandb = True):
    cumulative_loss = 0
    step = 0
    criterion = model.criterion 
    model.train()
    model.to(device)
    for samples, labels in tqdm.tqdm(train_dataloader):
        '''
        samples : [audio_processed,expression_processed]
        labels : [emotion, intensity, gender, actor_id] ->[ int, int, int, int]
        '''
        audio_samples = processor(samples[0],sampling_rate=16000, return_tensors="pt").input_values[0].to(device)
        expression_samples = samples[1].to(device)
        audio_embedding, expression_embedding = model(audio_samples, expression_samples)
        # while setting up the matched matrix, various methods can be used
        # PCME++ used mixup augmentation and label smoothing but for now
        # we use a vanila match matrix 
        batch_size = len(audio_samples)
        matched = torch.eye(batch_size).to(device)
        # 
        loss, loss_dict = criterion(expression_embedding, audio_embedding,matched)
        if log_wandb:
            wandb.log({k: (v.item() if hasattr(v, 'item') else v) for k, v in loss_dict.items()})
            wandb.log({"learning rate": optimizer.param_groups[0]['lr']})
        if step % 50 == 0:
            print(f"epoch {epoch}, step {step} loss : {loss}")
            
        step+=1
        optimizer.zero_grad()
        loss.backward()
        optimizer.step() 
        
        audio_samples = audio_samples.detach().cpu()
        expression_samples = expression_samples.detach().cpu()
        cumulative_loss += loss.item() # culminative loss is only for CLIP loss
        del samples
        del audio_samples
        del expression_samples
        torch.cuda.empty_cache()
            
    scheduler.step()
    batch_num_per_epoch = len(train_dataloader)
    print(f"Epoch {epoch} loss: {cumulative_loss/batch_num_per_epoch}") # train_dataloader
    
@torch.no_grad()
def val_one_epoch(args, model, val_dataloader , device, processor, epoch=None, log_wandb = True):
    cumulative_loss = 0
    step = 0
    model.eval()
    criterion = model.criterion
    exp_means = []
    audio_means = []
    exp_sigmas = []
    audio_sigmas = []
    emotion_list = []
    intensity_list = []
    gender_list = []
    actor_list = []
    for samples, labels in tqdm.tqdm(val_dataloader):
        '''
        samples : [audio_processed,expression_processed]
        labels : [emotion, intensity, gender, actor_id] ->[ int, int, int, int]
        '''
        audio_samples = processor(samples[0],sampling_rate=16000, return_tensors="pt").input_values[0].to(device)
        expression_samples = samples[1].to(device)
        audio_embedding, expression_embedding = model(audio_samples, expression_samples)

        
        exp_means.append(expression_embedding['mean'].detach().cpu().numpy())
        exp_sigmas.append(expression_embedding['std'].detach().cpu().numpy())
        audio_means.append(audio_embedding['mean'].detach().cpu().numpy())
        audio_sigmas.append(audio_embedding['std'].detach().cpu().numpy())
        emotion, intensity, gender, actor_name = labels
        
        emotion = torch.tensor(emotion).unsqueeze(1) #(BS,1)
        intensity = torch.tensor(intensity).unsqueeze(1) #(BS,1)
        gender = torch.tensor(gender).unsqueeze(1)
        
        emotion_list.append(emotion)
        intensity_list.append(intensity)
        gender_list.append(gender)
        actor_list.append(actor_name) 

        batch_size = len(audio_samples)
        matched = torch.eye(batch_size).to(device)
        
        loss, loss_dict = criterion(expression_embedding, audio_embedding, matched)

        if step % 50 == 0:
            print(f"epoch {epoch}, step {step} loss : {loss}")
            
        step+=1

        audio_samples = audio_samples.detach().cpu()
        expression_samples = expression_samples.detach().cpu()
        cumulative_loss += loss.item() # culminative loss is only for CLIP loss
        del samples
        del audio_samples
        del expression_samples
        torch.cuda.empty_cache()
        
    audio_means = np.concatenate(audio_means, axis=0)
    audio_sigmas = np.concatenate(audio_sigmas, axis=0)
    exp_means = np.concatenate(exp_means, axis=0)
    exp_sigmas = np.concatenate(exp_sigmas, axis=0)
    
    emotion = torch.cat(emotion_list, dim=0) # (validation_size,1)
    intensity = torch.cat(intensity_list, dim=0) # (validation_size,1)
    gender = torch.cat(gender_list, dim = 0) # (validation_size,1)
    DB_emositygen = torch.cat((emotion, intensity, gender), dim=1) # (val_size,3)
    
    print(f'Computing sims...')
    now = datetime.now()
    if args.inference_method == 'sampling':
        if args.add_n_mog >= 1:
            n_gaussians = args.add_n_mog+1 # 1 for the original gaussian
            sampled_exp_features = []
            sampled_audio_features = []
            for i in range(n_gaussians):
                sampled_exp_features.append(sample_gaussian_tensors(# BS, num_samples, dim
                    torch.tensor(exp_means)[:,i,:], torch.tensor(exp_sigmas)[:,i,:], args.num_samples))
                sampled_audio_features.append(sample_gaussian_tensors(
                    torch.tensor(audio_means)[:,i,:], torch.tensor(audio_sigmas)[:,i,:], args.num_samples))
            sampled_exp_features = torch.concat(sampled_exp_features, dim=1) # (Bs, n_gaussians*num_samples, dim)
            sampled_audio_features = torch.concat(sampled_audio_features, dim=1)
        else: # no mixture of gaussian
            sampled_exp_features = sample_gaussian_tensors(torch.tensor(exp_means), torch.tensor(exp_sigmas), args.num_samples)
            sampled_audio_features = sample_gaussian_tensors(torch.tensor(audio_means), torch.tensor(audio_sigmas), args.num_samples)
            
        sims = compute_matching_prob_sims(
            sampled_exp_features, sampled_audio_features, 8 * (args.add_n_mog + 1),
            criterion.negative_scale, criterion.shift).T
        
    elif args.inference_method == 'csd':
        sims = compute_csd_sims(exp_means, audio_means, exp_sigmas, audio_sigmas).T
    print(sims)
    print(f'Computing sims {sims.shape=} takes {datetime.now() - now}')  

    exp_retrieve_accuracy, audio_retrieve_accuracy, matched_audio, matched_expression = evaluation.cross_retrival_accuracy(sims, DB_emositygen, visualize=True)
    # visualize.visualize_retrieval(exp_retrieve_accuracy, audio_retrieve_accuracy, matched_audio, matched_expression, args, epoch=epoch)
    batch_num_per_epoch = len(val_dataloader)
    print(f"Epoch {epoch} loss: {cumulative_loss/batch_num_per_epoch}") # train_dataloader
    if log_wandb:
        wandb.log({"expression accuracy loss": exp_retrieve_accuracy,
                    "audio accuracy loss": audio_retrieve_accuracy})
        wandb.log({"val loss": cumulative_loss/batch_num_per_epoch})
        mean_acc = (audio_retrieve_accuracy + exp_retrieve_accuracy)/2.

    
    
    
def main(args): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training start using {device}...")
    seed_everything(42)
    print("Loading models...")
    DEE = models.ProbDEE(args)
    DEE = DEE.to(device)
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    
    print("Loading dataset...")
    val_dataset = datasets.AudioExpressionDataset(args, dataset = 'MEAD', split = 'val')
    print("length of val dataset: ", len(val_dataset))
    start_time = time.time()
    train_dataset = datasets.AudioExpressionDataset(args, dataset=args.dataset, split = 'train') # debug
    print(f"Dataset loaded in {time.time() - start_time} seconds")
    print("length of train dataset: ", len(train_dataset))
    datum = train_dataset[0]
    print("audio slice shape: ", datum[0][0].shape)
    print("expression parameter slice shape:", datum[0][1].shape)
    # jb - batch size 10 -> changed to args.batch_size
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(DEE.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0.000001)

    print("Training loop...")
    best_val_acc = 0
    for epoch in range(1, args.epochs+1):
        training_time = time.time()
        train_one_epoch(args, DEE, optimizer,scheduler, train_dataloader, device,processor, epoch=epoch, log_wandb = True)
        print('training time for this epoch :', time.time() - training_time)
        
        validation_time = time.time()
        val_one_epoch(args, DEE, val_dataloader, device,processor, epoch=epoch, log_wandb = True)
        print('validation time for this epoch :', time.time() - validation_time)
    
        if epoch % args.val_freq == 0 :
            torch.save(DEE.state_dict(), f"{args.save_dir}/model_{epoch}.pt")
            print(f"Model saved at {args.save_dir}/model_{epoch}.pt")
            print("Traininig DONE!")

# get retrival accuraccy on the fly

def save_arguments_to_file(args, filename='arguments.json'):
    save_path = args.save_dir + '/' + filename
    with open(save_path, 'w') as file:
        json.dump(vars(args), file)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser('train', parents=[get_args_parser()])
    args = parser.parse_args()
    time_stamp = generate_date_time()
    args.save_dir = args.save_dir+ '_' +time_stamp
    
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    wandb.init(project = args.project_name,
               name = args.wandb_name,
               config = args)
    
    print(args.save_dir)
    save_arguments_to_file(args)
    main(args)

    print("DONE!")
