import random

import torch
from torch.nn.utils import rnn

import io
import json
import logging
import os
import pickle
import re
import shutil
import urllib
import urllib.error
import urllib.request
from typing import Optional
import numpy as np
from urllib.parse import urlparse

def build_one_instance_text_stream(tokenizer, conversations):
    input_ids, target_ids = [], []
    pre_coe = "You are an empathetic conversational agent. Your goal is to understand the user’s emotions and intentions, and respond or comfort them with appropriate language that helps them feel understood and cared for. Avoid rushing into your response; instead, carefully engage in a step-by-step, in-depth analysis before providing an answer.. Now here is the dialogue context:" 
    one_input_id = tokenizer(pre_coe, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += [-100] * len(one_input_id)
    for i, turn in enumerate(conversations['dialogue_history']):
        if i%2 == 0:
            text = 'Human: <Aud> <Vid>' + turn['utterance'] + '\n### Assistant: '
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(one_input_id)
        else:
            text = '<Aud> <Vid>' + turn['utterance'] + '\n###'
            one_input_id = tokenizer(text, add_special_tokens=False).input_ids
            input_ids += one_input_id
            target_ids += [-100] * len(one_input_id) 
    
    coe = f"Firstly, the event scenario of this conversation is {conversations['coe']['event_scenario']} .\
            Secondly, the emotion of the speaker is <emo> {conversations['coe']['speaker_emotion']} </emo>.\
            Thirdly, the emotion cause is {conversations['coe']['emotion_cause']}.\
            Fourthly, the goal to response is {conversations['coe']['goal_to_response']}.\
            Finally, output the empathetic response."
    response = conversations['response'] + '</response>' + '\n###'
    one_input_id = tokenizer(pre_coe+coe+response, add_special_tokens=False).input_ids
    input_ids += one_input_id
    target_ids += one_input_id
    assert len(input_ids) == len(target_ids)
    # print(f"############## length of input ##########: {len(input_ids)}")
    return input_ids, target_ids 


def process_batch_text_stream(tokenizer, batch_of_conversations, max_tgt_len):
    batch_input_ids, batch_target_ids = [], []
    for conversations in batch_of_conversations:
        one_input_ids, one_target_ids = build_one_instance_text_stream(tokenizer, conversations)
        batch_input_ids.append(torch.LongTensor(one_input_ids))
        batch_target_ids.append(torch.LongTensor(one_target_ids))
    input_ids = rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = rnn.pad_sequence(batch_target_ids, batch_first=True, padding_value=-100)
    assert input_ids.size() == target_ids.size()
    input_ids = input_ids[:, :max_tgt_len]
    target_ids = target_ids[:, :max_tgt_len]
    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    assert attention_mask.size() == input_ids.size()
    return input_ids, target_ids, attention_mask.long()


def is_url(url_or_filename):
    parsed = urlparse(url_or_filename)
    return parsed.scheme in ("http", "https")


def l2_loss(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
  """
  Args:
    u: (N, T_I_V_A.txt, D) tensor.
    v: (N, T_I_V_A.txt, D) tensor.
  Returns:
    l1_loss: (N,) tensor of summed L1 loss.
  """
  assert u.shape == v.shape, (u.shape, v.shape)
  return ((u - v) ** 2).sum(dim=-1) ** 0.5

def masked_l2_loss(u: torch.Tensor, v: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Args:
        u: (N, T, D) tensor.
        v: (N, T, D) tensor.
        mask: (N, T) tensor, where 1 indicates valid positions and 0 indicates masked positions.
    Returns:
        masked_l2_loss: (N,) tensor of summed L2 loss for valid positions.
    """
    assert u.shape == v.shape, (u.shape, v.shape)
    assert mask.shape == u.shape[:2], (mask.shape, u.shape[:2])

    # Flatten tensors to process valid positions efficiently
    mask = mask.bool()  # Ensure mask is boolean
    u_valid = u[mask]  # Extract valid positions, shape: (valid_positions, D)
    v_valid = v[mask]  # Extract valid positions, shape: (valid_positions, D)

    # Compute L2 loss for valid positions
    diff = (u_valid - v_valid) ** 2  # Shape: (valid_positions, D)
    masked_l2_loss = diff.sum(dim=-1).sqrt()  # Per valid position, shape: (valid_positions,)

    return masked_l2_loss

def get_modality(path_list):
    _postfix = os.path.splitext(path_list[0])[-1]
    if _postfix == '.jpg':
        return 'image'
    elif _postfix == '.wav':
        return 'audio'
    elif _postfix == '.mp4':
        return 'video'
    else:
        raise NotImplementedError
