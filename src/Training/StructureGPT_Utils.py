# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 09:32:01 2022

@author: Nicanor
"""
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import math
from src.StructureGPT_Preprocessing import *
from torch.utils.data import Dataset
import random
import itertools
import os
from multiprocessing import Pool
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# Positional encoding class.
class PositionalEncoding(nn.Module):
    
    def __init__(self, emb_sz: int, dropout, maxlen=15000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0., emb_sz, 2) * math.log(10000) / emb_sz)
        pos = torch.arange(0., maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_sz))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)
        
    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + 
                            self.pos_embedding[:token_embedding.size(0), :])


class PositionWiseEmbedding(nn.Module):

    def __init__(self, vocab_size, emb_sz, dropout, max_len=15000):
        super(PositionWiseEmbedding, self).__init__()
        self.tok_embedding = nn.Embedding(vocab_size, emb_sz)
        self.pos_embedding = nn.Embedding(max_len, emb_sz)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([emb_sz]))

    def forward(self, inputs):

        # inputs = [inputs len, batch size]
        try:
            batch_size = inputs.shape[1]
            inputs_len = inputs.shape[0]
            #print(inputs.shape)
            pos = torch.arange(0, inputs_len).unsqueeze(0).repeat(batch_size, 1).transpose(0, 1).to(inputs.device)
            scale = self.scale.to(inputs.device)
            embedded = (self.tok_embedding(inputs.long()) * scale) + self.pos_embedding(pos.long())

            # output = [batch size, inputs len, hid dim]
            output = self.dropout(embedded)
            return output
        except Exception as e:
            print(f"Error durante positional embeding: {e}")
            print(f"Dimensiones de entrada: {batch_size}, tgt: {inputs_len.shape}")
            raise e

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_sz):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_sz)
        self.emb_size = emb_sz

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


def mask_coordinates(coords):
    size = coords.shape
    tokens_mask = [torch.ones(size[-1]) for i in range(2)]
    for i in range(2, size[0]-2):
        masked_token = torch.zeros(size[-1])
        for j in range(0, size[-1]//3-1):
            check = random.randint(0, 1)
            if check:
                masked_token[3*j:3*(j+1)] = torch.tensor([1., 1., 1.])
        tokens_mask.append(masked_token)
    tokens_mask = tokens_mask + [torch.ones(size[-1]) for i in range(2)]

    coords_mask = torch.cat(tokens_mask)
    coords_mask = torch.reshape(coords_mask, size)

    return coords*coords_mask

class CoordsPosEmbedding(nn.Module):
    def __init__(self, vocab_sz: int, emb_sz, dropout, max_len=5000):
        super(CoordsPosEmbedding, self).__init__()
        self.coords_emb = nn.Sequential(nn.Linear(vocab_sz, emb_sz), nn.Dropout(dropout))
        self.pos_embedding = nn.Embedding(max_len, emb_sz)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([emb_sz]))

    def forward(self, inputs):
        inputs_len = inputs.shape[0]
        batch_size = inputs.shape[1]

        pos = torch.arange(0, inputs_len).unsqueeze(0).repeat(batch_size, 1).transpose(0, 1).to(inputs.device)
        scale = self.scale.to(inputs.device)

        return (self.coords_emb(inputs) * scale) + self.pos_embedding(pos.long())


class ProteinDataset(Dataset):

    def __init__(self, path_to_data, format='.mmcif'):
        #Look for format is important
        self.path_to_data = path_to_data
        self.look_up_dic = {i:file for (i, file) in enumerate(os.listdir(path_to_data))}
        self.format = format

    def __getitem__(self, idx):
        try:
            coords = coordinates_to_tensor(self.look_up_dic[idx], self.path_to_data, self.format)
            seq = sequence_to_tensor(self.look_up_dic[idx], self.path_to_data, self.format)
        except ValueError:
            return (None, None)
        else:
            return (coords, seq)

    def __len__(self):
        return len(os.listdir(self.path_to_data))


def generate_batch(data_batch, pad_tok=pad_tok):
    coords_batch, seq_batch = [], []
    for coords_item, seq_item in data_batch:
        if coords_item != None:
            coords_batch.append(coords_item)
            seq_batch.append(seq_item)
        else:
            continue
    padding_length = max([coords.shape[0] for coords in coords_batch])

    return pad_coordinates(coords_batch, padding_length, pad_tok), pad_sequence(seq_batch,
                                                                                padding_value=RES_DIC['PAD_IDX'])


def variations_with_repetition(n: list, r: int, num_sols: int):
    # This function randomly creates variations with repetition of length r by repeating the elements in n
    variations = []
    control = 0
    while len(variations) < len(n) ** r:
        if control:
            break
        else:
            variation = []
            for i in range(r):
                variation.append(random.choice(n))
            if variation not in variations:
                variations.append(variation)
                if len(variations) == num_sols:
                    control = 1
                    break
    variations.sort()

    return variations


def variations_with_repetition_ordered(n, r, num_sols):
    # Generate all possible variations with repetition using itertools.product
    all_variations = itertools.product(n, repeat=r)

    # Convert itertools.product object to a list of lists and slice the first num_sols variations
    variations = [list(variation) for variation in itertools.islice(all_variations, num_sols)]

    # Return the requested number of variations
    return variations


def custom_variation_order(n, r, num_sols):
    # Helper function to generate next variation
    def next_variation(current, n, r):
        for i in range(r):
            if current[i] < len(n) - 1:
                current[i] += 1
                return [0]*i + [current[i]] + current[i+1:]
        return None  # No more variations

    # Generate the first variation
    variation = [0] * r  # Represents the indexes of elements in n, not the elements themselves
    count = 0

    while count < num_sols:
        # Convert indexes to actual elements
        yield [n[i] for i in variation]

        count += 1
        if count == num_sols:
            break

        # Generate the next variation based on custom order
        variation = next_variation(variation, n, r)
        if variation is None:
            break  # Stop if there are no more variations to generate


def indices_for_replacement(num_Aa: int, num_sols: int, control_Aa: int):

    # First we create a list with the first class indices list.
    probs_list = [[0 for i in range(num_Aa)]]
    gen_sols = 1
    check = False
    if not control_Aa:
        control_Aa = num_Aa

    # Then we generate the rest of class indices lists.
    for j in range(1, 25):
        if check:
            break
        for k in range(num_Aa):
            class_probs = [0 for i in range(num_Aa)]
            class_probs[k] = j
            gen_sols += 1
            if class_probs not in probs_list:
                probs_list.append(class_probs)
                if gen_sols == num_sols:
                    check = True
                    break
                elif gen_sols % control_Aa == 0:
                    break

    return probs_list


# This function can create all possible combinations of class indices to induce mutation in a protein
# sequence while applying the "translate" function
def indices_for_mutations(positions: list, num_sols: int = 0, num_classes: int = 4):

    # Creates all possible combinations of length len(positions) with the numbers contained in nums_to_combine
    indices_list = list(custom_variation_order([i for i in range(num_classes)], len(positions), num_sols))
    final_indices = []
    for indices in indices_list:
        if indices not in final_indices:
            final_indices.append(indices)

    # If a given number of solutions is provided, final_indices is reset to have the adequate length
    if num_sols:
        final_indices = final_indices[:num_sols]

    return final_indices


# This function takes the list created by indices_for_mutation and creates all possible mutated sequences
def mutation_decode(positions: list, model, src, src_mask, max_len, start_symbol: list, device, pos_emb,
                      num_sols: int = 0, num_classes: int = 4, tgt=None):

    indices_list = indices_for_mutations(positions, num_sols, num_classes)
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask, pos_emb)

    if tgt == None:
        ys = torch.cat([torch.ones(1, 1).fill_(start_symbol[0]).type(torch.long).to(device),
                        torch.ones(1, 1).fill_(start_symbol[1]).type(torch.long).to(device)])
    else:
        ys = tgt.to(device)

    solutions = []
    probabilities = []
    for list in indices_list:
        probability = []
        for i in range(max_len - 1):
            memory = memory.to(device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0), device).type(torch.bool))
            out = model.decode(ys, memory, tgt_mask, pos_emb)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            probs, indices = torch.sort(prob, descending=True)
            if i + 1 in positions:
                next_aa = indices[0][list[positions.index(i + 1)]]
                probability.append((i + 1, torch.softmax(probs, -1)[0][list[positions.index(i + 1)]].item()))
            else:
                next_aa = indices[0][0]
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_aa)], dim=0)
            if next_aa == RES_DIC['EOS']:
                solutions.append(ys)
                ys = torch.cat([torch.ones(1, 1).fill_(start_symbol[0]).type(torch.long).to(device),
                                torch.ones(1, 1).fill_(start_symbol[1]).type(torch.long).to(device)])
                probabilities.append(probability)
                break

    return solutions, probabilities


def replacement_decode(num_sols: int, num_classes: int, model, src, src_mask, max_len, start_symbol: list, device,
                     pos_emb, tgt=None):

    indices_lists = indices_for_replacement(max_len - 1, num_sols, num_classes)
    src = src.to(device)
    src_mask = src_mask.to(device)
    memory = model.encode(src, src_mask, pos_emb)

    if tgt == None:
        ys = torch.cat([torch.ones(1, 1).fill_(start_symbol[0]).type(torch.long).to(device),
                        torch.ones(1, 1).fill_(start_symbol[1]).type(torch.long).to(device)])
    else:
        ys = tgt.to(device)

    solutions = []
    for list in indices_lists:
        for i in range(max_len - 1):
            memory = memory.to(device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0), device).type(torch.bool))
            out = model.decode(ys, memory, tgt_mask, pos_emb)
            out = out.transpose(0, 1)
            prob = model.generator(out[:, -1])
            probs, indices = torch.sort(prob, descending=True)
            next_aa = indices[0][list[i]]
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_aa)], dim=0)
            if next_aa == RES_DIC['EOS']:
                solutions.append(ys)
                ys = torch.cat([torch.ones(1, 1).fill_(start_symbol[0]).type(torch.long).to(device),
                        torch.ones(1, 1).fill_(start_symbol[1]).type(torch.long).to(device)])
                break

    return solutions


def greedy_decode(model, src, src_mask, max_len, start_symbol, device, pos_emb, tgt=None):
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask, pos_emb)
    if tgt == None:
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(device)
    else:
        ys = tgt.to(device)

    for i in range(max_len - 1):
        memory = memory.to(device)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0), device).type(torch.bool))
        out = model.decode(ys, memory, tgt_mask, pos_emb)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_aa = torch.max(prob, dim=1)
        next_aa = next_aa.item()
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_aa)], dim=0)
        if next_aa == RES_DIC['EOS']:
            break

    return ys


def translate(model, device, src, pos_emb, positions: list = [], num_sol: int = 1, num_classes: int = 0,
              mode='literal', tgt_vocab=REV_RES_DIC, translation_dic=TRANSLATION_DIC, tgt=None):
    model.eval()

    translated_sols = []
    if mode == 'replacement':
        res_to_delete = [src[i + 1] for i in positions]
        src = torch.cat([i for i in src if not any([(i == j).all() for j in res_to_delete])], dim=0)
        src = src.unsqueeze(1)
        src_length = src.shape[0]
        src_mask = torch.zeros((src_length, src_length)).type(torch.bool)
        tgt_tokens_list = replacement_decode(num_sol, num_classes, model, src, src_mask,
                                           src_length + 2000, [RES_DIC['BOS'], RES_DIC['BOC']], device, pos_emb)
        for tgt_tokens in tgt_tokens_list:
            three_letter_translation = [tgt_vocab[tok.item()] for tok in tgt_tokens.flatten()]
            translated_sols.append(''.join([translation_dic[idx] for idx in three_letter_translation])
                                   .replace('<bos>', '').replace('<boc>', '').replace('<eos>', '')
                                   .replace('<eoc>', '').replace('<pad>', ''))
        return translated_sols

    elif mode == 'literal':
        src_length = src.shape[0]
        src_mask = torch.zeros((src_length, src_length)).type(torch.bool)
        tgt_tokens = greedy_decode(model, src, src_mask, device=device, max_len=src_length + 2000,
                               start_symbol=RES_DIC['BOS'], pos_emb=pos_emb, tgt=tgt).flatten()
        three_letter_translation = [tgt_vocab[tok.item()] for tok in tgt_tokens]
        return ''.join([translation_dic[idx] for idx in three_letter_translation]).replace('<bos>', '') \
            .replace('<boc>', '-').replace('<eos>', '').replace('<eoc>', '').replace('<pad>', '')

    elif mode == 'mutation':
        src_length = src.shape[0]
        src_mask = torch.zeros((src_length, src_length)).type(torch.bool)
        tgt_tokens_list, probs = mutation_decode(positions, model, src, src_mask, src_length + 4,
                                          [RES_DIC['BOS'], RES_DIC['BOC']], device, pos_emb, num_sol, num_classes, tgt)
        for tgt_tokens in tgt_tokens_list:
            three_letter_translation = [tgt_vocab[tok.item()] for tok in tgt_tokens.flatten()]
            translated_sols.append(''.join([translation_dic[idx] for idx in three_letter_translation])
                                   .replace('<bos>', '').replace('<boc>', '').replace('<eos>', '')
                                   .replace('<eoc>', '').replace('<pad>', ''))
        return translated_sols, probs

def process_file(file_path):
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if '_entity.pdbx_description' in line:
                    description = line.strip().split(None, 1)[1].strip().strip('"')
                    if 'ATP synthase' in description:
                        return file_path  # Return the file path if ATP synthase is found
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
    return None  # Return None if ATP synthase is not found or an error occurs

def process_files(file_paths):
    with Pool(os.cpu_count()) as pool:
        results = pool.map(process_file, file_paths)
    # Filter out None values and return the list of files containing 'ATP synthase'
    return [result for result in results if result is not None]

def count_and_list_atp_synthase_in_cifs(folder_path):
    cif_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.cif')]
    atp_synthase_files = process_files(cif_files)
    count = len(atp_synthase_files)
    return count, atp_synthase_files

def freeze_except_last(model):
    def freeze_except_last(model):
        # Comprobar si el modelo está envuelto en DistributedDataParallel
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module

        # Congelar todas las capas excepto 'generator'
        for name, param in model.named_parameters():
            if 'generator' not in name:
                param.requires_grad = False

        # Asegurarse de que la capa generator está activa para entrenamiento
        for param in model.generator.parameters():
            param.requires_grad = True
