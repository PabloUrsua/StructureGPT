# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 12:52:40 2022

@author: Nicanor
"""

import numpy as np
import os
import torch
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


# A list containing all the possible atom classes.
ATOM_LIST = ['N', 'CA', 'C', 'O', 'CB', 'CG', 'SD', 'CE', 'CG1', 'CG2', 'CD1', 'CD2', 'OG', 'CD', 'OE1', 'OE2', 'NE1', 
             'CE2', 'CE3', 'CZ2', 'CZ3', 'CH2', 'NE2', 'ND1', 'CE1', 'NZ', 'OD1', 'OD2', 'NE', 'CZ', 'NH1', 'NH2',
             'OG1', 'OG2', 'OH', 'ND2', 'SG', 'OXT']
ATOM_LIST.sort()

# A dictionary that mapes atom classes to an index.
ATOM_DIC = {atom: i * 3 for i, atom in enumerate(ATOM_LIST)}

# A list containing all possible amino acid types.
RES_LIST = ['ASP', 'ASN', 'GLU', 'GLN', 'GLY', 'ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PRO', 'SER', 'THR', 'TYR', 'TRP',
            'ARG', 'LYS', 'CYS', 'HIS', 'PHE', 'BOS', 'EOS', 'BOC', 'EOC', 'PAD_IDX']
RES_LIST.sort()

# A dictionary that maps amino acid types to an index.
RES_DIC = {res: i for i, res in enumerate(RES_LIST)}

# A dictionary that maps index to amino acid types.
REV_RES_DIC = {RES_DIC[key]: key for key in RES_DIC}

# A dictionary that maps three-letter code amino acids to one-letter code amino acids.
TRANSLATION_DIC = {'GLY': 'G', 'ALA': 'A', 'VAL': 'V', 'LEU': 'L', 'ILE': 'I', 'PHE': 'F', 'TYR': 'Y', 'TRP': 'W',
                   'SER': 'S', 'THR': 'T', 'CYS': 'C', 'MET': 'M', 'PRO': 'P', 'ASP': 'D', 'GLU': 'E', 'ASN': 'N',
                   'GLN': 'Q', 'LYS': 'K', 'ARG': 'R', 'HIS': 'H', 'BOS': '<bos>', 'BOC': '<boc>', 'EOS': '<eos>',
                   'EOC': '<eoc>', 'PAD_IDX': '<pad>'}

# Initial sequence token for atomic coordinates.
init_seq_tok = np.ones((len(ATOM_DIC) * 3, ))

# Initial chain token for atomic coordinates.
init_chain_tok = np.array([1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.])

# Final sequence token for atomic coordinates.
end_seq_tok = - np.ones((len(ATOM_DIC) * 3, ))

# Final chain token for atomic coordinates.
end_chain_tok = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., -1., -1., -1., 0., 0., 0., 0., 0., 0.])

# Padding token
pad_tok = np.array([1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


def coordinates_to_tensor(pdb_id, path=DATA_FOLDER, format='.afcif'):
    
    # Converts atom coordinates into coordinates tensor of shape 
    # (num_res + 2(k + 1), 114), where k is the number of protein chains.

    if format == '.pdb':
        # Extracts all lines that start with the "ATOM" keyword.
        lines = {('line %s' %i):line.split() for i, line in enumerate(
            [line for line in open(os.path.join(path, pdb_id), encoding='utf-8') if line.startswith('ATOM')])}
    
        chains = []
        chain = []
        chain_type = lines['line 0'][4]
    
        # For each chain, it creates a list of lines that belong
        # to that chain and appends that list to "chains" list.
        for line in lines:
            if lines[line][4] == chain_type:
                chain.append(lines[line])
            else:
                chains.append(chain)
                chain = []
                chain_type = lines[line][4]
                chain.append(lines[line])
        chains.append(chain)
    
        # Initializes protein coordinates with initial-sequence token.
        prot_coords = [init_seq_tok]
    
        # Creates a list of atom coordinates of all residues of each chain
        # in the protein.
        for i, chain in enumerate(chains):
        
            chain_coords = [init_chain_tok]
            res_type = chain[0][3]
            res_num = chain[0][5]
            res_coord = np.zeros((len(ATOM_DIC) * 3, ))
        
            for line in chain:
            
                if line[2] in ATOM_DIC and line[3] == res_type and line[5] == res_num:
                
                    atom_idx = ATOM_DIC[line[2]]
                    res_coord[atom_idx:atom_idx + 3] = line[6:9]
            
                elif line[2] in ATOM_DIC and (line[3] != res_type or line[5] != res_num):
                
                    chain_coords.append(res_coord)
                    res_type = line[3]
                    res_num = line[5]
                    res_coord = np.zeros((len(ATOM_DIC) * 3, ))
                    atom_idx = ATOM_DIC[line[2]]
                    res_coord[atom_idx:atom_idx +3] = line[6:9]
        
            # Appends coordinates of the last residue.
            chain_coords.append(res_coord)
            chain_coords.append(end_chain_tok)
        
            # If it is the last chain it appends the end-sequence token
            # and adds the chain to prot_coords.
            if i == len(chains) - 1:
                chain_coords.append(end_seq_tok)
                prot_coords += chain_coords
            else:
                prot_coords += chain_coords


        # Creates a tensor from prot_coords.
        final_coords= torch.tensor(np.array(prot_coords), dtype=torch.float)
        #final_coords = final_coords.type(torch.long)

        return final_coords.reshape((final_coords.shape[0], 1, final_coords.shape[-1]))

    else:
        # Extracts all lines that start with the "ATOM" keyword.
        lines = {('line %s' % i): line.split() for i, line in enumerate(
            [line for line in open(os.path.join(path, pdb_id), encoding='utf-8') if line.startswith('ATOM')
             and len(line.split()[0]) == 4])}

        chains = []
        chain = []
        chain_type = lines['line 0'][6]

        # For each chain, it creates a list of lines that belong
        # to that chain and appends that list to "chains" list.
        for line in lines:
            if lines[line][6] == chain_type:
                chain.append(lines[line])
            else:
                chains.append(chain)
                chain = []
                chain_type = lines[line][6]
                chain.append(lines[line])
        chains.append(chain)

        # Initializes protein coordinates with initial-sequence token.
        prot_coords = [init_seq_tok]

        # Creates a list of atom coordinates of all residues of each chain
        # in the protein.
        for i, chain in enumerate(chains):

            chain_coords = [init_chain_tok]
            res_type = chain[0][5]
            res_num = chain[0][8]
            res_coord = np.zeros((len(ATOM_DIC) * 3,))

            for line in chain:

                if line[3] in ATOM_DIC and line[5] == res_type and line[8] == res_num:

                    atom_idx = ATOM_DIC[line[3]]
                    res_coord[atom_idx:atom_idx + 3] = line[10:13]

                elif line[3] in ATOM_DIC and (line[5] != res_type or line[8] != res_num):

                    chain_coords.append(res_coord)
                    res_type = line[5]
                    res_num = line[8]
                    res_coord = np.zeros((len(ATOM_DIC) * 3,))
                    atom_idx = ATOM_DIC[line[3]]
                    res_coord[atom_idx:atom_idx + 3] = line[10:13]

            # Appends coordinates of the last residue.
            chain_coords.append(res_coord)
            chain_coords.append(end_chain_tok)

            # If it is the last chain it appends the end-sequence token
            # and adds the chain to prot_coords.
            if i == len(chains) - 1:
                chain_coords.append(end_seq_tok)
                prot_coords += chain_coords
            else:
                prot_coords += chain_coords

        # Creates a tensor from prot_coords.
        final_coords = torch.tensor(np.array(prot_coords), dtype=torch.float)
        # final_coords = final_coords.type(torch.long)

        return final_coords.reshape((final_coords.shape[0], 1, final_coords.shape[-1]))


def sequence_to_tensor(pdb_id, path=DATA_FOLDER, format='.afcif'):
    # Converts amino acid sequence into one-hot tensor of shape
    # (num_res + 2(k + 1), 24), where k is the number of protein chains.

    residues = ['BOS', 'BOC']
    indices = []
    chain = ''
    if format == '.pdb':
        # Extracts all lines that start with the "SEQRES" keyword.
        lines = {('line %s' % i): line.split() for i, line in enumerate(
            [line for line in open(os.path.join(path, pdb_id), encoding='utf-8') if line.startswith('SEQRES')])}

        # Initialize chain_type with the chain type of the first chain.
        chain_type = lines['line 0'][2]

        # Extracts amino acid sequence.
        for line in lines:
            if lines[line][2] == chain_type:
                residues += lines[line][4:]
            else:
                residues.append('EOC')
                residues.append('BOC')
                residues += lines[line][4:]
                chain_type = lines[line][2]
        residues.append('EOC')
        residues.append('EOS')

    elif format == '.afcif':
        # Extracts all lines that start with the "SEQRES" keyword.
        lines = {('line %s' % i): line.split() for i, line in enumerate(
            [line for line in open(os.path.join(path, pdb_id), encoding='utf-8') if
             (line.split()[0] == 'A' and len(line.split()) == 7)])}

        # Initialize chain_type with the chain type of the first chain.
        chain_type = lines['line 0'][0]

        # Extracts amino acid sequence.
        for i in range(len(lines.keys())):
            if lines[f'line {i}'][0] == chain_type:
                residues.append(lines[f'line {i}'][1])
            else:
                residues.append('EOC')
                residues.append('BOC')
                residues.append(lines[f'line {i}'][1])
                chain_type = lines[f'line {i}'][0]
        residues.append('EOC')
        residues.append('EOS')

    elif format == '.mmcif':
        lines = {('line %s' % i): line.split() for i, line in enumerate(
            [line for line in open(os.path.join(path, pdb_id), encoding='utf-8') if
             line.startswith('ATOM') and len(line.split()[0]) == 4 and len(line.split()) >= 21])}

        current_res_type = lines['line 0'][5]
        current_chain = lines['line 0'][6]
        current_res_num = lines['line 0'][8]
        residues.append(current_res_type)

        for i in range(len(lines.keys())):
            if lines[f'line {i}'][8] != current_res_num:
                residues.append(lines[f'line {i}'][5])
                current_res_num = lines[f'line {i}'][8]
            elif lines[f'line {i}'][6] != current_chain:
                current_chain = lines[f'line {i}'][6]
                residues.append('EOC')
                residues.append('BOC')
        residues.append('EOC')
        residues.append('EOS')


    # Switches three-letter-amino acid code to indexes.
    for res in residues:
        if res in RES_DIC:
            indices.append(RES_DIC[res])

    # Applies one-hot codification for 24 indexes
    # (20 aa + boc + bos + eoc + eos).
    seq_tensor = torch.tensor(indices, dtype=torch.long)

    return seq_tensor


def pad_coordinates(coords_list, padding_length: int, pad_tok=pad_tok):

    final_list = []
    pad_tok = torch.tensor(pad_tok, dtype=torch.float).reshape(1, 1, pad_tok.shape[0])

    for coords in coords_list:
        if coords.shape[0] < padding_length:
            coords = torch.cat([coords, torch.cat([pad_tok for i in range(padding_length -
                                                                          coords.shape[0])])])
        final_list.append(coords)

    return torch.cat(final_list, dim=1)


# Creates a subsequent mask to attend subsequent words.
def generate_square_subsequent_mask(sz: int, device):

    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    
    return mask


# Generates a mask that highlights coordinates padding.
def generate_coordinates_padding_mask(src_coords: torch.Tensor, device):
    
    a, b = src_coords.shape[0], src_coords.shape[1]
    p_tok = torch.tensor(pad_tok, device=device).float().reshape((1, pad_tok.shape[-1]))
    
    padding_mask = torch.zeros((a, b), device=device)
    
    for i in range(a):
        for j in range(b):
            padding_mask[i][j] = (src_coords[i][j] == p_tok).all()
    
    return padding_mask.transpose(0, 1)



# Creates src_mask, tgt_mask, src_padding_mask and tgt_padding_mask.
def create_mask(src: torch.Tensor, tgt: torch.Tensor, device):
    
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]
    
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device) # This mask allows the decoder to attend subsequent positions.
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=device).type(torch.bool)  # This mask allows the encoder to attend all positions.
    
    src_padding_mask = generate_coordinates_padding_mask(src, device)
    tgt_padding_mask = (tgt == RES_DIC['PAD_IDX']).transpose(0, 1)
    
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

