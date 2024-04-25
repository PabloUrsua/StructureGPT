# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 08:31:54 2022

@author: Nicanor
"""

from StructureGPT_Utils import *
from torch.nn import TransformerEncoderLayer, TransformerEncoder, TransformerDecoderLayer, TransformerDecoder


#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



# Implementation of the transformer described in the paper
# titled "Attention is all you need".
class StructureGPT_TransformerEncoderDecoder(nn.Module):
    
    def __init__(self, num_blocks: int, num_heads: int, emb_sz: int, src_vocab_sz: int,
                 tgt_vocab_sz: int, dim_feedforward: int = 512, dropout: float = 0.1, pos_emb=False):
        
        super(StructureGPT_TransformerEncoderDecoder, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_sz, nhead=num_heads,
                                                dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_blocks)
        decoder_layer = TransformerDecoderLayer(d_model=emb_sz, nhead=num_heads,
                                                dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_blocks)
        self.generator = nn.Linear(emb_sz, tgt_vocab_sz)

        # Here max_length should be modified if sequence is longer than 5000.
        if pos_emb:
            self.src_pos_emb = CoordsPosEmbedding(src_vocab_sz, emb_sz, dropout)
            self.tgt_pos_emb = PositionWiseEmbedding(tgt_vocab_sz, emb_sz, dropout)
        else:
            self.src_tok_emb = nn.Linear(src_vocab_sz, emb_sz)  # This line has been modified!!
            self.tgt_tok_emb = TokenEmbedding(tgt_vocab_sz, emb_sz)
            self.positional_encoding = PositionalEncoding(emb_sz, dropout=dropout)
        self.initParams()
        
    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor, tgt_mask: Tensor, src_padding_mask: Tensor,
                tgt_padding_mask: Tensor, memory_key_padding_mask: Tensor, pos_emb):
        if pos_emb:
            src_emb = self.src_pos_emb(src)
            tgt_emb = self.tgt_pos_emb(tgt)
        else:
            src_emb = self.positional_encoding(self.src_tok_emb(src))
            tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None,
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)
    
    def encode(self, src: Tensor, src_mask: Tensor, pos_emb):
        if pos_emb:
            return self.transformer_encoder(self.src_pos_emb(src), src_mask)
        else:
            return self.transformer_encoder(self.positional_encoding(
                self.src_tok_emb(src)), src_mask)
    
    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor, pos_emb):
        if pos_emb:
            return self.transformer_decoder(self.tgt_pos_emb(tgt), memory, tgt_mask)
        else:
            return self.transformer_decoder(self.positional_encoding(
                self.tgt_tok_emb(tgt)), memory, tgt_mask)

    def initParams(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)