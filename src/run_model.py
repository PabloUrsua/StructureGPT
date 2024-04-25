import argparse
import datetime
import sys
import json
import torch.optim as optim
import pandas as pd
from StructureGPT_Trainer import *
import os
from StructureGPT_Trainer import *
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



def parse_args():
    # Prepare argument parser:
    CLI = argparse.ArgumentParser()
    CLI.add_argument("model_type", nargs=1, type=str, default='transformer', help='Type of model to run.')
    CLI.add_argument("--name", nargs="?", type=str, default=str(datetime.datetime.now()), help='name for the model')
    CLI.add_argument("--dataset", nargs="?", type=str, default=None, help='Dataset to be used for training.')
    CLI.add_argument("--embsize", nargs="?", type=int, default=None, help='embeding size')
    CLI.add_argument("--nheads", nargs="?", type=int, default=None, help='number of heads')
    CLI.add_argument("--nblocks", nargs="?", type=int, default=None, help='number of heads')
    CLI.add_argument("--optimizer", nargs="?", type=str, default=None, help='optimizer')
    CLI.add_argument("--metric", nargs="?", type=str, default=None, help='metric')
    CLI.add_argument("--loss", nargs="?", type=str, default=None,
                     help="Cost function. Options: 'cross_entropy' for CrossEntropyLoss; 'focal' for FocalLoss")
    CLI.add_argument("--lr", nargs="?", type=float, default=None,
                     help="Value of the optimizer's learning rate")
    CLI.add_argument("--wd", nargs="?", type=float, default=None,
                     help="Value of the optimizer's weight decay.")
    CLI.add_argument("--dr", nargs="?", type=float, default=None,
                     help="Value of the drop rate.")
    CLI.add_argument("--posemb", nargs="?", type=bool, default=False,
                     help="Using positional embeding, in otherwise using positional encoding,")
    CLI.add_argument("--scheduler", nargs="?", type=bool, default=False,
                     help="Using scheduler")
    CLI.add_argument("--warmup", nargs="?", type=int, default=None,
                     help="Using warmup steps")
    CLI.add_argument("--ffdim", nargs="?", type=int, default=None,
                     help="Using positional embeding, in otherwise using positional encoding,")
    CLI.add_argument("--paralel", nargs="?", type=bool, default=False,
                     help="option for train the model in more than one GPU")
    CLI.add_argument("--fully", nargs="?", type=bool, default=False,
                     help="option for train the model in more than one GPU using Fully Sharded data paralel")
    CLI.add_argument("--loadhiperparams", nargs="?", type=bool, default=False,
                     help="option for load hiperparams from snapshot")

    return CLI.parse_args()

def full_process(model_class, name, config_file_name, dataset, optimizer, metric, learning_rate, weight_decay, drop_rate,
                 loss_function, num_blocks, emb_size, num_heads, ff_dim, pos_emb, paralel, scheduler, warmup_steps, fully,loadsnapshothiperparams):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    info_data = {}
    with open(os.path.join(CONFIG_PATH, config_file_name)) as config_file:
        config_data = json.load(config_file)
        train_params = config_data['train_params']
        num_epochs = train_params['num_epochs']
        batch_size = train_params['batch_size']
        save_every = train_params['save_every']
        using_tensorboard = train_params['using_tensorboard']
        model_params = config_data['model_params'][model_type]
        format = config_data['dataset_params'][dataset]["format"]
        src_vocab_size = config_data['dataset_params'][dataset]["src_vocab_size"]
        tgt_vocab_size = config_data['dataset_params'][dataset]["tgt_vocab_size"]

        if model_class == StructureGPT_TransformerEncoderDecoder:
            # Prepare info_data for storing metadata about the parameters of the model to be saved:
            info_data['model'] = 'OmegaFold_TransformerEncoderDecoder'
            info_data['ff_dim'] = ff_dim
            info_data['pos_emb'] = pos_emb
            info_data['dataset_name'] = dataset
            info_data['batch_size'] = batch_size
            info_data['growth_rate'] = model_params['growth_rate']
            info_data['memory_efficient'] = model_params['memory_efficient']
            if drop_rate is None:
                drop_rate = model_params['drop_rate']
        if emb_size is None:
            emb_size = model_params['emb_size']
            info_data['emb_size'] = emb_size
        if num_heads is None:
            num_heads = model_params['num_heads']
        if num_blocks is None:
            num_blocks = model_params['num_blocks']
        if ff_dim is None:
            ff_dim = model_params['ff_hid_dim']
        if optimizer is None:
            optimizer = model_params['optimizer']
        if metric is None:
            metric = model_params['metric']
        if loss_function is None:
            loss_function = model_params['loss_function']
        if learning_rate is None:
            learning_rate = model_params['learning_rate']
        if weight_decay is None:
            weight_decay = model_params['weight_decay']
        if warmup_steps is None:
            warmup_steps = model_params['warmup_steps']

    if paralel:
        # instance model
        if model_class == StructureGPT_TransformerEncoderDecoder:
            model = model_class(num_blocks, num_heads, emb_size, src_vocab_size,
                                tgt_vocab_size, ff_dim, dropout=drop_rate, pos_emb=pos_emb)
        # instance the optimizer algorithm
        if optimizer == 'sgd':
            # We pass only the non-frozen Parameters to the optimizer:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            if scheduler:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=0.6 * learning_rate,
                                                                 last_epoch=-1, verbose=True)
            else:
                scheduler = None
        elif optimizer == 'adam':
            # DEBUG: Testing much smaller values for learning rate when using Adam optimizer
            # optimizer = optim.Adam(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9,
                                    weight_decay=weight_decay)
            if scheduler:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=0.6 * learning_rate,
                                                                 last_epoch=-1, verbose=True)
            else:
                scheduler = None
        else:
            raise Exception('Compatibility with the given optimizer has not been implemented yet')
        if metric == 'acc':
            metric = MulticlassAccuracy(num_classes=25, ignore_index=RES_DIC['PAD_IDX']).to(device)
        else:
            raise Exception('Not implemented yet')

        # instance the loss function
        if loss_function == 'cross_entropy':
            loss_fn = nn.CrossEntropyLoss(ignore_index=RES_DIC['PAD_IDX']).to(device)
        else:
            exit()
            # for other loss functions

        # train the model
        print('Voy a entrenar\n')
        main(num_epochs, save_every, TRAIN_FOLDER, VAL_FOLDER, batch_size, model, optimizer, scheduler,
             warmup_steps, metric, loss_fn, format, name=name, pos_emb=pos_emb, profiler=None, fully=fully, loadsnapshothiperparams = loadsnapshothiperparams )

    else:
        # instance model
        if model_class == StructureGPT_TransformerEncoderDecoder:
            model = model_class(num_blocks, num_heads, emb_size, src_vocab_size,
                                tgt_vocab_size, ff_dim, dropout=drop_rate, pos_emb=pos_emb)
        #send model to device
        model = model.to(device)
        #instance the optimizer algorithm
        if optimizer == 'sgd':
            # We pass only the non frozen Parameters to the optimizer:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
            if scheduler:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=learning_rate * 0.6,
                                                                 last_epoch=-1, verbose=True)
            else:
                scheduler = None
        elif optimizer == 'adam':
            # DEBUG: Testing much smaller values for learning rate when using Adam optimizer
            # optimizer = optim.Adam(trainable_parameters(), lr=learning_rate, weight_decay=weight_decay)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9,
                                    weight_decay=weight_decay)
            if scheduler:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=learning_rate * 0.6,
                                                                last_epoch=-1, verbose=True)
            else:
                scheduler = None
        else:
            raise Exception('Compatibility with the given optimizer has not been implemented yet')
        if metric == 'acc':
            metric = MulticlassAccuracy(num_classes=25, ignore_index=RES_DIC['PAD_IDX']).to(device)
        else:
            raise Exception('Not implemented yet')


        #instance the loss function
        if loss_function == 'cross_entropy':
            loss_fn = nn.CrossEntropyLoss(ignore_index=RES_DIC['PAD_IDX']).to(device)
        else:
            exit()
            #for other loss functions

        #train the model
        print('Voy a entrenar\n')
        model, metrics = train_model(model, num_epochs, TRAIN_FOLDER, VAL_FOLDER, batch_size, optimizer, metric,
                                     device, loss_fn, pos_emb, using_tensorboard=using_tensorboard, name=name)
        #save metrics and model
        df = pd.DataFrame.from_dict(metrics)
        df.to_csv(os.path.join(BACKUP_PATH, 'metrics.csv'))
        torch.save(model.state_dict(), os.path.join(BACKUP_PATH, name))
    return

