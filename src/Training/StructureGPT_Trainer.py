import functools
import pickle
from distutils.version import LooseVersion
import torch.distributed as dist
import torch.optim.optimizer
from torch.cuda import nccl
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchmetrics.classification import MulticlassAccuracy
from StructureGPT_Transformer import *
import os
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



def ddp_setup():
    init_process_group(backend='nccl')

class Trainer:
    def __init__(self, model: torch.nn.Module, train_data: DataLoader, val_data: DataLoader,
                 optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, warmup_steps: int, save_every: int,
                 snapshot_path: str, metric, loss_fn, name=None, pos_emb=False, profiler=None, fully=False, loadsnapshothiperparams = False) -> None:

        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.global_rank = int(os.environ['RANK'])
        self.model = model.to(self.local_rank)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_steps = warmup_steps
        self.save_every = save_every
        self.fully = fully
        if not fully:
            self.model = DDP(self.model, device_ids=[self.local_rank])
        self.metric = metric.to(self.local_rank)
        self.loss_fn = loss_fn.to(self.local_rank)
        self.pos_emb = pos_emb
        self.name = name
        self.profiler = profiler
        self.epochs_run = 0
        self.train_losses = 0
        self.eval_losses = 0
        self.train_accuracies = 0
        self.eval_accuracies = 0
        self.losses = {}
        self.accs = {}
        self.format = format
        self.loadsnapshothiperparams = loadsnapshothiperparams
        if os.path.exists(os.path.join(snapshot_path, self.name+'_snapshot.pt')):
            print('Loading snapshot')
            self._load_snapshot(snapshot_path,self.loadsnapshothiperparams)

    def _load_snapshot(self, snapshot_path=os.path.join(''), loadsnapshothiperparams = False):
        snapshot = torch.load(os.path.join(snapshot_path, self.name + '_snapshot.pt'))
        self.model.module.load_state_dict(snapshot['MODEL_STATE'])

        if loadsnapshothiperparams:
            self.epochs_run = snapshot['EPOCHS_RUN']
            self.optimizer.load_state_dict(snapshot['OPTIMIZER_STATE'])
            self.scheduler.load_state_dict(snapshot['SCHEDULER_STATE'])

        else:
            self.optimizer = self.optimizer
            self.scheduler = self.scheduler

        freeze_except_last(self.model)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=1e-4)


        print(f'Resuming training from snapshot at Epoch {self.epochs_run}')

    def _run_batch(self, src, tgt):
        try:
            self.model.train()

            tgt_input = tgt[:-1, :]
            #print(f"src shape: {src.shape}, tgt shape: {tgt.shape}")

            # Creates masks for the forward pass:
            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.local_rank)

            # Computes the forward pass:
            logits = self.model(src, tgt_input, src_mask, tgt_mask,
                                src_padding_mask, tgt_padding_mask, src_padding_mask, self.pos_emb)

            # Sets grads to zero:
            self.optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            # Computes loss:
            self.train_loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            # Computes metric value:
            self.train_metric = self.metric(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            # Computes backward pass:
            self.train_loss.backward()
            # Applies optimizer:
            self.optimizer.step()
            if self.profiler:
            # Applies profiler:
                self.profiler.step()
            else:
                pass
            self.train_losses += self.train_loss.item()
            self.train_accuracies += self.train_metric.item()
            torch.cuda.empty_cache()

        except Exception as e:
                    print(f"Error durante _run_batch: {e}")
                    print(f"Dimensiones de src: {src.shape}, tgt: {tgt.shape}")
                    raise e  # Vuelve a lanzar la excepción para detener la ejecución o manejarla de otra manera

    def _run_epoch(self, epoch):
        for src, tgt in self.train_data:
            src = src.to(self.local_rank)
            tgt = tgt.to(self.local_rank)
            self._run_batch(src, tgt)
        if self.scheduler:
            # It checks (warmup_steps + 5) because we want to reduce the lr just in 5 epochs!
            if self.warmup_steps is not None and self.warmup_steps + 5 > epoch >= self.warmup_steps:
                self.scheduler.step()
            elif self.warmup_steps is None:
                self.scheduler.step()
        print(f'[GPU{self.global_rank}] | Epoch: {epoch} | Train loss: {self.train_losses / len(self.train_data)} | Train accuracy: {self.train_accuracies / len(self.train_data)}')
        self.losses[f'Epoch {epoch} in GPU{self.global_rank} train loss'] = self.train_losses / len(self.train_data)
        self.accs[f'Epoch {epoch} in GPU{self.global_rank} train acc'] = self.train_accuracies / len(self.train_data)
        self.train_losses = 0
        self.train_accuracies = 0

    def _eval_batch(self, src, tgt):
        self.model.eval()

        tgt_input = tgt[:-1, :]

        # Creates masks for evaluation:
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, self.local_rank)

        # Computes forward pass for evaluation:
        logits = self.model(src, tgt_input, src_mask.detach(), tgt_mask.detach(),
                       src_padding_mask.detach(), tgt_padding_mask.detach(), src_padding_mask.detach(),
                       self.pos_emb).detach()

        tgt_out = tgt[1:, :]
        self.val_loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        self.val_metric = self.metric(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        self.eval_losses += self.val_loss.item()
        self.eval_accuracies += self.val_metric.item()
        torch.cuda.empty_cache()

    def _eval_epoch(self, epoch):
        for src, tgt in self.val_data:
            src = src.to(self.local_rank)
            tgt = tgt.to(self.local_rank)
            self._eval_batch(src, tgt)
        print(f'[GPU{self.global_rank}] | Epoch: {epoch} | Val. loss: {self.eval_losses / len(self.val_data)} | Val. accuracy: {self.eval_accuracies / len(self.val_data)}')
        self.losses[f'Epoch {epoch} in GPU{self.global_rank} val loss'] = self.eval_losses / len(self.val_data)
        self.accs[f'Epoch {epoch} in GPU{self.global_rank} val acc'] = self.eval_accuracies / len(self.val_data)
        self.eval_losses = 0
        self.eval_accuracies = 0

    def _save_snapshot(self, epoch, backup_path=os.path.join('')):
        if self.fully:
            snapshot = {}
            snapshot['EPOCHS_RUN'] = epoch
            snapshot['OPTIMIZER_STATE'] = self.optimizer.state_dict()
            if self.scheduler is not None:  # Verifica si el scheduler está inicializado
                snapshot['SCHEDULER_STATE'] = self.scheduler.state_dict()
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
                cpu_state = model.state_dict()
            if self.global_rank == 0:
                snapshot['MODEL_STATE'] = model.state_dict()
            torch.save(snapshot, os.path.join(backup_path, self.name + '_snapshot.pt'))
            print(f'Epoch {epoch} | Training snapshot saved at snapshot.pt')

        else:
            snapshot = {}
            snapshot['MODEL_STATE'] = self.model.module.state_dict()
            snapshot['EPOCHS_RUN'] = epoch
            snapshot['OPTIMIZER_STATE'] = self.optimizer.state_dict()
            if self.scheduler is not None:  # Verifica si el scheduler está inicializado
                snapshot['SCHEDULER_STATE'] = self.scheduler.state_dict()
            torch.save(snapshot, os.path.join(backup_path, self.name + '_snapshot.pt'))
            print(f'Epoch {epoch} | Training snapshot saved at snapshot.pt')

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            self._run_epoch(epoch)
            self._eval_epoch(epoch)
            if self.local_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        with open(os.path.join(''), 'wb') as f:
            pickle.dump(self.losses, f)
        with open(os.path.join(''), 'wb') as g:
            pickle.dump(self.accs, g)


def prepare_dataloader(path_to_data, batch_size: int, format='.cif'):
    data = ProteinDataset(path_to_data, format)
    return DataLoader(data,
                      batch_size=batch_size,
                      pin_memory=True,
                      shuffle=False,
                      sampler=DistributedSampler(data),
                      collate_fn=generate_batch)


def main(total_epochs: int, save_every: int, path_to_train_data, path_to_val_data, batch_size, model, optimizer,
         scheduler, warmup_steps, metric, loss_fn, format='.cif', name=None,
         snapshot_path=os.path.join(''),
         pos_emb=False, profiler=None, fully=False, loadsnapshothiperparams = False):
    ddp_setup()
    train_iter = prepare_dataloader(path_to_train_data, batch_size, format)
    val_iter = prepare_dataloader(path_to_val_data, batch_size, format)
    #check_batches(train_iter)
    if fully:
        bfSixteen = MixedPrecision(
            param_dtype=torch.float32,
            # Gradient communication precision.
            reduce_dtype=torch.float32,
            # Buffer precision.
            buffer_dtype=torch.float32,
        )
        my_auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                TransformerDecoderLayer,  # < ---- Your Transformer layer class
            },  # min_num_params=int(54463001/torch.cuda.device_count())
        )
        sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP  # for Zero2 and FULL_SHARD for Zero3
        bf16_ready = (
                torch.version.cuda
                and torch.cuda.is_bf16_supported()
                and LooseVersion(torch.version.cuda) >= "11.0"
                and dist.is_nccl_available()
                and nccl.version() >= (2, 10)
        )

        if bf16_ready:
            mp_policy = bfSixteen
        else:
            mp_policy = None  # defaults to fp32

        model = FSDP(model,
                     auto_wrap_policy=my_auto_wrap_policy,
                     mixed_precision=mp_policy,
                     sharding_strategy=sharding_strategy,
                     device_id=torch.cuda.current_device(),
                     backward_prefetch=BackwardPrefetch.BACKWARD_PRE)
    trainer = Trainer(model, train_iter, val_iter, optimizer, scheduler, warmup_steps, save_every,
                      snapshot_path, metric, loss_fn, name, pos_emb, profiler, fully, loadsnapshothiperparams)
    trainer.train(total_epochs)
    dist.barrier()
    destroy_process_group()
