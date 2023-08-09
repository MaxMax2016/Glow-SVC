import os
import time
import logging
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel

from vits_extend.dataloader import create_dataloader_train
from vits_extend.dataloader import create_dataloader_eval
from vits_extend.validation import validate
from vits_extend.writer import MyWriter
from vits_extend.ssim import SSIM
from vits.utils import load_class
from vits.losses import mle_loss
from vits.commons import clip_grad_value_


def load_part(model, saved_state_dict):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('TODO'):
            new_state_dict[k] = v
        else:
            new_state_dict[k] = saved_state_dict[k]
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model


def load_model(model, saved_state_dict):
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
        except:
            print("%s is not in the checkpoint" % k)
            new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    return model


def train(rank, args, chkpt_path, hp, hp_str):

    if args.num_gpus > 1:
        init_process_group(backend=hp.dist_config.dist_backend, init_method=hp.dist_config.dist_url,
                           world_size=hp.dist_config.world_size * args.num_gpus, rank=rank)

    torch.cuda.manual_seed(hp.train.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    model_g = load_class(hp.train.model)(
        hp.data.mel_channels,
        hp.data.segment_size // hp.data.hop_length,
        hp).to(device)

    optim_g = torch.optim.AdamW(model_g.parameters(),
                                lr=hp.train.learning_rate, betas=hp.train.betas, eps=hp.train.eps)

    init_epoch = 1
    step = 0

    # define logger, writer, valloader, stft at rank_zero
    if rank == 0:
        pth_dir = os.path.join(hp.log.pth_dir, args.name)
        log_dir = os.path.join(hp.log.log_dir, args.name)
        os.makedirs(pth_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, '%s-%d.log' % (args.name, time.time()))),
                logging.StreamHandler()
            ]
        )
        logger = logging.getLogger()
        writer = MyWriter(hp, log_dir)
        valloader = create_dataloader_eval(hp)

    if os.path.isfile(hp.train.pretrain):
        if rank == 0:
            logger.info("Start from 32k pretrain model: %s" % hp.train.pretrain)
        checkpoint = torch.load(hp.train.pretrain, map_location='cpu')
        load_model(model_g, checkpoint['model_g'])

    if chkpt_path is not None:
        if rank == 0:
            logger.info("Resuming from checkpoint: %s" % chkpt_path)
        checkpoint = torch.load(chkpt_path, map_location='cpu')
        load_model(model_g, checkpoint['model_g'])
        optim_g.load_state_dict(checkpoint['optim_g'])
        init_epoch = checkpoint['epoch']
        step = checkpoint['step']

        if rank == 0:
            if hp_str != checkpoint['hp_str']:
                logger.warning("New hparams is different from checkpoint. Will use new.")
    else:
        if rank == 0:
            logger.info("Starting new training run.")

    if args.num_gpus > 1:
        model_g = DistributedDataParallel(model_g, device_ids=[rank])

    # this accelerates training when the size of minibatch is always consistent.
    # if not consistent, it'll horribly slow down.
    torch.backends.cudnn.benchmark = True

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hp.train.lr_decay, last_epoch=init_epoch-2)

    spkc_criterion = nn.CosineEmbeddingLoss()

    mel_ssim = SSIM()

    trainloader = create_dataloader_train(hp, args.num_gpus, rank)

    for epoch in range(init_epoch, hp.train.epochs):

        trainloader.batch_sampler.set_epoch(epoch)

        if rank == 0 and epoch % hp.log.eval_interval == 0:
            with torch.no_grad():
                validate(hp, model_g, valloader, writer, step, device)

        if rank == 0:
            loader = tqdm.tqdm(trainloader, desc='Loading train data')
        else:
            loader = trainloader

        model_g.train()

        for vec, vec_l, pit, spk, spec, spec_l in loader:

            vec = vec.to(device)
            pit = pit.to(device)
            spk = spk.to(device)
            spec = spec.to(device)
            vec_l = vec_l.to(device)
            spec_l = spec_l.to(device)

            # generator
            optim_g.zero_grad()

            mel_fake, z_mask, \
                (z_f, logdet_f, m_p, logs_p), spk_preds = model_g(
                    vec, pit, spec, spk, vec_l, spec_l)

            # Spk Loss
            spk_loss = spkc_criterion(spk, spk_preds, torch.Tensor(spk_preds.size(0))
                                .to(device).fill_(1.0))
            # Mel Loss
            mel_real = spec
            mel_loss = F.l1_loss(mel_fake * z_mask, mel_real * z_mask)

            # Mel ssim
            ssim_loss = mel_ssim(mel_fake, mel_real, z_mask)

            # Kl Loss
            loss_kl = mle_loss(z_f, m_p, logs_p, logdet_f, z_mask)

            # Loss
            loss_g =  ssim_loss + mel_loss + loss_kl + spk_loss * 2
            loss_g.backward()
            clip_grad_value_(model_g.parameters(),  None)
            optim_g.step()

            step += 1
            # logging
            loss_g = loss_g.item()
            loss_k = loss_kl.item()
            loss_m = mel_loss.item()
            loss_i = spk_loss.item()
            loss_s = ssim_loss.item()

            if rank == 0 and step % hp.log.info_interval == 0:
                writer.log_training(loss_g, loss_k, loss_m, loss_s, step)
                logger.info("epoch %d | g %.04f m %.04f s %.04f k %.04f i %.04f | step %d" % (
                    epoch, loss_g, loss_m, loss_s, loss_k, loss_i, step))

        if rank == 0 and epoch % hp.log.save_interval == 0:
            save_path = os.path.join(pth_dir, '%s_%04d.pt'
                                     % (args.name, epoch))
            torch.save({
                'model_g': (model_g.module if args.num_gpus > 1 else model_g).state_dict(),
                'optim_g': optim_g.state_dict(),
                'step': step,
                'epoch': epoch,
                'hp_str': hp_str,
            }, save_path)
            logger.info("Saved checkpoint to: %s" % save_path)

        scheduler_g.step()
