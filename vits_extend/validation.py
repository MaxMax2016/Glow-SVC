import tqdm
import torch
import torch.nn.functional as F


def validate(hp, generator, valloader, writer, step, device):
    generator.eval()
    torch.backends.cudnn.benchmark = False

    loader = tqdm.tqdm(valloader, desc='Validation loop')
    mel_loss = 0.0
    for idx, (vec, vec_l, pit, spk, spec, spec_l) in enumerate(loader):
        vec = vec.to(device)
        pit = pit.to(device)
        spk = spk.to(device)
        spec = spec.to(device)
        vec_l = vec_l.to(device)

        if hasattr(generator, 'module'):
            mel_fake = generator.module.infer(vec, pit, spk, vec_l)
        else:
            mel_fake = generator.infer(vec, pit, spk, vec_l)

        mel_real = spec
        mel_loss += F.l1_loss(mel_fake, mel_real).item()

        if idx < hp.log.num_audio:
            mel_fake = mel_fake[0].cpu().detach().numpy()
            mel_real = mel_real[0].cpu().detach().numpy()
            writer.log_fig_mel(mel_fake, mel_real, idx, step)

    mel_loss = mel_loss / len(valloader.dataset)
    writer.log_validation(mel_loss, step)

    torch.backends.cudnn.benchmark = True
