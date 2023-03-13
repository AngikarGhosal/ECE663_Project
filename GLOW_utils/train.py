import yaml
import torch
from GLOW import *
from setup_glow import *
import torch.optim as optim
import sys

from tqdm.auto import tqdm


def train_GLOW(NF, optimizer, data_loader, n_epoch, device, backdoor = False):
    # Training function

    pbar = tqdm(total=n_epoch)
    train_loss_list = []
    for i in range(n_epoch):
        train_loss = 0
        for batch_idx, data in enumerate(data_loader):
            # Hardcoded shape for USPS, update in v2
            x = data[0].float().reshape(-1, 24*24).to(device)
            t = data[1].float().to(device)
                
            optimizer.zero_grad()
            loss = (-NF.log_probs(x, t.reshape(len(x),1))).mean()

            # Glow Latent Space backdoor attack
            if backdoor:
                loss = (-NF.log_probs_backdoor(x, t.reshape(len(x),1))).mean() 

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            pbar.set_description("Curr E-b: {}-{} | NF Loss: {}".format(
                    i+1, 
                    batch_idx+1,
                    round(loss.item(), 2)
            ))

        train_loss_list.append(train_loss/(batch_idx + 1))

    return NF, train_loss_list


if __name__ == '__main__':
    cfg_filepath = "NF_params.yaml"
    if("-f" in  sys.argv):
        cfg_filepath = sys.argv[sys.argv.index("--f") + 1]

    train_cfg = setup(cfg_filepath)
    NF, train_loss_list = train_GLOW(**train_cfg)

    torch.save(NF.state_dict(), "/NF.pt")





