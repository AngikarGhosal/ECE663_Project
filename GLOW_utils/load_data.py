import torchvision
from torchvision.datasets import USPS
from torchvision.datasets import KMNIST
from torchvision import transforms
import torch

def load_data(data_type="USPS"):
    if data_type=="USPS":
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((24,24)),
             transforms.Normalize((0.5,), (0.5,)),
             ])

        usps_dataset = torchvision.datasets.USPS(root='../usps_24', download=True, transform=transform)

        data_loader = torch.utils.data.DataLoader(
            usps_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True
        )


    return data_loader



