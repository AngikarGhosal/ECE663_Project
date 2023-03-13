import torch
import yaml
import torch.optim as optim

from load_data import *
from GLOW import *


def setup(cfg_filepath):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	with open(cfg_filepath, 'r') as f:
	    cfg = yaml.load(f, yaml.SafeLoader)

	NF_param = cfg["NF"]
	NF_param_net = NF_param["net"]
	optim_param = cfg["optim"]
	data_param = cfg["dataset"]

	# Creating NF object
	nf_modules = []
	mask = torch.arange(0, NF_param_net['num_inputs']) % 2
	mask = mask.to(device).float()
	for _ in range(NF_param["num_blocks"]):
	    nf_modules += [
	        BatchNormFlow(NF_param_net['num_inputs']),
	        LUInvertibleMM(NF_param_net['num_inputs']),
	        CouplingLayer(**NF_param_net, mask=mask)
	    ]
	    mask = 1 - mask


	NF = FlowSequential(*nf_modules).to(device)
	NF.num_inputs = NF_param_net['num_inputs']
	for module in NF.modules():
	    if isinstance(module, nn.Linear):
	        nn.init.orthogonal_(module.weight)
	        if hasattr(module, 'bias') and module.bias is not None:
	            module.bias.data.fill_(0)

	# NF optimizers         
	optimizer = getattr(optim, optim_param['name'])(NF.parameters(), lr = optim_param["lr"])

	# DataLoaders Not implemented yet
	data_loader = load_data(**data_param)

	train_cfg = {
	"NF": NF,
	"optimizer": optimizer,
	"data_loader": data_loader,
	"n_epoch": optim_param["n_epochs"],
	"device": device,
	"backdoor": False,
	}

	return train_cfg






