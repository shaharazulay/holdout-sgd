import torch
import numpy as np
from functools import reduce


def get_average_gradients(participants):
	g_array = [p.get_gradients() for p in participants]
	return torch.mean(torch.stack(g_array), dim=0)
