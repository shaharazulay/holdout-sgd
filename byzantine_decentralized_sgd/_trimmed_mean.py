import torch
import numpy as np
from scipy import stats


def trimmed_mean(participants, f):
	n = len(participants)
	g_array = [p.get_gradients() for p in participants]
	g_orig = torch.stack(g_array)
	g_trimmed_mean = torch.tensor(stats.trim_mean(g_orig, f/n, axis=0))

	return g_trimmed_mean