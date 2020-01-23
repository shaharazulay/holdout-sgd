import torch
import numpy as np


def trimmed_mean(participants, f):
	n = len(participants)
	g_array = [p.get_gradients() for p in participants]
	g_orig = torch.stack(g_array)
	g_median = torch.median(torch.stack(g_array), dim=0)[0]

	g_diff = torch.stack([torch.abs(gi - g_median) for gi in g_array])
	_, indices = torch.sort(g_diff, 0)
	U = g_orig[indices, np.arange(g_orig.shape[1])[None, :]]

	U_top = U[: n - 2 * f - 1, :]
	g_trimmed_mean = torch.mean(U_top, dim=0)

	return g_trimmed_mean