import numpy as np

from _krum import krum


def setup_lp_norm_attack(participants, byzantine_idx, mu, std, w_before, f):
	gamma = 0.5
	gamma_best = gamma

	# choose gamma
	while True:
		[n.setup_attack(mu, std, gamma) for n in participants if n.id in byzantine_idx]
		krum_node_idx, krum_scores = krum(participants, f=f)
		selected_node = participants[krum_node_idx]
		is_byzantine_selected = int(selected_node.id in byzantine_idx)

		if not is_byzantine_selected:
			break

		gamma_best = gamma
		gamma += 0.5

	# prepare byzantines workers
	[n.set_weights(w_before) for n in participants if n.id in byzantine_idx]
	[n.setup_attack(mu, std, gamma_best) for n in participants if n.id in byzantine_idx]
	[n.take_step() for n in participants if n.id in byzantine_idx]
	[n.setup_attack(mu, std, gamma_best) for n in participants if n.id in byzantine_idx]

	return gamma_best
