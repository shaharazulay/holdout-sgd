import numpy as np


def krum(participants, f):
	n = len(participants)
	scores = np.zeros((n,))

	for i in range(n):
		gi = participants[i].get_gradients()
		dists_i = []
		for j in range(len(participants)):
			dists_i.append(_distance(gi, participants[j].get_gradients()))
		score_i = np.mean(sorted(dists_i)[: n - f - 1])

		scores[i] = score_i

	return np.argmin(scores), scores


def _distance(g1, g2):
	return np.linalg.norm(g1 - g2, ord=2)
