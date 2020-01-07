import numpy as np


def krum(participants):
	scores = np.zeros((len(participants),))

	for i in range(len(participants)):
		gi = participants[i].get_gradients()
		score_i = 0
		for j in range(len(participants)):
			score_i += _distance(gi, participants[j].get_gradients())

		scores[i] = score_i

	return np.argmin(scores), scores

def _distance(g1, g2):
	return np.linalg.norm(g1 - g2, ord=2)