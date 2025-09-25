import pandas as pd
import numpy as np
import random
from multiprocessing import Manager, Process, freeze_support

# ======================
# Support function
# ======================
def support(I, data, class_col):
	mask = np.ones(len(data), dtype=bool)
	for a, (b, e) in I.items():
		mask &= (data[a] >= b) & (data[a] < e)
	total = data[class_col].sum()
	return data.loc[mask, class_col].sum() / total if total > 0 else 0

# ======================
# Predecessor / Successor checks
# ======================
def is_predecessor(Ip, I):
	for a in I:
		b1, e1 = Ip[a]
		b2, e2 = I[a]
		if not (b1 <= b2 and e2 <= e1):
			return False
	return True

def is_successor(In, I):
	for a in I:
		b1, e1 = In[a]
		b2, e2 = I[a]
		if not (b1 <= b2 and e2 <= e1):
			return False
	return True

# ======================
# Worker distribuito
# ======================
def distributed_worker(GP, GS, GNS, data, epsilon, class_col, attributes):
	LS, LNS = {}, {}
	while True:
		try:
			I = GP.get_nowait()  # get itemset casuale
		except:
			break

		key = str(I)
		if key in LS or key in LNS:
			continue

		# Pruning globale tramite predecessori / successori
		if any(is_predecessor(eval(k), I) for k in GS.keys()) or \
		   any(is_successor(eval(k), I) for k in GNS.keys()):
			continue

		s = support(I, data, class_col)

		if s >= epsilon:
			GS[key] = s
			LS[key] = s
			# Generazione figli
			for a in attributes:
				b, e = I[a]
				if b + 1 < e:
					new_I = I.copy()
					new_I[a] = (b + 1, e)
					GP.put(new_I)
				if b < e - 1:
					new_I = I.copy()
					new_I[a] = (b, e - 1)
					GP.put(new_I)
		else:
			GNS[key] = s
			LNS[key] = s

# ======================
# Randomic Apriori Distribuito
# ======================
def apriori_randomic_distributed(data, epsilon, class_col='class', num_workers=2):
	attributes = [c for c in data.columns if c != class_col]
	max_vals = {a: int(np.ceil(data[a].max())) for a in attributes}
	I0 = {a: (0, max_vals[a]) for a in attributes}

	manager = Manager()
	GP = manager.Queue()
	GS = manager.dict()
	GNS = manager.dict()

	# seed iniziale
	GP.put(I0)

	# avvio worker
	processes = [Process(target=distributed_worker,
						 args=(GP, GS, GNS, data, epsilon, class_col, attributes))
				 for _ in range(num_workers)]
	for p in processes:
		p.start()
	for p in processes:
		p.join()

	return dict(GS)


# ======================
# Esecuzione
# ======================
if __name__ == "__main__":
	freeze_support() # necessaria su windows

	data = pd.DataFrame({
	'A1': [1.2, 2.5, 3.1, 2.9, 1.5], 
	'A2': [5.0, 5.5, 6.0, 5.2, 5.1], 
	'C': [1, 1, 1, 1, 1]})

	epsilon = 0.4

	GS = apriori_randomic_distributed(data, epsilon, 'C', num_workers=2)
	print("Distributed Randomic Apriori Result:")
	for k, v in GS.items():
		print("itemset: " + str(k)  + "  " + "support: " +str(v))
		