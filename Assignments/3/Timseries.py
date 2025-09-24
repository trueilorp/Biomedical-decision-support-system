import numpy as np
import random
import pandas as pd
from sktime.datasets import load_from_tsfile_to_dataframe
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def euclidean_distance(x_slice, x_ref):
	x_ref_value = float(np.array(x_ref).flatten()[0])
	return np.sqrt(np.sum((x_slice - x_ref_value) ** 2))

def manhattan_distance(x_slice, x_ref):
	x_ref_value = float(np.array(x_ref).flatten()[0])
	return np.sum(np.abs(x_slice - x_ref_value))


def cosine_distance(x_slice, x_ref):
	"""
	Distanza coseno tra x_ref e x_slice trattata come vettore
	Se vuoi element-wise, considera x_ref broadcasted
	"""
	x_ref_value = float(np.array(x_ref).flatten()[0])
	x_slice_vec = np.array(x_slice)
	# coseno tra vettori: x_ref replicato
	dot = x_slice_vec * x_ref_value
	norm = np.linalg.norm(x_slice_vec) * np.linalg.norm([x_ref_value])
	return 1 - np.sum(dot) / (norm + 1e-9)

class Node():
	def __init__(self, x_barr, channel, bi, ei, df, t):
		self.x_barr = x_barr
		self.channel = channel
		self.begin_idx = bi
		self.end_idx = ei
		self.distance_f_node = df 
		self.threshold = t
		
		# figli
		self.node_dx = None # Node()
		self.node_sx = None # Node()

class LeafNode():
	def __init__(self, class_f):
		self.classification_f = class_f
		
class PromptTree():
	def __init__(self, X, y):
		# Require
		self.X = X
		self.y = y
		self.current_path = None # lista di dizionari
		self.distance_functions = self.get_distance_functions()
		self.root = self.prompt_tree_fit_algo(self.X, self.y, self.current_path)
		
		self.v = [] # vertices
		self.e = [] # edges
		self.s = [] # rst
		self.l = [] # labels
	
	# store informations in the nodes
	def prompt_tree_fit_algo(self, X, y, path):
		print("\n==============================")
		print("Inizio nuova ricorsione...")
		if path is None:
			path = []
			B = [0]
			E = [0]
		else:
			B, E = [], []
			for node in path:
				B.append(node.begin_idx)
				E.append(node.end_idx)

		print("B (begin indices):", B)
		print("E (end indices):", E)
		print("Numero esempi a questo livello:", len(X))

		# Condizione di stop
		if self.stopping_f(path, X, y):
			leaf_class = self.classification_f(X, y)
			print(f"FOGLIA trovata! Classificazione: {leaf_class}")
			return LeafNode(leaf_class)

		# Candidate intervals
		candidate_intervals = self.promptness_f(X, y, B, np.max(E) + 1)
		print("Candidate intervals:", candidate_intervals)

		# Candidate tests
		candidate_tests = self.sampling_f(X, y, candidate_intervals)
		print("Candidate tests:", candidate_tests)

		# Optimal candidate
		optimal_candidate, df_true, df_false = self.optimization_f(X, y, candidate_tests)
		print("Optimal candidate selezionato:", optimal_candidate)

		# Recupero valori del nodo
		x_barr, c, begin_idx, end_idx, dist_f, threshold = (
			optimal_candidate[k] for k in ['x_barr', 'channel', 'b', 'e', 'dist_fun', 'threshold']
		)

		# Creazione nodo
		node = Node(x_barr, c, begin_idx, end_idx, dist_f, threshold)
		print(f"NODO creato: channel={c}, begin={begin_idx}, end={end_idx}, threshold={threshold}")
		print(f"x_barr={x_barr}")

		# Preparazione dati per i rami sinistro e destro
		X_true = pd.DataFrame([x for x, _ in df_true], columns=X.columns)
		y_true = np.array([label for _, label in df_true])
		X_false = pd.DataFrame([x for x, _ in df_false], columns=X.columns)
		y_false = np.array([label for _, label in df_false])
		print(f"Esempi ramo LEFT (<= threshold): {len(X_true)}, RIGHT (> threshold): {len(X_false)}")

		# Ricorsione ramo sinistro
		print("=== Ricorsione ramo LEFT ===")
		path_true = path.copy()
		path_true.append(node)
		node.node_sx = self.prompt_tree_fit_algo(X_true, y_true, path_true)

		# Ricorsione ramo destro
		print("=== Ricorsione ramo RIGHT ===")
		path_false = path.copy()
		path_false.append(node)
		node.node_dx = self.prompt_tree_fit_algo(X_false, y_false, path_false)

		print(f"Ritorno nodo channel={c} a livello superiore")
		return node

			
	def post_pruning (self):
		pass
	
	def get_paths_of_tree(Self):
		pass


	def promptness_f(self, X, y, B, max_e):
		''' Propone un set di coppie: canale, intervallo'''
		total_pairs = random.randint(1, len(B)//2 + 1)
		pairs = []
		
		channels = list(X.columns) # get channels
		k = random.randint(1, len(channels)) # 
		selected_channels = random.sample(channels, k) # select some channels inside channels
		
		for channel in selected_channels: # loop over selected channels
			for i in range(total_pairs):
				b = random.randrange(B[0], max_e)
				# print(b, max_e)
				pair = {'channel': channel, "interval": (b, max_e)}
				pairs.append(pair)
		return pairs

	def sampling_f(self, X, y, candidate_intervals):
		candidate_tests = []
		for candidate in candidate_intervals:
			# generare un test candidate, per cui aggiungo threshold, distance function
			channel = candidate['channel']
			b = candidate['interval'][0]
			e = candidate['interval'][1]
			dist_fun = random.choice(self.distance_functions)
			threshold = random.uniform(0,2)
			
			# seleziono il sample di riferimento considerando self.X come un dataframe, forse devo fare un for
			ref_idx = np.random.randint(0, X.shape[0]) # selezione l'indice del sample random
			cols = [col for col in X.columns if col.startswith(channel)] # estrapolo solo le colonne relative al canale
			x_barr = X.iloc[ref_idx][cols].apply(lambda s: s[b:e]).values # seleziono le colonne del canale per quel sample, estraggo l’intervallo [b:e] da ogni serie, 
																		  # .values converte in array NumPy, pronto per calcolare la distanza
			
			candidate_test = { 'x_barr': x_barr,
							   'channel' : channel, 
							   'b': b,
							   'e' : e, 
							   'dist_fun' : dist_fun, 
							   'threshold' : threshold}
			candidate_tests.append(candidate_test)
		return candidate_tests
	
	def optimization_f(self, X, y, candidate_tests):
		entropy_test = 1000
		df_true = None
		df_false = None
		for candidate in candidate_tests:
			candidate_tests_true = []
			candidate_tests_false = []
			# print("Candidate:", candidate)
			for i, (idx, row) in enumerate(X.iterrows()):
				# print("Row:", row)
				# col = [col for col in X.columns if col.startswith(candidate['channel'])]
				col = candidate['channel']
				# print("Col:", col)
				slice_x = row[col].iloc[candidate['b']:candidate['e']].values # slice all'interno della riga
				# print("Slice len:", len(slice_x))
				# print("Candidate x_barr:", candidate['x_barr'])
				# print("Slice x:", slice_x)
				# if candidate['e'] > len(slice_x): # CONTROLLARE PERCHE' NON PUO' ESSERE PIU GRANDE END_INDEX
				# 	# print(f"Warning: candidate['e']={candidate['e']} > slice length={len(slice_x)}")
				# 	continue
				dist = np.mean(candidate['dist_fun'](slice_x, candidate['x_barr'])) # calcola la distanze di ogni elemento del mio slice da x_barr, dopodichè faccio la media cosi ottengo solo 1 valore
				# print("Dist:", dist)
				
				if (dist <= candidate['threshold']):
					candidate_tests_true.append((row, y[i])) # tutto il sample con la sua label va nel ramo true
				else:
 					candidate_tests_false.append((row, y[i]))
						
			y_true = [label for _, label in candidate_tests_true] # passo tutte le etichette di entrambi i subset
			y_false = [label for _, label in candidate_tests_false] # mi tengo solo le label
			
			# print("Y_TRUE:", y_true)
			# print("Y_FALSE:", y_false)
			current_entropy = self.calculate_entropy(y_true, y_false)
			if (current_entropy < entropy_test): # calcolo l'entropia sulle label
				best_test_candidate = candidate
				entropy_test = current_entropy
				df_true = candidate_tests_true
				df_false = candidate_tests_false
		return best_test_candidate, df_true, df_false
		# x_barr, c, begin_idx, end_idx, dist_f, threshold
		
	def entropy(self, labels):
		from collections import Counter
		import math
		n = len(labels)
		if n == 0:
			return 0.0
		counts = Counter(labels)
		probs = [c/n for c in counts.values()]
		return -sum(p * math.log2(p) for p in probs if p > 0)
	
	def calculate_entropy(self, y_true, y_false):
		n_true = len(y_true)
		n_false = len(y_false)
		total = n_true + n_false
		H_true = self.entropy(y_true)
		H_false = self.entropy(y_false)
		H_total = (n_true/total) * H_true + (n_false/total) * H_false
		return H_total

	def classification_f(self, X, y):
		# Ritorna distribuzione semplice delle classi
		classes, counts = np.unique(y, return_counts=True)
		distr = {cls: cnt/len(y) for cls, cnt in zip(classes, counts)}
		return distr
	
	def stopping_f(self, path, X, y): # da includere anche path in qualche modo
		'''Stop when samples are less than a threshold'''
		return len(y) < 20  # se pochi campioni, fermati

	
	def get_distance_functions(self):
		return [euclidean_distance, manhattan_distance, cosine_distance]

# Load .ts file
X_train, y_train = load_from_tsfile_to_dataframe("C:/Users/Simone/Desktop/UNIVERSITA/MAGISTRALE/BIOMEDICAL DECISION SUPPORT SYSTEM/Multivariate_ts/Cricket/Cricket_TRAIN.ts")
X_test, y_test = load_from_tsfile_to_dataframe("C:/Users/Simone/Desktop/UNIVERSITA/MAGISTRALE/BIOMEDICAL DECISION SUPPORT SYSTEM/Multivariate_ts/Cricket/Cricket_TEST.ts")

# X_train, y_train = load_from_tsfile_to_dataframe("C:/Users/Simone/Desktop/UNIVERSITA/MAGISTRALE/BIOMEDICAL DECISION SUPPORT SYSTEM/Univariate_ts/Car/Car_TRAIN.ts")
# X_test, y_test = load_from_tsfile_to_dataframe("C:/Users/Simone/Desktop/UNIVERSITA/MAGISTRALE/BIOMEDICAL DECISION SUPPORT SYSTEM/Univariate_ts/Car/Car_TEST.ts")

# Encoding label if necessary
le = LabelEncoder()
y_train = le.fit_transform(y_train)
y_test = le.transform(y_test)

# for i in range(3):
# 	print(f"Trace {i}:")
# 	for col in X_train.columns:
# 		print("Column:", col)
# 		serie = X_train.iloc[i][col]
# 		print(f"  Channel {col}: {serie.head(10).tolist()}")
# 	print("Label:", y_train[i])
# 	print("----")


# # Plot channels
# sample = X_train.iloc[0]
# plt.figure(figsize=(12, 6))
# for col in sample.index:
# 	series = sample[col]  # pandas.Series con i valori temporali del canale
# 	plt.plot(series.values, label=col)
# plt.xlabel("Tempo")
# plt.ylabel("Valore")
# plt.legend()
# plt.show()

### CONSTANT


# Create prompt_tree
prompt_tree = PromptTree(X_train, y_train)
def print_tree(node, prefix=""):
	"""Stampa ricorsiva di Node e LeafNode in modo leggibile"""
	if isinstance(node, LeafNode):
		print(f"{prefix}└── Leaf: class={node.classification_f}")
		return
	elif isinstance(node, Node):
		print(f"{prefix}└── Node: channel={node.channel}, x_barr={node.x_barr.flatten()}, "
			  f"begin={node.begin_idx}, end={node.end_idx}, threshold={node.threshold}")
		
		if node.node_sx is not None or node.node_dx is not None:
			# Determina i prefissi per sinistro e destro
			if node.node_sx is not None:
				print(f"{prefix}    Left:")
				print_tree(node.node_sx, prefix + "        ")
			if node.node_dx is not None:
				print(f"{prefix}    Right:")
				print_tree(node.node_dx, prefix + "        ")


print_tree(prompt_tree.root)