import pandas as pd
import numpy as np

# ------------------------
# Funzioni di utilità
# ------------------------
def bottom_itemset(df, weight_col=None):
	"""Costruisce l'itemset iniziale {col: (0, max)} per ogni attributo"""
	base = {}
	for col in df.columns:
		if col == weight_col:
			continue
		max_val = int(np.ceil(df[col].max()))
		base[col] = (0, max_val)
	return base

def delta(itemset, max_vals):
	"""
	Calcola Δ(I) = somma dei δ su tutte le colonne:
	δ((l,u), [0,max]) = (l - 0) + (max - u)
	"""
	return sum((l - 0) + (max_vals[col] - u) for col, (l,u) in itemset.items())

def support(df, itemset, weight_col=None):
	"""Calcola il supporto relativo di un itemset quantitativo"""
	if weight_col is None:
		df["__w__"] = 1
		weight_col = "__w__"
	total_weight = df[weight_col].sum()

	mask = np.ones(len(df), dtype=bool)
	for col, (low, high) in itemset.items():
		mask &= (df[col] >= low) & (df[col] < high)

	num_weight = df.loc[mask, weight_col].sum()
	return num_weight / total_weight

# ------------------------
# Apriori Quantitativo (versione delta professore)
# ------------------------
def apriori_quant_prof(df, epsilon, weight_col=None):
	R = {}  # itemset frequenti
	attributes = [col for col in df.columns if col != weight_col]
	max_vals = {col: int(np.ceil(df[col].max())) for col in attributes}

	# Itemset iniziale
	I0 = {col: (0, max_vals[col]) for col in attributes}
	SWk = [I0]

	while SWk:
		Wk = []

		# Genera candidati
		for I in SWk:
			for a in attributes:
				low, high = I[a]
				# shrink a sinistra e a destra
				for mid in range(low + 1, high):
					# shrink sinistra
					new_I_left = I.copy()
					new_I_left[a] = (low, mid)
					if delta(new_I_left, max_vals) == delta(I, max_vals) + 1:
						Wk.append(new_I_left)

					# shrink destra
					new_I_right = I.copy()
					new_I_right[a] = (mid, high)
					if delta(new_I_right, max_vals) == delta(I, max_vals) + 1:
						Wk.append(new_I_right)

		# Valuta supporto
		SWk = []
		for I in Wk:
			s = support(df, I, weight_col)
			if s >= epsilon:
				R[str(I)] = s
				SWk.append(I)

	return R

df = pd.DataFrame({
    'A1': [1.2, 2.5, 3.1, 2.9, 1.5],
    'A2': [5.0, 5.5, 6.0, 5.2, 5.1],
    'weight': [1,1,1,1,1]
})

results = apriori_quant_prof(df, epsilon=0.2, weight_col='weight')

print("\n--- Frequent itemsets ---")
for item, sup in results.items():
    print(f"{item} -> support = {sup:.2f}")
