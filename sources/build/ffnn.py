from sklearn.datasets import make_classification

from sources.ffnn.gen_ffnn import GenerateFFNN

X, y = make_classification(n_samples=1000, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

model = GenerateFFNN(X, y)
result = model.predict([[-0.41713292, -0.32475336, 0.0686618, -0.15090511], [0, 0, 0, 0]])
print(result)
