from sklearn.decomposition import TruncatedSVD
from scipy import sparse
import numpy as np
svd = TruncatedSVD(n_components=1500, n_iter=7, random_state=42)
svd.n_jobs=-1
matrix = sparse.load_npz("./document_matrix.npz")
matrix = matrix.T
svd.fit(matrix)
import pickle
pickle.dump(svd,open("./lsa.pkl","wb"))
matrix = svd.transform(matrix)
np.save("./compress_matrix",matrix)
