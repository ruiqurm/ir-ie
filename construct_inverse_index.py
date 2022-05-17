import multiprocessing
import jieba
import tqdm
from collections import defaultdict
import os
import json
import pandas as pd
import numpy as np
from typing import Dict
import sqlite3
from scipy.sparse import lil_matrix

con = sqlite3.connect("index.db")
cur = con.cursor()
cur.execute("""create table if not exists words( word_index int primary key,word varchar(256))""")
cur.execute("""create table if not exists inverse_index(word_index int,document int,tfidf float,PRIMARY KEY (word_index,document))""")
cur.execute("""create table if not exists dc(word_index int primary key,document_count int)""")
con.commit()
# 预处理数据
data = []
for directory, dirnames, filenames in os.walk(r'./data/candidates'):
	for name in filenames:
		json_file = os.path.join(directory, name)
		data.append(json.load(open(json_file,encoding="utf-8")))
data = pd.DataFrame(data)
data = data[["qw"]].to_dict()["qw"]
data = [data[i] for i in range(len(data.keys()))]
document = data # 换个名字

# 读取停用词
with open("LeCaRD/data/others/stopword.txt",encoding="utf-8") as f:
	stopwords = set(f.read().split("\n"))
stopwords.add(" ")

# 定义结构体
class Word:
	def __init__(self,document_id,tf):
		self.document_id = document_id
		self.tf = tf

def f(arg)->Dict[str,Word]:
	# 处理函数
	i,passage = arg
	ret = dict()
	seg = [i for i in list(jieba.cut(passage,cut_all=True)) if i not in stopwords]
	seg_count_dict = pd.Series(seg).value_counts().to_dict()
	for word in seg_count_dict:
		tf = seg_count_dict[word]
		ret[word] = Word(i,tf)
	return ret
def init_process():
	# 初始化进程
	jieba.load_userdict('LeCaRD/data/others/criminal charges.txt')
def merge_list_of_dictionaries(dict_list):
	# 合并进程数据
	new_dict = defaultdict(list)
	for d in tqdm.tqdm(dict_list):
		for key in d:
			new_dict[key].append(d[key])
	return new_dict
if __name__ == "__main__":
	# 统计
	with multiprocessing.Pool(8,initializer=init_process) as pool:
		inverted_index = defaultdict(list)
		result  = pool.map(f, enumerate(document))
		result = list(tqdm.tqdm(pool.imap(f, enumerate(document)), total=len(document)))
	result = merge_list_of_dictionaries(result)
	words = list(result.keys())
	cur.executemany("insert or ignore into words values (?, ?)",enumerate(words))
	con.commit()
	print("计算idf")
	
	matrix = np.zeros((len(words),len(document)))
	for i,key in enumerate(tqdm.tqdm(words)):
		# construct matrix
		for doc in result[key]:
			j = doc.document_id
			matrix[i,j] = 1
	print("回填和计算tfidf")
	for key_id,df in enumerate(tqdm.tqdm(matrix.sum(axis=1))):
		# 记录df
		# all_df[key_id] = df
		cur.execute("insert or ignore into dc values (?, ?)",(key_id,df))
		# 计算idf
		for word in result[words[key_id]]:
			idf = np.log(len(document)/df)
			word.idf = idf
			word.tfidf = word.tf*word.idf
			if word.tfidf >= 0.0001:
				cur.execute("insert or ignore into inverse_index values (?, ?,?)",(key_id,word.document_id,word.tfidf))
	con.commit()


	# result[words[key_id]].sort(key=lambda x:x.tfidf,reverse=True)		
	# del matrix
	# # regenerate a matrix for document vector
	print("保存文档矩阵")
	_matrix = lil_matrix((len(words),len(document)), dtype=np.float32)
	for word_index,word in enumerate(tqdm.tqdm(words)):
		for token in result[word]:
			_matrix[word_index,token.document_id]= token.tfidf
	import scipy
	matrix = _matrix.tocsr()
	scipy.sparse.save_npz('document_matrix.npz',matrix)
	# np.save("document_tfidf.npy",matrix)
	# pickle.dump({"inverse_index":result,"document_vector":matrix,"df":all_df},open("inverted_index.pkl","wb"))
