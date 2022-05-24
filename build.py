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
with open("data/stopword.txt",encoding="utf-8") as f:
	stopwords = set(f.read().split("\n"))
stopwords.add(" ")
def insert_by_batch(con,command,array):
	cur = con.cursor()
	for batch in [array[i:i+2000] for i in range(len(array))[::2000]]:
		cur.executemany(command,batch)
	con.commit()
def init(con):
	cur = con.cursor()
	cur.execute("""create table if not exists words( word_index int primary key,word varchar(256))""")
	cur.execute("""create table if not exists inverse_index(word_index int,document int,tfidf float,PRIMARY KEY (word_index,document))""")
	cur.execute("""create table if not exists dc(word_index int primary key,document_count int)""")
	cur.execute("""create table if not exists cases( id integer primary key,qw text,writId varchar(64),path varchar(256))""")
	con.commit()



	charges = json.load(open("data/documents/common_charge.json",encoding="utf-8"))
	query_related = {v for key in charges for v in charges[key][:100] if v.endswith(".json")}
	with open("data/query.json",encoding="utf-8") as f:
		for i in f.readlines():
			t =  json.loads(i)
			if t["path"].endswith(".json"):
				query_related.add(t["path"])
	data = []
	for item in query_related:
		obj = json.load(open("data/documents/documents/{}".format(item),encoding="utf-8"))
		obj["path"] = item
		data.append(obj)

	going_inserted = [(index,i["qw"],i["writId"],i["path"]) for index,i in enumerate(data)]
	insert_by_batch(con,"insert or ignore into cases values (?,?, ?,?)",going_inserted)
	data = pd.DataFrame(data)
	data = data[["qw","writId","path"]]
	document = data["qw"].to_list()
	print("读取数据完成")
	return document


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
	jieba.load_userdict('data/criminal_charges.txt')
def merge_list_of_dictionaries(dict_list):
	# 合并进程数据
	new_dict = defaultdict(list)
	for d in tqdm.tqdm(dict_list):
		for key in d:
			new_dict[key].append(d[key])
	return new_dict
	

if __name__ == "__main__":
	# 统计
	print("初始化")
	con = sqlite3.connect("index.db")
	document = init(con)
	print("分词和统计")
	with multiprocessing.Pool(4,initializer=init_process) as pool:
		inverted_index = defaultdict(list)
		result  = pool.map(f, enumerate(document))
		result = list(tqdm.tqdm(pool.imap(f, enumerate(document)), total=len(document)))
	result = merge_list_of_dictionaries(result)
	words = list(result.keys())
	insert_by_batch(con,"insert or ignore into words values (?, ?)",[(i,k) for i,k in enumerate(words)])
	print("计算idf")
	cur = con.cursor()

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
