from fastapi import FastAPI
import pydantic
import uvicorn
import os,json,re
import pandas as pd
import jieba
import tqdm
from collections import defaultdict
import numpy as np
import sqlite3
from scipy import sparse
from scipy.sparse.linalg import norm
from typing import List,Tuple
from fastapi.middleware.cors import CORSMiddleware

"""
init
"""
if not os.path.isfile("index.db") or not os.path.exists("document_matrix.npz"):
    raise Exception("不存在索引文件或文档矩阵")
    
conn = sqlite3.connect("index.db")
cur = conn.cursor()
cur.execute("select name from sqlite_master where type = 'table';")
cur.fetchall()

matrix = sparse.load_npz("document_matrix.npz")
matrix = matrix.T
cur.execute("""select * from words""")
words = [word[1] for word in cur.fetchall()]
cur.execute("""select * from dc""")
word_doc_count = [dc[1] for dc in cur.fetchall()]
words_idx = {word:idx for idx,word in enumerate(words)}
documents_length = cur.execute("""select count(*) from cases""").fetchone()[0]


def get_documents_matrix(keys:list):
	l = []
	for key in keys:
		if key in words:
			l.append(words_idx[key])
	sql = """select document from inverse_index where word_index in ({})""".format(",".join(["?"]*len(l)) )
	cur.execute(sql,l)
	return np.array(list({i[0] for i in cur.fetchall()}))
	
def query(key:str,n:int=5,ret_parse_key=False):
	key_with_count = pd.Series(list(jieba.cut(key))).value_counts().to_dict()
	_x = sparse.lil_matrix((len(words),1))
	# 生成查询向量的tf-idf
	for word in key_with_count:
		if word in words:
			pos = words_idx[word]
			_x[pos,0] = key_with_count[word] * np.log( (documents_length+1) / (1 + word_doc_count[pos]))
	x = _x.tocsr()
	# 分词，利用倒排索引
	docuement_to_query = get_documents_matrix(key_with_count.keys())
	m = matrix[docuement_to_query]
	# 计算查询向量与文档向量的余弦相似度
	cosine_distance = (m@x).toarray().flatten()/(norm(m,axis=1) * norm(x))

	ind = np.argpartition(cosine_distance, -n)[-n:] # ind -> cosine
	unorder_result = cosine_distance[ind]
	result_order = np.flip(np.argsort(unorder_result))
	# print(ind[result_order])
	# print(unorder_result[result_order])
	# import pdb;pdb.set_trace()
	doc_ids = docuement_to_query[ind[result_order]]
	documents = get_documents(doc_ids)
	result =  [{"id":int(i),"cosine_distance":float(j),"title":k[0],"fact":k[1],"verdict":k[2]} for i,j,k in zip(doc_ids,unorder_result[result_order],documents)]
	if ret_parse_key:
		return result,key_with_count.keys()
	else:
		return (result,)

def get_documents(document_ids):
	command = "select ajName as case_name,ajjbqk as basic_fact,cpfxgc||pjjg as court_verdict from cases where id in ({})"\
				.format(",".join([str(i) for i in document_ids]))
	cur.execute(command)
	return cur.fetchall()

app = FastAPI()

origins = ["*"]
 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/query")
async def q(key:str,n:int=10):
	result,key = query(key,n,ret_parse_key=True)

	return {"result":result,"keys":list(key)}