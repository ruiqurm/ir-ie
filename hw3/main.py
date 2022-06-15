import torch
from model import *
from utils import *
from config import *
from dataset import *
import pandas as pd
from tqdm import tqdm

# def get_len_max():
# 	patient_notes = pd.read_csv('input/nbme-score-clinical-patient-notes/patient_notes.csv')
# 	for text_col in ['pn_history']:
# 		pn_history_lengths = []
# 		tk0 = tqdm(patient_notes[text_col].fillna("").values, total=len(patient_notes))
# 		for text in tk0:
# 			length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
# 			pn_history_lengths.append(length)
# 	features = pd.read_csv('input/nbme-score-clinical-patient-notes/features.csv')
# 	def preprocess_features(features):
# 		features.loc[27, 'feature_text'] = "Last-Pap-smear-1-year-ago"
# 		return features
# 	features = preprocess_features(features)
# 	for text_col in ['feature_text']:
# 		features_lengths = []
# 		tk0 = tqdm(features[text_col].fillna("").values, total=len(features))
# 		for text in tk0:
# 			length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
# 			features_lengths.append(length)
# 	CFG.max_len = max(pn_history_lengths) + max(features_lengths) + 3 # cls & sep & sep

"""
初始化
"""
features = pd.read_csv('input/nbme-score-clinical-patient-notes/features.csv')
fe = features[["feature_num","feature_text"]].to_dict()
features = {fe["feature_num"][i]:fe["feature_text"][j] for i,j in zip(fe["feature_num"],fe["feature_text"])}
CFG.max_len = 466
tokenizer = AutoTokenizer.from_pretrained("./input/token")
CFG.tokenizer = tokenizer



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CustomModel(CFG, config_path=None, pretrained=True)
model_para = torch.load("./microsoft-deberta-base_fold0_best.pth")
model.load_state_dict(model_para["model"])
model.to(device)


"""
API
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def prepare_input(cfg, text, feature_text):
    inputs = cfg.tokenizer(text, feature_text, 
                           add_special_tokens=True,
                           max_length=CFG.max_len,
                           padding="max_length",
                           return_offsets_mapping=False)
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs
from typing import List
from pydantic import BaseModel
def merge(intervals: List[List[int]]) -> List[List[int]]:
	if len(intervals)==0:
		return []
	intervals.sort()
	res = [intervals[0]]
	for i in intervals[1:]:
		if res[-1][1] >= i[0]:
			res[-1][1] = max(i[1],res[-1][1])
		else:
			res.append(i)
	return res
class QueryArgs(BaseModel):
	text: str
	feature_ids: List[int]
@app.post("/query")
async def q(q:QueryArgs):
	feature_ids = q.feature_ids
	notes = q.text
	inputs = {"input_ids":[],"attention_mask":[],"token_type_ids":[]}
	assert len(feature_ids) <=32
	feature_ids = [int(i) for i in feature_ids]
	n = len(feature_ids)
	for feature_id in feature_ids:
			input = prepare_input(CFG,notes,features[feature_id])
			inputs["input_ids"].append(input["input_ids"])
			inputs["attention_mask"].append(input["attention_mask"])
			inputs["token_type_ids"].append(input["token_type_ids"])
	for key in inputs:
		inputs[key] = torch.concat(inputs[key],0).view(n,-1)	
	model.eval()
	for k, v in inputs.items():
		inputs[k] = v.to(device)
	with torch.no_grad():
		y_preds = model(inputs)
		y_preds = y_preds.sigmoid().to('cpu').numpy()
		y_preds = y_preds.reshape((n, CFG.max_len))
	predictions = y_preds.reshape((n, CFG.max_len))
	char_probs = get_char_probs(np.array([notes for i in range(n)]), predictions, CFG.tokenizer)
	results = get_results(char_probs, th=0.5)
	preds = get_predictions(results)	
	pred_content = []
	for pred in preds:
		pred_content.append("\n".join([notes[slice(*i)] for i in pred]))
	new_preds = []
	for i in preds:
		new_preds.extend(i)
	intervals = merge(new_preds)
	return {"result":[{"feature":features[i],"pos":pred,"content":content} for pred,i,content in zip(preds,feature_ids,pred_content)],"intervals":intervals}

@app.get("/feature")
async def fe_list():
	return [{"feature_id":key,"feature_text":features[key]} for key in features]

