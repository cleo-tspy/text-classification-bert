import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import pandas as pd
from datetime import datetime

class SecurityDetectDataset(Dataset):

    # 讀取資料並編碼
    def __init__(self, mode, tokenizer): # 讀取前處理後的檔並初始化一些參數
        assert mode in ["train", "test","inference"]
        self.mode = mode
        self.df = pd.read_csv('./data/'+mode + ".csv").fillna("") # 預測檔案
        self.len = len(self.df)
        self.label_map = {'國安': 0, '非國安': 1}
        self.tokenizer = tokenizer  # 將使用 BERT tokenizer
    
    # 定義回傳一筆訓練 / 測試數據的函式
    def __getitem__(self, idx):

        if self.mode == "inference":
            text_a = self.df.iloc[idx, :].values[0]
            label_tensor = None
        else:
            text_a, label = self.df.iloc[idx, :].values
            # 將 label 文字也轉換成索引方便轉換成 tensor
            label_id = self.label_map[label]
            label_tensor = torch.tensor(label_id)
            
        # 建立第一個句子的 BERT tokens 並加入分隔符號 [SEP]
        word_pieces = ["[CLS]"]
        tokens_a = self.tokenizer.tokenize(text_a)
        word_pieces += tokens_a
        len_a = len(word_pieces)

        
        # 將整個 token 序列轉換成索引序列
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # 將第一句包含 [SEP] 的 token 位置設為 0
        segments_tensor = torch.tensor([0] * len_a,
                                        dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor)
    
    def __len__(self):
        return self.len

        
def create_mini_batch(samples): # 批次整理資料
    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    # 測試集有 labels
    if samples[0][2] is not None:
        label_ids = torch.stack([s[2] for s in samples])
    else:
        label_ids = None
    
    # zero pad 到同一序列長度
    tokens_tensors = pad_sequence(tokens_tensors, 
                                  batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, 
                                    batch_first=True)
    
    # attention masks，將 tokens_tensors 裡頭不為 zero padding
    # 的位置設為 1 讓 BERT 只關注這些位置的 tokens
    masks_tensors = torch.zeros(tokens_tensors.shape, 
                                dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(
        tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

def get_predictions(model, dataloader, compute_acc=False): # 預測
    predictions = None
    correct = 0
    total = 0
      
    with torch.no_grad():
        # 遍巡整個資料集
        for data in tqdm(dataloader):
            # 將所有 tensors 移到 GPU 上 沒有則忽略
            if next(model.parameters()).is_cuda:
                data = [t.to("cuda:0") for t in data if t is not None]
            
            tokens_tensors, segments_tensors, masks_tensors = data[:3]
                
            outputs = model(input_ids=tokens_tensors, 
                            token_type_ids=segments_tensors, 
                            attention_mask=masks_tensors)
            
            logits = outputs[0]
            _, pred = torch.max(logits.data, 1)
            
            # 用來計算訓練集的分類準確率
            if compute_acc:
                labels = data[3]
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
            # 將當前 batch 記錄下來
            if predictions is None:
                predictions = pred
            else:
                predictions = torch.cat((predictions, pred))
    
    if compute_acc:
        acc = correct / total
        return predictions, acc

    return predictions


class Inference():

    def __init__(self) -> None:

#         print('tokenizer loading')
        self.tokenizer = AutoTokenizer.from_pretrained('./model_hf1cbw') # #"hfl/chinese-bert-wwm" 
        # model initialization classification/code
        PRETRAINED_MODEL_NAME = "model_hf1cbw" #"hfl/chinese-bert-wwm" 

        NUM_LABELS = 2
#         print('model loading...')
        model = BertForSequenceClassification.from_pretrained(PRETRAINED_MODEL_NAME, num_labels= NUM_LABELS)
        checkpoint = torch.load('model_security/checkpoint.pth.tar', map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['model_security/state_dict'])
        
        device = torch.device("cpu")
        self.model = model.to(device)
        self.outputpath = "./bert_predict.csv"

    def preprocessing(self):
        pass


    def start(self, mode):

        self.model.eval()
#         print('model loaded')
        # 建立資料集。batch_size可調整
        if mode=='inference':

            infertset = SecurityDetectDataset("inference", tokenizer=self.tokenizer)
            inferenceloader = DataLoader(infertset, batch_size=1, collate_fn=create_mini_batch)
            predictions = get_predictions(self.model, inferenceloader, compute_acc=False)

            # 用來將預測的 label id 轉回 label 文字
            index_map = {v: k for k, v in infertset.label_map.items()}
            # 生成predict 結果檔案
            df = pd.DataFrame({"predict": predictions.tolist()})
            df['predict'] = df.predict.apply(lambda x: index_map[x])

            df_pred = pd.concat([infertset.df.loc[:, ["text"]], 
                                    df.loc[:, 'predict']], axis=1)
            
            df_pred.to_csv(self.outputpath, index=False)

        if mode=='test':
            testset = SecurityDetectDataset("test", tokenizer=self.tokenizer)
            testloader = DataLoader(testset, batch_size=1, collate_fn=create_mini_batch)
            predictions, acc = get_predictions(self.model, testloader, compute_acc=True)

#             print(acc)
            # 用來將預測的 label id 轉回 label 文字
            index_map = {v: k for k, v in testset.label_map.items()}
            # 生成predict 結果檔案
            df = pd.DataFrame({"predict": predictions.tolist()})
            df['predict'] = df.predict.apply(lambda x: index_map[x])
            df_pred = pd.concat([testset.df.loc[:, ["text","label"]], 
                                    df.loc[:, 'predict']], axis=1)

            df_pred.to_csv(self.outputpath, index=False)

if __name__=='__main__':

    inference = Inference()
    inference.start('inference')




