from Pruning.hg_transformers.configuration_auto import AutoConfig
from Pruning.hg_transformers.modeling_auto import AutoModelForSequenceClassification
from Pruning.hg_transformers.tokenization_auto import AutoTokenizer
import os

bert_path = 'models/bert_pt'
config = AutoConfig.from_pretrained('bert-base-uncased')

print('Downloading/Loading pre-trained BERT...')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', config=config)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

if not os.path.exists(bert_path):
    os.makedirs(bert_path)
    
print('Saving pre-trained BERT to %s'%bert_path)
model.save_pretrained(bert_path)
tokenizer.save_pretrained(bert_path)
