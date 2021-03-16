import argparse, logging, os, sys, torch
from typing import Dict, Optional

import numpy as np
import torch.nn.utils.prune as prune

from transformer.configuration_bert_prun import BertConfigPrun
from transformer.modeling_prun import TinyBertForSequenceClassification as PrunTinyBertForSequenceClassification
from transformer.tokenization import BertTokenizer


logger = logging.getLogger(__name__)

intermedia_modules, attn_modules = [], []
for i in range(12):
    intermedia_modules += ['bert.encoder.layer.%d.intermediate.dense'%i, 'bert.encoder.layer.%d.output.dense'%i]
    attn_modules += ['bert.encoder.layer.%d.attention.self.query'%i, 'bert.encoder.layer.%d.attention.self.key'%i,\
                    'bert.encoder.layer.%d.attention.self.value'%i, 'bert.encoder.layer.%d.attention.output.dense'%i]

# structured pruning prunes input or output neurons of a matrix
prune_out, prune_in = [], []
for i in range(12):
    prune_out += ['bert.encoder.layer.%d.attention.self.query'%i, 'bert.encoder.layer.%d.attention.self.key'%i, \
                     'bert.encoder.layer.%d.attention.self.value'%i,'bert.encoder.layer.%d.intermediate.dense'%i]
    prune_in += ['bert.encoder.layer.%d.attention.output.dense'%i, 'bert.encoder.layer.%d.output.dense'%i] 

class PruningMethod(prune.BasePruningMethod):
    """Prune every other entry in a tensor
    """
    PRUNING_TYPE = 'unstructured'
    def __init__(self, prun_ratio, score):
        self.prun_ratio = prun_ratio
        self.score = score

    def compute_mask(self, t, default_mask):
        tensor_size = t.nelement()
        nparams_toprune = prune._compute_nparams_toprune(self.prun_ratio, tensor_size)
        mask = default_mask.clone()
        topk = torch.topk(
            torch.abs(self.score).view(-1), k=nparams_toprune, largest=False
        )
        # topk will have .indices and .values
        mask.view(-1)[topk.indices] = 0
        return mask


def unstructured_pruning(module, name, prun_ratio, score):
    PruningMethod.apply(module, name, prun_ratio, score)
    return module


def Taylor_pruning_structured(model, prun_ratio, num_heads, keep_heads, taylor_path, 
                            emb_hidden_dim, config):
    """
    Args:
        emb_hidden_dim[int]: Hidden size of embedding factorization. Choose from 128, 256, 512.
                                Do not factorize embedding if value==-1
    """
    #Counting scores
    taylor_dict = torch.load(taylor_path)
    intermedia_scores, attn_scores = [], []
    for i in range(len(model.bert.encoder.layer)):
        score_inter_in = taylor_dict['encoder.layer.%d.intermediate.dense.weight'%i]
        score_inter_out = taylor_dict['encoder.layer.%d.output.dense.weight'%i]
        score_inter = score_inter_in.sum(1) + score_inter_out.sum(0)
        intermedia_scores.append(score_inter)

        score_attn_output = taylor_dict['encoder.layer.%d.attention.output.dense.weight'%i]
        score_attn = score_attn_output.sum(0)
        attn_score_chunks = torch.split(score_attn, 64)
        score_attn = torch.tensor([chunk.sum() for chunk in attn_score_chunks])
        attn_scores.append(score_attn)

    with torch.no_grad():
        #Factorizing Embedding Matrix
        if emb_hidden_dim!=-1:
            if hasattr(model.bert.embeddings, 'word_embeddings'):
                print('Factorizing embedding matrix...')
                emb = model.bert.embeddings.word_embeddings.weight.data.numpy()
                u, s, v = np.linalg.svd(emb)
                s = np.eye(emb.shape[1])*s
                temp = np.dot(u[:, :emb_hidden_dim], s[:emb_hidden_dim, :emb_hidden_dim])
                new_word_emb1 = torch.from_numpy(temp)
                new_word_emb2 = torch.from_numpy(v[:emb_hidden_dim])
                model.bert.embeddings.word_embeddings1 = torch.nn.Embedding(config.vocab_size, emb_hidden_dim, padding_idx=0)
                model.bert.embeddings.word_embeddings1.weight.data = new_word_emb1.clone()
                model.bert.embeddings.word_embeddings2 = torch.nn.Linear(emb_hidden_dim, config.hidden_size, bias=False)
                model.bert.embeddings.word_embeddings2.weight.data = new_word_emb2.t().clone()
                del model.bert.embeddings.word_embeddings
            else:
                model.bert.embeddings.word_embeddings1.weight.data = model.bert.embeddings.word_embeddings1.weight.data[:, :emb_hidden_dim]
                model.bert.embeddings.word_embeddings2.weight.data = model.bert.embeddings.word_embeddings2.weight.data[:, :emb_hidden_dim]

        layer_id = 0
        for name, module in model.named_modules():
            #Pruning Attention Heads
            if (name in attn_modules) and (not keep_heads==model.config.num_attention_heads):
                score_attn = attn_scores[layer_id]
                attn_size = module.weight.size(0)/float(num_heads) if name in prune_out \
                                            else module.weight.size(1)/float(num_heads)
                _, indices = torch.topk(score_attn, keep_heads)
                if name in prune_out:
                    weight_chunks = torch.split(module.weight.data, int(attn_size), dim=0)
                    bias_chunks = torch.split(module.bias.data, int(attn_size))
                    module.bias.data = torch.cat([bias_chunks[i] for i in indices])
                    module.weight.data = torch.cat([weight_chunks[i] for i in indices], dim=0)
                elif name in prune_in:
                    weight_chunks = torch.split(module.weight.data, int(attn_size), dim=1)
                    module.weight.data = torch.cat([weight_chunks[i] for i in indices], dim=1)

            #Pruning FFN
            if (name in intermedia_modules) and (not prun_ratio==1):
                score_inter = intermedia_scores[layer_id]
                expand_score = score_inter.expand(config.hidden_size, score_inter.size(0))
                if name in prune_out:
                    unstructured_pruning(module, 'bias', 1-prun_ratio, score_inter)
                    unstructured_pruning(module, 'weight', 1-prun_ratio, expand_score.t())
                    prune.remove(module, 'bias')
                    prune.remove(module, 'weight')
                    module.bias.data = module.bias.data.masked_select(module.bias.data!=0)
                    module.weight.data = module.weight.data.masked_select(module.weight.data!=0).view(-1, config.hidden_size)
                elif name in prune_in:
                    unstructured_pruning(module, 'weight', 1-prun_ratio, expand_score)
                    prune.remove(module, 'weight')
                    module.weight.data = module.weight.data.masked_select(module.weight.data!=0).view(config.hidden_size, -1)
                    layer_id += 1
    return model


num_labels = {
        "cola": 2,
        "sst-2": 2,
        "qnli": 2,
        "qqp": 2,
        "mnli": 3
        }

def main():
    parser = argparse.ArgumentParser(description='pruning_one-step.py')
    parser.add_argument('-model_path', default='../KD/models/bert_ft', type=str,
                        help="distill type")
    parser.add_argument('-output_dir', default='models/prun_bert', type=str,
                        help="output dir")
    parser.add_argument('-task', default='CoLA', type=str,
                        help="Name of the task")
    parser.add_argument('-keep_heads', type=int, default=2,
                    help="the number of attention heads to keep")
    parser.add_argument('-ffn_hidden_dim', type=int, default=512,
                    help="Hidden size of the FFN subnetworks.")
    parser.add_argument('-num_layers', type=int, default=8,
                    help="the number of layers of the pruned model")
    parser.add_argument('-emb_hidden_dim', type=int, default=128,
                    help="Hidden size of embedding factorization. \
                    Do not factorize embedding if value==-1")
    args = parser.parse_args()

    torch.manual_seed(0)

    args.model_path = os.path.join(args.model_path, args.task)
    args.output_dir = os.path.join(args.output_dir, args.task)

    print('Loading BERT from %s...'%args.model_path)
    model = PrunTinyBertForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=num_labels[args.task.lower()]
    )
    config = model.config
    tokenizer = BertTokenizer.from_pretrained(args.model_path, do_lower_case=True)
    model.bert.encoder.layer = torch.nn.ModuleList([model.bert.encoder.layer[i] for i in range(args.num_layers)])

    if args.ffn_hidden_dim>config.prun_intermediate_size or \
    (args.emb_hidden_dim>config.emb_hidden_dim and config.emb_hidden_dim!=-1):
        raise ValueError('Cannot prune the model to a larger size!')

    args.prun_ratio = args.ffn_hidden_dim/config.prun_intermediate_size
    print('Pruning to %d heads, %d layers, %d FFN hidden dim, %d emb hidden dim...'%
            (args.keep_heads, args.num_layers, args.ffn_hidden_dim, args.emb_hidden_dim))
    importance_dir = os.path.join(args.model_path, 'taylor_score', 'taylor.pkl')
    new_config = BertConfigPrun(num_attention_heads = args.keep_heads,
                            prun_hidden_size = int(args.keep_heads*64),
                            prun_intermediate_size = args.ffn_hidden_dim,
                            num_hidden_layers = args.num_layers,
                            emb_hidden_dim = args.emb_hidden_dim)
    model = Taylor_pruning_structured(model, args.prun_ratio, config.num_attention_heads, 
                                            args.keep_heads, importance_dir, 
                                            args.emb_hidden_dim, new_config)

    output_dir = os.path.join(args.output_dir, 'a%d_l%d_f%d_e%d'
            %(args.keep_heads, args.num_layers, args.ffn_hidden_dim, args.emb_hidden_dim))
    
    print('Saving model to %s'%output_dir) 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    new_config.save_pretrained(output_dir)
    tokenizer.save_vocabulary(output_dir)
    model = PrunTinyBertForSequenceClassification.from_pretrained(output_dir, num_labels=num_labels[args.task.lower()])
    torch.save(model.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
    print("Number of parameters: %d"%sum([model.state_dict()[key].nelement() for key in model.state_dict()]))
    print(model.state_dict().keys())


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
