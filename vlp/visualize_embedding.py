# -*- coding: utf-8 -*-
import os
import logging
import glob
import json
import argparse
import math
import codecs
from collections import OrderedDict
from tqdm import tqdm, trange
import numpy as np
import torch
import random
import pickle
import sys
sys.path.append('/data/private/chenyutong/VLP/')
sys.path.append('/data/private/chenyutong/XEmbedding/')
import matplotlib 
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from DisplayEmbedding import Cosine_Sim,sim_histgram,tSNE_reduce,save_numpy
import sklearn
from sklearn.datasets import fetch_mldata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer, Indexer
from transformers import XLMTokenizer
from pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder, BertForPreTrainingLossMask
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear


from vlp.loader_utils import batch_list_to_batch_tensors
import vlp.seq2seq_loader as seq2seq_loader
from vlp.lang_utils import language_eval
from misc.data_parallel import DataParallelImbalance



def tSNE_reduce_visual(concept_list, embeddings, title):
    X = []
    num_vis = len(embeddings['vis'][0])
    num_concept = len(embeddings['en'])
    for lang in embeddings:
        assert len(embeddings[lang])==num_concept
        if lang in ['en','zh']:
            X.extend(embeddings[lang])
        else:
            for ls in embeddings[lang]:
                assert len(ls)==num_vis, (num_vis, len(ls))
                X.extend(ls)

    pca = PCA(n_components=10)
    pca_result = pca.fit_transform(X)
    tsne = TSNE(n_components=2, metric='cosine')
    tsne_results = tsne.fit_transform(X)
    df = {}
    df['tsne-one'] = tsne_results[:,0]
    df['tsne-two'] = tsne_results[:,1] 

    fig, ax = plt.subplots()

    import matplotlib.lines as mlines
    cmap = plt.cm.tab20
    for i in range(num_concept):
        # concept_handles.append(mlines.Line2D([], [], c=cmap(i/num_concept),marker='o', linestyle='None',
        #                   markersize=10, label=concept_list[i]))
        en_embed = tsne_results[i,:]
        ax.scatter(
            x=[en_embed[0]], y=[en_embed[1]], 
            c=[i/num_concept], cmap='tab20', vmin=0, vmax=1, alpha=0.9,edgecolors='black',
            marker='o',s=70)

        zh_embed = tsne_results[i+num_concept,:]
        ax.scatter(
            x=[zh_embed[0]], y=[zh_embed[1]], 
            c=[i/num_concept], cmap='tab20', vmin=0, vmax=1, alpha=0.9,edgecolors='black',
            marker='^',s=70)

        vis_embed = tsne_results[2*num_concept+i*num_vis:2*num_concept+(i+1)*num_vis,:]
        ax.scatter(
            x=vis_embed[:,0], y=vis_embed[:,1],
            c=[i/num_concept]*num_vis, cmap='tab20', vmin=0, vmax=1, 
            marker='+')

        cx,cy = np.mean(vis_embed[:,0]), np.mean(vis_embed[:,1])
        ax.text(cx, cy, concept_list[i]) 

    en = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                              markersize=10, label='En')
    zh = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                              markersize=10, label='Zh')
    vis = mlines.Line2D([], [], color='black', marker='+', linestyle='None',
                              markersize=10, label='Vis')

    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0 + box.height * 0.3,
    #                  box.width, box.height * 0.7])
    ax.legend(handles=[en,zh,vis])
    ax.set_xlabel('tsne-one')
    ax.set_ylabel('tsne-two')
    ax.set_title(title.split('/')[-1])
    fig.savefig('{}.png'.format(title))
    plt.close(fig)

    concept_handles = []
    fig, ax = plt.subplots()
    for i in range(num_concept):
        concept_handles.append(mlines.Line2D([], [], c=cmap(i/num_concept),marker='o', linestyle='None',
                          markersize=10, label=concept_list[i]))
        en_embed = tsne_results[i,:]
        ax.scatter(
            x=[en_embed[0]], y=[en_embed[1]], 
            c=[i/num_concept], cmap='tab20', vmin=0, vmax=1, alpha=0.9,edgecolors='black',
            marker='o',s=70)
        zh_embed = tsne_results[i+num_concept,:]
        ax.scatter(
            x=[zh_embed[0]], y=[zh_embed[1]], 
            c=[i/num_concept], cmap='tab20', vmin=0, vmax=1, alpha=0.9,edgecolors='black',
            marker='^',s=70)

    en = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                              markersize=10, label='En')
    zh = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                              markersize=10, label='Zh')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.3,
                     box.width, box.height * 0.7])
    ax.legend(handles=concept_handles+[en,zh], bbox_to_anchor=(0.5, 0),ncol=4,loc='upper center')
    ax.set_xlabel('tsne-one')
    ax.set_ylabel('tsne-two')
    ax.set_title(title.split('/')[-1])
    fig.savefig('{}_word.png'.format(title))
    plt.close(fig)

    return 



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--enable_vis', action='store_true',help='whether to visualize visual embedding')
    parser.add_argument('--num_concept', default='-1', type=int, help='number of concepts to visualize')
    parser.add_argument('--num_vis', default='1', type=int, help='number of visual embeddings per concept')
    parser.add_argument('--dataset', default='txt', type=str, help='txt -> self-customized')
    parser.add_argument('--src_lang', default='en', type=str, help='')
    parser.add_argument('--tgt_lang', default='zh', type=str, help='')
    parser.add_argument('--max_len_en', default=25, type=int, help='maximum length of English in **bilingual** corpus')
    parser.add_argument('--max_len_zh', default=25, type=int, help='maximum length of Chinese in **bilingual** corpus')
    #parser.add_argument("--vocab_file", default='./src.txt', type=str, help="The input data file name.")
    parser.add_argument('--vocab_file',  type=str, required=True, nargs = '+')
    parser.add_argument('--en_first', action='store_true',help='always to put english as the first sentence')

    # General
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased.")
    parser.add_argument("--xml_vocab",type=str, default='./download_models/xml_vocab.json')
    parser.add_argument("--xml_merge",type=str, default='./download_models/xml_merges.txt')
    parser.add_argument("--model_recover_path", default=None, type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument('--max_position_embeddings', type=int, default=512,
                        help="max position embeddings")

    # For decoding
    #parser.add_argument('--fp16', action='store_true',
     #                   help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--seed', type=int, default=123,
                        help="random seed for initialization")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="Batch size for decoding.")
    parser.add_argument('--beam_size', type=int, default=1,
                        help="Beam size for searching")
    parser.add_argument('--length_penalty', type=float, default=0,
                        help="Length penalty for beam search")

    parser.add_argument('--forbid_duplicate_ngrams', action='store_true')
    parser.add_argument('--forbid_ignore_word', type=str, default=None,
                        help="Forbid the word during forbid_duplicate_ngrams")
    parser.add_argument("--min_len", default=None, type=int)
    parser.add_argument('--ngram_size', type=int, default=3)


    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--enable_butd', action='store_true',
                        help='set to take in region features')
    parser.add_argument('--output_dir', default='./result', type=str)

    #useless
    parser.add_argument('--split', type=str, default='val') #wmt?
    parser.add_argument('--len_vis_input', type=int, default=100, help="The length of visual token input")



    args = parser.parse_args()
    print(args.vocab_file)
    if args.enable_vis or '.pkl' in args.vocab_file[0]:
        args.vocab_file = args.vocab_file[0]
        assert '.pkl' in args.vocab_file, args.vocab_file

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    tokenizer_en = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case,
        cache_dir=args.output_dir+'/.pretrained_model')
    if args.max_position_embeddings:
        tokenizer_en.max_len = args.max_position_embeddings
    tokenizer_zh = XLMTokenizer(args.xml_vocab, args.xml_merge)
    tokenizer_zh.tokenize = lambda x: tokenizer_zh._tokenize(x, lang='zh', bypass_tokenizer=False)
    with open(args.xml_vocab,'r') as f:
        tokenizer_zh.vocab = json.load(f)
    indexer = Indexer([os.path.join(args.bert_model,'vocab.txt'), args.xml_vocab])
    with open('full_vocab.json','w') as f:
        json.dump(indexer.ids_to_tokens, f)
    tokenizers = {'en':tokenizer_en, 'zh':tokenizer_zh}
    print('tokenizer created')


    concept_list = []
    if args.enable_vis or 'pkl' in args.vocab_file:
        with open(args.vocab_file,'rb') as f:
            vocab_list = pickle.load(f)
        vocab_list = vocab_list[::-1]
        if args.num_concept==-1:
            args.num_concept = len(vocab_list)
        vocab = [] #[ [En, Zh, Vis1(list), Vis2, Vis3, Vis4,..] *num_concept]
        for key, ls in vocab_list[40:40+args.num_concept]:
            concept = [ls[0][0], ls[0][1]] #En, Zh
            concept_list.append(ls[0][0])
            for inst in ls[:args.num_vis]:
                concept.append(inst[2:]) #2,3,4
            vocab.append(concept)
        print('Number of Concept {}'.format(len(vocab)))
        if args.num_concept!=-1:
            print(concept_list)
        print('Number of visual instance per concept {}'.format(len(vocab[0])-2))
        #print('Example {}'.format(vocab[0]))
        #input()
    else:
        vocab = []
        for filename in args.vocab_file:
            with open(filename) as f:
                v = f.readlines()
            if args.num_concept==-1:
                args.num_concept = len(v)
            v = [(a.split('    ')[0].strip(), a.split('    ')[-1].strip()) for a in v[:args.num_concept]] #EN ZH 
            vocab.extend(v)
        print('Number of vocabulary {} {}'.format(len(vocab),vocab[0]))

    cls_num_labels = 2
    type_vocab_size = 12 if args.new_segment_ids else 12
    mask_word_id, eos_word_ids = indexer(
        ["[MASK]", "[SEP]"])
    forbid_ignore_set = None #default None
    relax_projection, task_idx_proj = 0, 3
    print(args.model_recover_path)
    model_recover = torch.load(args.model_recover_path)
    model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model,
        max_position_embeddings=args.max_position_embeddings, config_path=args.config_path,
        state_dict=model_recover, num_labels=2,
        vocab_size=len(indexer),
        type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id, #img2txt
        search_beam_size=args.beam_size, length_penalty=args.length_penalty,
        eos_id=eos_word_ids, forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
        forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size, min_len=args.min_len,
        enable_butd=True, len_vis_input=args.len_vis_input)
    del model_recover
    model.to(device)
    torch.cuda.empty_cache()
    model.eval()

    N_layer = 12
    embeddings = OrderedDict({'en': [[] for i in range(N_layer)],'zh':[[] for i in range(N_layer)]})
    if args.enable_vis:
        embeddings['vis'] = [[] for i in range(N_layer)]
    for _,pair in tqdm(enumerate(vocab)):
        for w, lang in zip(pair, ('en','zh')):
            segment_id = 1 if lang=='en' else 6
            w_t = tokenizers[lang].tokenize(w)

            tokens = ['[CLS]']+w_t+['[SEP]']
            input_ids = indexer(tokens)
            token_type_ids = [segment_id]*len(input_ids)
            input_ids = np.expand_dims(np.array(input_ids), axis=0)
            token_type_ids = np.expand_dims(np.array(token_type_ids), axis=0)
            input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
            token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)

            output_embeddings = model.compute_embeddings(input_ids=input_ids, token_type_ids=token_type_ids, mode='txt2txt')
            #tuple 12 1,L,768
            for i,e in enumerate(output_embeddings):
                e = e.detach().cpu().numpy()
                ave = np.mean(e[0,1:-1,:], axis=0) # 768
                embeddings[lang][i].append(ave)

        if args.enable_vis:
            instance_embeddings = [[] for layer_i in range(N_layer)]
            for vis_embed in pair[2:]:
                vis_feats, vis_pe, cls_label = vis_embed[0], vis_embed[1], vis_embed[2] #1024
                vis_feats = torch.from_numpy(vis_feats).to(device)
                vis_feats = vis_feats.unsqueeze(0) 
                vis_pe = torch.from_numpy(vis_pe).to(device)
                vis_pe = vis_pe.unsqueeze(0) 
                cls_label = torch.from_numpy(cls_label).to(device)
                cls_label = cls_label.unsqueeze(0) #
                # lazy normalization of the coordinates... copy from seq2seq
                w_est = torch.max(vis_pe[:, [0, 2]])*1.+1e-5
                h_est = torch.max(vis_pe[:, [1, 3]])*1.+1e-5
                vis_pe[:, [0, 2]] /= w_est
                vis_pe[:, [1, 3]] /= h_est
                assert h_est > 0, 'should greater than 0! {}'.format(h_est)
                assert w_est > 0, 'should greater than 0! {}'.format(w_est)
                rel_area = (vis_pe[:, 3]-vis_pe[:, 1])*(vis_pe[:, 2]-vis_pe[:, 0])
                rel_area.clamp_(0)

                vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), vis_pe[:, 5:]), -1) # confident score
                normalized_coord = F.normalize(vis_pe.data[:, :5]-0.5, dim=-1)
                vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), \
                    F.layer_norm(cls_label, [1601])), dim=-1) # 1601 hard coded... #BL,H

                vis_feats = vis_feats.unsqueeze(0)
                vis_pe = vis_pe.unsqueeze(0)
                #print('input shape', vis_feats.shape, vis_pe.shape)
                segment_id = 0
                tokens = ['[CLS]','[UNK]','[SEP]']
                input_ids = indexer(tokens)
                token_type_ids = [segment_id]*len(input_ids)
                input_ids = np.expand_dims(np.array(input_ids), axis=0)
                token_type_ids = np.expand_dims(np.array(token_type_ids), axis=0)
                input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
                token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)


                vis_embeddings = model.compute_embeddings(vis_feats=vis_feats, vis_pe=vis_pe, 
                    input_ids=input_ids, token_type_ids=token_type_ids, mode='img2txt', len_vis_input=1)
                #print(len(vis_embeddings), vis_embeddings[0].shape)
                #input()

                for i,e in enumerate(vis_embeddings):
                    e = e.detach().cpu().numpy()
                    ave = np.mean(e[0,1:-1,:], axis=0) # 768
                    instance_embeddings[i].append(ave)
                    #embeddings['vis'][i].append(ave)

            for i, embed_list in enumerate(instance_embeddings):
                if args.num_vis==1:
                    embeddings['vis'][i].append(embed_list[0])
                else:
                    embeddings['vis'][i].append(embed_list) #list of array
    # args.output_dir = os.path.join(args.output_dir, 'embedding_vis')
    # if not os.path.exists(args.output_dir):
    #     os.makedirs(args.output_dir)


    for ly in range(N_layer):
        embed={'en':embeddings['en'][ly],'zh':embeddings['zh'][ly]}
        if args.enable_vis:
            embed['vis'] = embeddings['vis'][ly]

        #save_numpy(embed,os.path.join(args.output_dir,'hiddenstates_layer_{}.npy'.format(ly)))
        if args.num_vis==1:
            tSNE_reduce(embed, os.path.join(args.output_dir,'tSNE layer {}'.format(ly)))
            Cosine_Sim(embed, os.path.join(args.output_dir,'CosineSim Layer {}'.format(ly)))
            sim_histgram(embed, os.path.join(args.output_dir,'Histgram Layer {}'.format(ly)))
        if args.num_vis>1:
            tSNE_reduce_visual(concept_list, embed, os.path.join(args.output_dir,'tSNE layer {}'.format(ly)))
        print('Save layer {}'.format(ly))



if __name__ == "__main__":
    main()
