# -*- coding: utf-8 -*-
import os
import logging
import glob
import json
import argparse
import math
import codecs
from tqdm import tqdm, trange
from collections import defaultdict
import numpy as np
import torch
import random
import pickle
import sys
sys.path.append('/data/private/chenyutong/VLP/')
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer, Indexer
from transformers import XLMTokenizer
from pytorch_pretrained_bert.modeling import BertForSeq2SeqDecoder, BertForPreTrainingLossMask
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
import torch
import torch.nn as nn
import torch.nn.functional as F

from vlp.loader_utils import batch_list_to_batch_tensors
import vlp.seq2seq_loader as seq2seq_loader
from vlp.lang_utils import language_eval
from misc.data_parallel import DataParallelImbalance

# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)
# logger = logging.getLogger(__name__)

def detokenize(tk_list):
    r_list = []
    for tk in tk_list:
        if tk.startswith('##') and len(r_list) > 0:
            r_list[-1] = r_list[-1] + tk[2:]
        else:
            r_list.append(tk)
    return r_list


def ascii_print(text):
    text = text.encode("ascii", "ignore")
    print(text)

def postprocess(traces, beam_size, tgt_lang, indexer):
    if beam_size > 1:
        traces = {k: v.tolist() for k, v in traces.items()}
        output_ids = traces['pred_seq']
    else:
        output_ids = traces[0].tolist()

    for ii,w_ids in enumerate(output_ids):
        output_buf = indexer.convert_ids_to_tokens(w_ids)
        output_tokens = []
        for t in output_buf:
            if t in ("[SEP]", "[PAD]"):
                break
            output_tokens.append(t)
            #print(t)
        if tgt_lang=='en':
            output_sequence = ' '.join(detokenize(output_tokens))
            output_sequence = output_sequence.replace(' @ - @ ','-')
        if tgt_lang=='zh':
            output_sequence = ''.join(detokenize(output_tokens)).replace('</w>','').replace('[SEP]','')
    if len(output_sequence)>10:
        output_sequence = output_sequence[:10]+'trunc'
    return output_sequence

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', default='txt', type=str, help='txt -> self-customized')
    # parser.add_argument('--src_lang', default='en', type=str, help='')
    # parser.add_argument('--tgt_lang', default='zh', type=str, help='')
    parser.add_argument('--max_len_en', default=25, type=int, help='maximum length of English in **bilingual** corpus')
    parser.add_argument('--max_len_zh', default=25, type=int, help='maximum length of Chinese in **bilingual** corpus')
    parser.add_argument("--src_file", default='./.pkl', type=str, help="The input data file name.")

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
    parser.add_argument('--len_vis_input', type=int, default=1, help="The length of visual token input region 1")


    with open('/data/private/chenyutong/dataset/concept_count/word_concept_count.pkl','rb') as f:
        word_fre = pickle.load(f)
    word_fre = defaultdict(int, word_fre)

    args = parser.parse_args()

    assert args.batch_size==1, 'only support batch_size=1'
    args.max_tgt_length = max(args.max_len_en,args.max_len_zh)

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
    #tokenizer_en= WhitespaceTokenizer() if args.tokenized_input else tokenizer_en
    tokenizer_zh = XLMTokenizer(args.xml_vocab, args.xml_merge)
    tokenizer_zh.tokenize = lambda x: tokenizer_zh._tokenize(x, lang='zh', bypass_tokenizer=False)
    with open(args.xml_vocab,'r') as f:
        tokenizer_zh.vocab = json.load(f)
    indexer = Indexer([os.path.join(args.bert_model,'vocab.txt'), args.xml_vocab])
    with open('full_vocab.json','w') as f:
        json.dump(indexer.ids_to_tokens, f)
    tokenizers = {'en':tokenizer_en, 'zh':tokenizer_zh}
    print('tokenizer created')

    assert '.pkl' in args.src_file
    with open(args.src_file,'rb') as f:
        src_data = pickle.load(f) 
    # list [pred_id, vocab, vis, pos, distribution]
    # dict {'vgid':{'en':,'zh':,'region_features':[img, conf, fea[i], pos[i],dist]}}
    amp_handle = None
    if args.amp:
        from apex import amp

    # Prepare model
    cls_num_labels = 2
    type_vocab_size = 12 if args.new_segment_ids else 12
    mask_word_id, eos_word_ids = indexer(
        ["[MASK]", "[SEP]"])
    forbid_ignore_set = None #default None
    relax_projection, task_idx_proj = 0, 3
    if args.forbid_ignore_word:
        w_list = []
        for w in args.forbid_ignore_word.split('|'):
            if w.startswith('[') and w.endswith(']'):
                w_list.append(w.upper())
            else:
                w_list.append(w)
        forbid_ignore_set = set(indexer(w_list))    

    print(args.model_recover_path)
    for model_recover_path in glob.glob(args.model_recover_path.strip()):
        #logger.info("***** Recover model: %s *****", model_recover_path)
        model_recover = torch.load(model_recover_path)
        model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model,
            max_position_embeddings=args.max_position_embeddings, config_path=args.config_path,
            state_dict=model_recover, num_labels=cls_num_labels,
            vocab_size=len(indexer),
            type_vocab_size=type_vocab_size, task_idx=3, mask_word_id=mask_word_id, #img2txt
            search_beam_size=args.beam_size, length_penalty=args.length_penalty,
            eos_id=eos_word_ids, forbid_duplicate_ngrams=args.forbid_duplicate_ngrams,
            forbid_ignore_set=forbid_ignore_set, ngram_size=args.ngram_size, min_len=args.min_len,
            enable_butd=args.enable_butd, len_vis_input=args.len_vis_input)

        del model_recover

        model.to(device)

        if args.amp:
            model = amp.initialize(model, opt_level='O2')#'02')
        torch.cuda.empty_cache()
        model.eval()

        fout = open(os.path.join(args.output_dir,'region2txt_output.txt'),'w')
        output_lines = []
        select_ids = [87, 120,179, 297,721,852,1025]
        for step_val, sd in enumerate(src_data.items()):
            # if step_val>=1:
            #     break
            vgid, input_item = sd
            en, zh = input_item['en'], input_item['zh']
            fout.writelines('\n'+'#'*10+'\n')
            fout.writelines('{}\n'.format(vgid))
            fout.writelines('{} coco: word_fre {}  vis_fre {} \n'.format(en, input_item['coco_fre']['word'], input_item['coco_fre']['vis']))
            fout.writelines('{} aic: word_fre {}  vis_fre {} \n'.format(zh, input_item['aic_fre']['word'], input_item['aic_fre']['vis']))
            print('step_val {} Process {}'.format(step_val, en))            
            for rf in tqdm(input_item['region_features']):
                filename, conf, vis_feats, vis_pe, cls_label = rf
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
                assert args.new_segment_ids==False, 'only support 0 1 6 now'
                tokens = ['[CLS]','[UNK]','[SEP]']
                input_ids = indexer(tokens)
                input_ids = np.expand_dims(np.array(input_ids), axis=0)
                input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)


                max_len_in_batch = len(tokens)+args.max_tgt_length
                _tril_matrix = torch.tril(torch.ones(
                            (max_len_in_batch, max_len_in_batch), dtype=torch.long))
                input_mask = torch.zeros(
                    max_len_in_batch, max_len_in_batch, dtype=torch.long, device=device)
                input_mask[:, :len(tokens)].fill_(1)
                second_st, second_end = len(tokens), max_len_in_batch
                input_mask[second_st:second_end, second_st:second_end].copy_(
                    _tril_matrix[:second_end-second_st, :second_end-second_st]) #L,L
                input_mask = input_mask.unsqueeze(0)

                position_ids = torch.arange(
                    max_len_in_batch, dtype=torch.long, device=device) #L
                position_ids = position_ids.unsqueeze(0) # B,L

                predictions = {'en':None, 'zh': None, 'en2zh':None, 'zh2en':None}
                for tgt_lang, lang_id in zip(['en','zh'],[1,6]):
                    token_type_ids = [0]*len(tokens)+[lang_id]*args.max_tgt_length
                    token_type_ids = np.expand_dims(np.array(token_type_ids), axis=0)
                    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device)            
                    with torch.no_grad():  
                        # print(token_type_ids[0])
                        # print(position_ids[0])
                        # print(input_ids[0])
                        # print(input_mask[0])
                        # input()
                        traces = model(vis_feats=vis_feats, vis_pe=vis_pe, 
                            input_ids=input_ids, token_type_ids=token_type_ids, 
                            position_ids=position_ids, attention_mask=input_mask, 
                            search_beam_size=args.beam_size, task_idx=3, 
                            mode='img2txt',
                            sample_mode='greedy') #validation greedy

                    output_sequence = postprocess(traces, args.beam_size, tgt_lang, indexer)
                    predictions[tgt_lang] = output_sequence #truncate

                for langs, lang_ids in zip(['en2zh','zh2en'],[[1,6],[6,1]]):
                    src_lang = langs[:2] #en,zh
                    tgt_lang = langs[-2:]
                    w = predictions[src_lang] # predictions['en']/ predictions['zh']
                    w_t = tokenizers[src_lang].tokenize(w)
                    tokens = ['[CLS]']+w_t+['[SEP]']
                    input_ids = indexer(tokens)
                    token_type_ids = [lang_ids[0]]*len(input_ids) + [lang_ids[1]]*args.max_tgt_length
                    input_ids = np.expand_dims(np.array(input_ids), axis=0)
                    token_type_ids = np.expand_dims(np.array(token_type_ids), axis=0)
                    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
                    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long, device=device) 

                    max_len_in_batch = len(tokens)+args.max_tgt_length #2+64 = 66
                    position_ids = torch.arange(
                        max_len_in_batch, dtype=torch.long, device=device) #L
                    position_ids = position_ids.unsqueeze(0) # B,L      
                    _tril_matrix = torch.tril(torch.ones(
                                (max_len_in_batch, max_len_in_batch), dtype=torch.long))
                    input_mask = torch.zeros(
                        max_len_in_batch, max_len_in_batch, dtype=torch.long, device=device)
                    input_mask[:, :len(tokens)].fill_(1)
                    second_st, second_end = len(tokens), max_len_in_batch
                    input_mask[second_st:second_end, second_st:second_end].copy_(
                        _tril_matrix[:second_end-second_st, :second_end-second_st]) #L,L
                    input_mask = input_mask.unsqueeze(0)
                    with torch.no_grad(): 
                        traces = model(vis_feats=None, vis_pe=None, 
                            input_ids=input_ids, token_type_ids=token_type_ids, 
                            position_ids=position_ids, attention_mask=input_mask, 
                            search_beam_size=args.beam_size, task_idx=3, 
                            mode='txt2txt',
                            sample_mode='greedy') #validation greedy
                    output_sequence = postprocess(traces, args.beam_size, tgt_lang, indexer)
                    predictions[langs] = output_sequence

                #print(predictions)
                fout.writelines('conf:{:.2f} en:{: <10} fre:{:<5d} en2zh:{: <10} zh:{: <10} fre:{:<5d} zh2en:{: <10} \n'.format(
                    conf, 
                    predictions['en'], 
                    word_fre['coco'][predictions['en']], 
                    predictions['en2zh'],
                    predictions['zh'], 
                    word_fre['aic'][predictions['zh']], 
                    predictions['zh2en']))

        fout.close()


                




if __name__ == "__main__":
    main()