# -*- coding: utf-8 -*-
import os
import logging
import glob
import json
import argparse
import math
import codecs
from tqdm import tqdm, trange
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='txt', type=str, help='txt -> self-customized')
    parser.add_argument('--src_lang', default='en', type=str, help='')
    parser.add_argument('--tgt_lang', default='zh', type=str, help='')
    parser.add_argument('--max_len_en', default=25, type=int, help='maximum length of English in **bilingual** corpus')
    parser.add_argument('--max_len_zh', default=25, type=int, help='maximum length of Chinese in **bilingual** corpus')
    parser.add_argument("--src_file", default='./src.txt', type=str, help="The input data file name.")
    parser.add_argument('--corpus', default='txt', type=str)
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
    args.tgt_lang = 'en' if args.src_lang=='zh' else 'zh'
    # print(sys.getfilesystemencoding())
    # print('这是中文')
    assert args.batch_size==1, 'only support batch_size=1'
    args.max_tgt_length = args.max_len_en if args.src_lang=='zh' else args.max_len_zh

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

    if '.txt' in args.src_file:
        with codecs.open(args.src_file, 'r') as f:
            src_lines = f.readlines()
        src_lines = [line.strip() for line in src_lines] 
        N_lines = len(src_lines) 
    elif 'hdf5' in args.src_file:
        assert 'wmt'==args.corpus   
        src_lines = args.src_file 
        N_lines = 1999
    else:
        raise

    pipeline = seq2seq_loader.Preprocess4Seq2SeqBilingualDecoder(
        corpus=args.corpus,file_src=src_lines, src_lang=args.src_lang, 
        indexer=indexer, tokenizers=tokenizers, 
        max_len=args.max_len_en+args.max_len_zh+3, max_tgt_length=args.max_tgt_length, 
        preprocessed=False if args.corpus=='txt' else True, new_segment_ids=args.new_segment_ids, mode='s2s')

    eval_dataset = seq2seq_loader.Txt2txtDataset(N_lines=N_lines, split=None,
        batch_size=args.batch_size, tokenizers=tokenizers, max_len=args.max_len_en+args.max_len_zh+3,
        preprocessed=False if args.corpus=='txt' else True, bi_uni_pipeline=[pipeline], s2s_prob=1, bi_prob=0)

    eval_dataloader = torch.utils.data.DataLoader(
                eval_dataset, batch_size=args.batch_size,
                sampler=SequentialSampler(eval_dataset), num_workers=4, 
                collate_fn=batch_list_to_batch_tensors, pin_memory=True)

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
        val_iter_bar = tqdm(eval_dataloader)
        output_lines = []
        for step_val, val_iter_output in enumerate(val_iter_bar):
            info_, batch = val_iter_output[0],val_iter_output[1]
            with torch.no_grad():
                batch = [t.to(device) for t in batch]
                input_ids, segment_ids, position_ids, input_mask, task_idx = batch    
                traces = model(None, None, input_ids, segment_ids, position_ids, input_mask, 
                    search_beam_size=args.beam_size, task_idx=task_idx, 
                    mode='txt2txt',
                    sample_mode='greedy') #validation greedy
                # if step_val==0:
                #     print(segment_ids[0])
                #     print(input_ids[0])
                #     print(position_ids[0])
                #     input()

                if args.beam_size > 1:
                    traces = {k: v.tolist() for k, v in traces.items()}
                    output_ids = traces['pred_seq']
                else:
                    output_ids = traces[0].tolist()
                #print(output_ids)
                #input()
                for ii,w_ids in enumerate(output_ids):
                    output_buf = indexer.convert_ids_to_tokens(w_ids)
                    output_tokens = []
                    for t in output_buf:
                        if t in ("[SEP]", "[PAD]"):
                            break
                        output_tokens.append(t)
                        #print(t)
                    if args.tgt_lang=='en':
                        output_sequence = ' '.join(detokenize(output_tokens))
                        output_sequence = output_sequence.replace(' @ - @ ','-')
                    #print(id_,output_sequence)
                    #id_ = step_val*args.batch_size+ii
                    #output_sequence = output_sequence.replace('</w>',' ').replace(' ','')
                    if args.tgt_lang=='zh':
                        output_sequence = ''.join(detokenize(output_tokens)).replace('</w>','').replace('[SEP]','')
                    output_lines.append(output_sequence)

        # with open(os.path.join(args.output_dir,'translation_output.json'),'w') as f:
        #     json.dump(output_lines, f)
        with open(os.path.join(args.output_dir,'translation_output.txt'),'w') as f:
            for line in output_lines:
                f.writelines(line+'\n')

        # if args.corpus=='wmt':
        #     import sacrebleu
        #     import jieba
        #     with open(os.path.join(os.path.dirname(args.src_file),'newstest2017.'+args.tgt_lang),'r') as f:
        #         refs = f.readlines()
        #     refs = [[r.strip() for r in refs[:5]]]
        #     sys = [' '.join(list(jieba.cut(li, cut_all=False))) for li in output_lines]
        #     print(sys)
        #     print(refs)
        #     bleu = sacrebleu.corpus_bleu(sys, refs, tokenize='zh')
        #     print(bleu.score)
                




if __name__ == "__main__":
    main()