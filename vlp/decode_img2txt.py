"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import glob
import json
import argparse
import math
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

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# SPECIAL_TOKEN = ["[UNK]", "[PAD]", "[CLS]", "[MASK]"]


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
    parser.add_argument('--dataset', default='coco', type=str, nargs='*', help='')
    parser.add_argument('--max_len_en', default=25, type=int, help='maximum length of English in **bilingual** corpus')
    parser.add_argument('--max_len_zh', default=25, type=int, help='maximum length of Chinese in **bilingual** corpus')
    parser.add_argument('--max_len_en_cap', default=25, type=int, help='maximum length of English in **img2txt** corpus')
    parser.add_argument('--max_len_zh_cap', default=25, type=int, help='maximum length of Chinese in **img2txt** corpus')
    parser.add_argument('--len_vis_input', type=int, default=100, help="The length of visual token input")
    parser.add_argument("--src_file", default='$DATA_ROOT/{}/annotations/{}_dataset.json',
                        type=str, help="The input data file name.")
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--file_valid_jpgs', default='$DATA_ROOT/{}/annotations/{}_valid_jpgs.json', type=str)
    parser.add_argument('--image_root', type=str, default='$DATA_ROOT/{}/region_feat_gvd_wo_bgd')
    parser.add_argument('--region_bbox_file', default='raw_bbox/{}_detection_vg_100dets_vlp_checkpoint_trainval_bbox', type=str)
    parser.add_argument('--region_det_file_prefix', default='feat_cls_1000/{}_detection_vg_100dets_vlp_checkpoint_trainval', type=str)

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
    parser.add_argument('--max_tgt_length', type=int, default=20,
                        help="maximum length of target sequence")


    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--enable_butd', action='store_true',
                        help='set to take in region features')
    parser.add_argument('--output_dir', default='./result', type=str)

    args = parser.parse_args()
    dataset = {}
    for d in args.dataset:
        assert d in ['coco','aic','wmt']
        if d == 'coco':
            dataset[d] = {'max_len_a': args.len_vis_input, 'max_len_b': args.max_len_en_cap}
        elif d == 'aic':
            dataset[d] = {'max_len_a': args.len_vis_input, 'max_len_b': args.max_len_zh_cap}
        else:# d == 'wmt':
            dataset[d] = {'max_len_a': args.max_len_en, 'max_len_b': args.max_len_zh}
        dataset[d]['max_seq_length'] = dataset[d]['max_len_a'] + dataset[d]['max_len_b'] + 3
    args.dataset = dataset

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if args.enable_butd:
        assert(args.len_vis_input == 100)
        args.region_bbox_file = os.path.join(args.image_root, args.region_bbox_file)
        args.region_det_file_prefix = os.path.join(args.image_root, args.region_det_file_prefix) 

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
    tokenizers = {'en':tokenizer_en}
    if 'aic' in args.dataset or 'wmt' in args.dataset:
        tokenizer_zh = XLMTokenizer(args.xml_vocab, args.xml_merge)
        tokenizer_zh.tokenize = lambda x: tokenizer_zh._tokenize(x, lang='zh', bypass_tokenizer=True)
        with open(args.xml_vocab,'r') as f:
            tokenizer_zh.vocab = json.load(f)
        tokenizers['zh'] = tokenizer_zh
    indexer = Indexer([os.path.join(args.bert_model,'vocab.txt'), args.xml_vocab])

    for corpus in args.dataset:
        if corpus in ['coco','aic']:
            tokenizer = tokenizers['en'] if corpus=='coco' else tokenizers['zh']
            decode_pipeline= [seq2seq_loader.Preprocess4Seq2seqDecoder(
                corpus,  
                'zh' if corpus in ['aic'] else 'en',
                list(tokenizer.vocab.keys()), indexer, 
                max_len=args.dataset[corpus]['max_seq_length'],
                max_tgt_length=args.max_tgt_length, new_segment_ids=args.new_segment_ids,
                mode='s2s', len_vis_input=args.len_vis_input, enable_butd=args.enable_butd,
                region_bbox_file=args.region_bbox_file.format(corpus.upper(), corpus.lower()), 
                region_det_file_prefix=args.region_det_file_prefix.format(corpus.upper(), corpus.lower()))]
            eval_dataset = seq2seq_loader.Img2txtDataset(
                                    args.src_file.format(corpus.upper(), corpus.lower()),
                                    args.image_root.format(corpus.upper()), 
                                    args.split, args.batch_size,
                                    tokenizer,
                                    args.dataset[corpus]['max_seq_length'], 
                                    preprocessed=True,
                                    file_valid_jpgs=args.file_valid_jpgs.format(corpus.upper(), corpus.lower()),
                                    bi_uni_pipeline=decode_pipeline, use_num_imgs=-1,
                                    s2s_prob=1, bi_prob=0,
                                    enable_butd=args.enable_butd, tasks='img2txt')
            args.dataset[corpus]['eval_dataloader'] = torch.utils.data.DataLoader(
                        eval_dataset, batch_size=args.batch_size,
                        sampler=SequentialSampler(eval_dataset), num_workers=4, 
                        collate_fn=batch_list_to_batch_tensors, pin_memory=True)
        else:
            raise NotImplementedError # only support aic and coco now

    amp_handle = None
    if args.amp:
        from apex import amp
    #    amp_handle = amp.init(enable_caching=True)
    #    logger.info("enable fp16 with amp")

    # Prepare model
    cls_num_labels = 2
    #type_vocab_size = 6 if args.new_segment_ids else 2
    type_vocab_size = 12 if args.new_segment_ids else 2
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
        logger.info("***** Recover model: %s *****", model_recover_path)
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
        for corpus in args.dataset:
            if not 'eval_dataloader' in args.dataset[corpus]:
                continue
            print('corpus {}'.format(corpus))
            output_lines = {}
            val_iter_bar = tqdm(args.dataset[corpus]['eval_dataloader'])
            for step_val, val_iter_output in enumerate(val_iter_bar):
                info_, batch = val_iter_output[0],val_iter_output[1]
                with torch.no_grad():
                    batch = [t.to(device) for t in batch]
                    input_ids, segment_ids, position_ids, input_mask, task_idx, img, vis_pe = batch
                    if args.enable_butd:
                        conv_feats = img.data # Bx100x2048
                        vis_pe = vis_pe.data
                    else:
                        conv_feats, _ = cnn(img.data) # Bx2048x7x7
                        conv_feats = conv_feats.view(conv_feats.size(0), conv_feats.size(1),
                            -1).permute(0,2,1).contiguous()
                    if args.amp:
                        conv_feats = conv_feats.half()
                        vis_pe = vis_pe.half()

                    traces = model(conv_feats, vis_pe, input_ids, segment_ids, position_ids, input_mask, 
                        search_beam_size=args.beam_size, task_idx=task_idx, sample_mode='greedy') #validation greedy
                    if args.beam_size > 1:
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
                        output_sequence = ' '.join(detokenize(output_tokens))
                        if corpus=='coco':
                            id_ = int(info_[ii][2].split('_')[2])
                        else:
                            id_ = info_[ii][2]
                        #print(id_,output_sequence)
                        output_lines[id_] = output_sequence
            predictions = [{'image_id': ids_, 'caption': output_lines[ids_]} for ids_ in output_lines]
            with open(os.path.join(args.output_dir,'{}_{}_predictions.json').format(args.split, corpus),'w') as f:
                json.dump(predictions, f)
            print('Begin evaluating '+corpus)
            lang_stats = language_eval(corpus, predictions, 
                args.model_recover_path.split('/')[-2]+'-'+args.split+'-'+args.model_recover_path.split('/')[-1].split('.')[-2], 
                args.split,
                ['Bleu','METEOR','Rouge','CIDEr'])
            with open(os.path.join(args.output_dir,'{}_{}_scores.json').format(args.split, corpus),'w') as f:
                json.dump(lang_stats, f)


if __name__ == "__main__":
    main()
