"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
sys.path.append('/data/private/chenyutong/VLP/')
import logging
import glob
import math
import json
import argparse
from tqdm import tqdm, trange
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import random, math
import copy

from pytorch_pretrained_bert.tokenization import BertTokenizer, WhitespaceTokenizer, Indexer
from transformers import XLMTokenizer
from pytorch_pretrained_bert.modeling import BertForPreTrainingLossMask, BertForSeq2SeqDecoder
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from vlp.loader_utils import batch_list_to_batch_tensors
import vlp.seq2seq_loader as seq2seq_loader
#from vlp.seq2seq_loader import Preprocess4Seq2seqBilingual
from vlp.scst_utils import *
from misc.data_parallel import DataParallelImbalance


def _get_max_epoch_model(output_dir):
    fn_model_list = glob.glob(os.path.join(output_dir, "model.*.bin"))
    fn_optim_list = glob.glob(os.path.join(output_dir, "optim.*.bin"))
    if (not fn_model_list) or (not fn_optim_list):
        return None
    both_set = set([int(Path(fn).stem.split('.')[-1]) for fn in fn_model_list]
                   ) & set([int(Path(fn).stem.split('.')[-1]) for fn in fn_optim_list])
    if both_set:
        return max(both_set)
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    # Data augmentation
    parser.add_argument('--dataset', default='coco', type=str, nargs='*', help='')
    parser.add_argument('--sampling_alpha', default=0.5, type=float)
    parser.add_argument('--sampling_beta', default=0.5, type=float, help='#samples per epoch=sampling_beta*total_num_samples')
    parser.add_argument('--max_len_en', default=25, type=int, help='maximum length of English in **bilingual** corpus')
    parser.add_argument('--max_len_zh', default=25, type=int, help='maximum length of Chinese in **bilingual** corpus')
    parser.add_argument('--max_len_en_cap', default=25, type=int, help='maximum length of English in **img2txt** corpus')
    parser.add_argument('--max_len_zh_cap', default=25, type=int, help='maximum length of Chinese in **img2txt** corpus')
    parser.add_argument('--len_vis_input', type=int, default=100, help="The length of visual token input")
    parser.add_argument('--wmt_N_lines', type=int, default=24752392, help='The total number of wmt lines')

    parser.add_argument("--src_file", default='$DATA_ROOT/{}/annotations/{}_dataset.json',
                        type=str, help="The input data file name.")
    parser.add_argument('--file_valid_jpgs', default='$DATA_ROOT/{}/annotations/{}_valid_jpgs.json', type=str)
    parser.add_argument('--image_root', type=str, default='$DATA_ROOT/{}/region_feat_gvd_wo_bgd')
    parser.add_argument('--region_bbox_file', default='raw_bbox/{}_detection_vg_100dets_vlp_checkpoint_trainval_bbox', type=str)
    parser.add_argument('--region_det_file_prefix', default='feat_cls_1000/{}_detection_vg_100dets_vlp_checkpoint_trainval', type=str)

    # General
    parser.add_argument("--bert_model", default="bert-base-cased", type=str,
                        help="Bert pre-trained model selected in the list: bert-base-cased, bert-large-cased.")
    parser.add_argument("--xml_vocab",type=str, default='./download_models/xml_vocab.json')
    parser.add_argument("--xml_merge",type=str, default='./download_models/xml_merges.txt')
    parser.add_argument("--config_path", default=None, type=str,
                        help="Bert config file path.")
    parser.add_argument("--output_dir",
                        default='tmp',type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_file",
                        default="training.log",
                        type=str,
                        help="The output directory where the log will be written.")
    parser.add_argument("--model_recover_path",
                        default=None,
                        type=str,
                        help="The file of fine-tuned pretraining model.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training. This should ALWAYS be set to True.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=64,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--label_smoothing", default=0, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay",
                        default=0.01,
                        type=float,
                        help="The weight decay rate for Adam.")
    parser.add_argument("--finetune_decay",
                        action='store_true',
                        help="Weight decay to the original weights.")
    parser.add_argument("--num_train_epochs",
                        default=30,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--global_rank",
                        type=int,
                        default=-1,
                        help="global_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    # parser.add_argument('--fp16', action='store_true',
    #                     help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp32_embedding', action='store_true',
                        help="Whether to use 32-bit float precision instead of 32-bit for embeddings")
    # parser.add_argument('--loss_scale', type=float, default=0,
    #                     help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
    #                          "0 (default value): dynamic loss scaling.\n"
    #                          "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--amp', action='store_true',
                        help="Whether to use amp for fp16")
    parser.add_argument('--from_scratch', action='store_true',
                        help="Initialize parameters with random values (i.e., training from scratch).")
    parser.add_argument('--new_segment_ids', action='store_true',
                        help="Use new segment ids for bi-uni-directional LM.")
    parser.add_argument('--tokenized_input', action='store_true',
                        help="Whether the input is tokenized.")
    parser.add_argument('--trunc_seg', default='b',
                        help="Truncate_config: first truncate segment A/B (option: a, b).")
    parser.add_argument('--always_truncate_tail', action='store_true',
                        help="Truncate_config: Whether we should always truncate tail.")
    parser.add_argument("--mask_prob", default=0.15, type=float,
                        help="Number of prediction is sometimes less than max_pred when sequence is short.")
    parser.add_argument('--max_pred', type=int, default=3,
                        help="Max tokens of prediction.")
    parser.add_argument("--num_workers", default=4, type=int,
                        help="Number of workers for the data loader.")
    parser.add_argument('--max_position_embeddings', type=int, default=None,
                        help="max position embeddings")

    # Others for VLP
    parser.add_argument('--enable_visdom', action='store_true')
    parser.add_argument('--visdom_port', type=int, default=8888)
    parser.add_argument('--enable_tensorboard', action='store_true')
    parser.add_argument('--summary_steps', type=int, default=100)
    # parser.add_argument('--resnet_model', type=str, default='imagenet_weights/resnet101.pth')
    parser.add_argument('--split', type=str, nargs='+', default=['train', 'restval'])

    parser.add_argument('--world_size', default = 1, type = int,
                        help = 'number of distributed processes')
    parser.add_argument('--dist_url', default='file://[PT_OUTPUT_DIR]/nonexistent_file', type = str,
                        help = 'url used to set up distributed training')
    parser.add_argument('--sche_mode', default='warmup_linear', type=str,
                        help="warmup_linear | warmup_constant | warmup_cosine")
    parser.add_argument('--drop_prob', default=0.1, type=float)
    parser.add_argument('--use_num_imgs', default=-1, type=int)
    parser.add_argument('--vis_mask_prob', default=0, type=float)
    parser.add_argument('--max_drop_worst_ratio', default=0, type=float)
    parser.add_argument('--drop_after', default=6, type=int)

    parser.add_argument('--s2s_prob', default=1, type=float,
                        help="Percentage of examples that are bi-uni-directional LM (seq2seq).")
    parser.add_argument('--bi_prob', default=0, type=float,
                        help="Percentage of examples that are bidirectional LM.")
    parser.add_argument('--enable_butd', action='store_true',
                        help='set to take in region features')
    parser.add_argument('--tasks', default='img2txt',
                        help='img2txt | vqa2')
    parser.add_argument('--relax_projection',
                        action='store_true',
                        help="Use different projection layers for tasks.")
    parser.add_argument('--scst', action='store_true',
                        help='Self-critical sequence training')

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
    print('global_rank: {}, local rank: {} Corpora: {}'.format(args.global_rank, args.local_rank, args.dataset))
    #input()

    args.mask_image_regions = (args.vis_mask_prob > 0) # whether to mask out image regions
    args.dist_url = args.dist_url.replace('[PT_OUTPUT_DIR]', args.output_dir)

    # arguments inspection
    assert(args.tasks in ('img2txt', 'vqa2'))
    assert args.enable_butd == True, 'only support region attn! featmap attn deprecated'
    assert (not args.scst) or args.dataset == 'coco', 'scst support on coco only!'
    if args.scst:
        assert args.dataset == 'coco', 'scst support on coco only!'
        assert args.max_pred == 0 and args.mask_prob == 0, 'no mask for scst!'
        rl_crit = RewardCriterion()

    if args.enable_butd:
        assert(args.len_vis_input == 100)
        args.region_bbox_file = os.path.join(args.image_root, args.region_bbox_file)
        #args.region_det_file_prefix = os.path.join(args.image_root, args.region_det_file_prefix) if args.dataset in ('cc', 'coco') and args.region_det_file_prefix != '' else ''
        args.region_det_file_prefix = os.path.join(args.image_root, args.region_det_file_prefix)  # not support flickr30k now
    # output config
    if args.local_rank in [-1,0]:
        os.makedirs(args.output_dir, exist_ok=True)
        json.dump(args.__dict__, open(os.path.join(
            args.output_dir, 'opt.json'), 'w'), sort_keys=True, indent=2)

    logging.basicConfig(
        filename=os.path.join(args.output_dir, 'rank{}_'.format(args.local_rank)+args.log_file),
        filemode='w',
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info(dataset)

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl', init_method = args.dist_url,
        #     world_size=args.world_size, rank=args.global_rank)
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, amp training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.amp))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(
        args.train_batch_size / args.gradient_accumulation_steps)

    # fix random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # plotting loss, optional
    if args.enable_visdom:
        import visdom
        vis = visdom.Visdom(port=args.visdom_port, env=args.output_dir)
        vis_window={'iter': None, 'score':None}
    if args.enable_tensorboard:
        from tensorboardX import SummaryWriter
        if args.local_rank in [-1,0]:
            writer = SummaryWriter(args.output_dir)

    tokenizer_en = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case,
        cache_dir=args.output_dir+'/.pretrained_model')
    if args.max_position_embeddings:
        tokenizer_en.max_len = args.max_position_embeddings
    tokenizer_en= WhitespaceTokenizer() if args.tokenized_input else tokenizer_en
    tokenizers = {'en':tokenizer_en}

    if 'aic' in args.dataset or 'wmt' in args.dataset:
        tokenizer_zh = XLMTokenizer(args.xml_vocab, args.xml_merge)
        tokenizer_zh.tokenize = lambda x: tokenizer_zh._tokenize(x, lang='zh', bypass_tokenizer=True)
        with open(args.xml_vocab,'r') as f:
            tokenizer_zh.vocab = json.load(f)
        tokenizers['zh'] = tokenizer_zh

    indexer = Indexer([os.path.join(args.bert_model,'vocab.txt'), args.xml_vocab])

    if args.do_train:

        for corpus in args.dataset:
            print('\nCorpus {}'.format(corpus))
            if corpus in ['coco', 'aic']:
                tokenizer = tokenizers['en'] if corpus=='coco' else tokenizers['zh']
                bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seq(
                    corpus, args.max_pred, args.mask_prob,
                    list(tokenizer.vocab.keys()), indexer, max_len=args.dataset[corpus]['max_seq_length'],
                    preprocessed=True,
                    new_segment_ids=args.new_segment_ids, 
                    truncate_config={
                        'max_len_b': args.dataset[corpus]['max_len_b'], 'trunc_seg': args.trunc_seg, 'always_truncate_tail':
                        args.always_truncate_tail}, 
                    mask_image_regions=args.mask_image_regions,
                    mode="s2s", len_vis_input=args.len_vis_input,
                    vis_mask_prob=args.vis_mask_prob, enable_butd=args.enable_butd,
                    region_bbox_file=args.region_bbox_file.format(corpus.upper(), corpus.lower()), 
                    region_det_file_prefix=args.region_det_file_prefix.format(corpus.upper(), corpus.lower()),
                    local_rank=args.local_rank, load_vqa_ann=(args.tasks=='vqa2'), lang='zh' if corpus in ['aic'] else 'en')]

                bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seq(
                    corpus, args.max_pred, args.mask_prob,
                    list(tokenizer.vocab.keys()), indexer, max_len=args.dataset[corpus]['max_seq_length'],
                    preprocessed=True,
                    new_segment_ids=args.new_segment_ids,  
                    truncate_config={
                        'max_len_b': args.dataset[corpus]['max_len_b'], 'trunc_seg': args.trunc_seg, 'always_truncate_tail':
                        args.always_truncate_tail}, 
                    mask_image_regions=args.mask_image_regions,
                    mode="bi", len_vis_input=args.len_vis_input,
                    vis_mask_prob=args.vis_mask_prob, enable_butd=args.enable_butd,
                    region_bbox_file=args.region_bbox_file.format(corpus.upper(), corpus.lower()), 
                    region_det_file_prefix=args.region_det_file_prefix.format(corpus.upper(), corpus.lower()),
                    local_rank=args.local_rank, load_vqa_ann=(args.tasks=='vqa2'), lang='zh' if corpus in ['aic'] else 'en'))

                split = args.split #'['train']
                if corpus=='coco' and split[0]=='train':
                    split = split+['restval']
                args.dataset[corpus]['train_dataset'] = seq2seq_loader.Img2txtDataset(
                    args.src_file.format(corpus.upper(), corpus.lower()), 
                    args.image_root.format(corpus.upper()), 
                    split, args.train_batch_size,
                    tokenizer, 
                    args.dataset[corpus]['max_seq_length'], 
                    preprocessed=True,
                    file_valid_jpgs=args.file_valid_jpgs.format(corpus.upper(), corpus.lower()),
                    bi_uni_pipeline=bi_uni_pipeline, use_num_imgs=args.use_num_imgs,
                    s2s_prob=args.s2s_prob, bi_prob=args.bi_prob,
                    enable_butd=args.enable_butd, tasks=args.tasks)

            elif corpus == 'wmt':
                #print(seq2seq_loader.__dict__)
                bi_uni_pipeline = [seq2seq_loader.Preprocess4Seq2seqBilingual(
                    'wmt',
                    args.src_file.format(corpus.upper(), corpus.lower()),
                    args.max_pred, args.mask_prob, 
                    list(indexer.vocab.keys()),  tokenizers, 
                    indexer, args.dataset[corpus]['max_seq_length'],
                    split=args.split,
                    preprocessed=True,
                    new_segment_ids=args.new_segment_ids, 
                    truncate_config={
                        'max_len_a': args.dataset[corpus]['max_len_a'],
                        'max_len_b': args.dataset[corpus]['max_len_b'], 
                        'trunc_seg': None, 'always_truncate_tail':args.always_truncate_tail}, 
                    mode='s2s', local_rank=args.local_rank)]
                bi_uni_pipeline.append(seq2seq_loader.Preprocess4Seq2seqBilingual(
                        'wmt',
                        args.src_file.format(corpus.upper(), corpus.lower()),
                        args.max_pred, args.mask_prob, 
                        list(indexer.vocab.keys()), tokenizers, 
                        indexer, args.dataset[corpus]['max_seq_length'],
                        split=args.split,
                        preprocessed=True,
                        new_segment_ids=args.new_segment_ids, 
                        truncate_config={
                            'max_len_a': args.dataset[corpus]['max_len_a'],
                            'max_len_b': args.dataset[corpus]['max_len_b'], 
                            'trunc_seg': None, 'always_truncate_tail':args.always_truncate_tail}, 
                        mode='bi', local_rank=args.local_rank)
                    )
                args.dataset[corpus]['train_dataset'] = seq2seq_loader.Txt2txtDataset(
                    args.wmt_N_lines,
                    args.split, args.train_batch_size,
                    tokenizers, args.dataset[corpus]['max_seq_length'], 
                    preprocessed=True,
                    bi_uni_pipeline=bi_uni_pipeline,s2s_prob=args.s2s_prob, bi_prob=args.bi_prob)  
        

        train_dataset = seq2seq_loader.CombinedDataset(
                    datasets_dict={c: args.dataset[c]['train_dataset'] for c in args.dataset})
        #train_sampler = RandomSampler(train_dataset, replacement=False)
        logger.info('************Data statistics******************')
        num_samples = []
        total_num_samples = 0
        for corpus in args.dataset:
            N = len(args.dataset[corpus]['train_dataset'])
            logger.info('{} #{}'.format(corpus, N))
            num_samples.append(N)
            total_num_samples += N
        logger.info('total number samples {}'.format(total_num_samples))
        logger.info('number samples per epoch {}'.format(total_num_samples*args.sampling_beta))
        if args.local_rank == -1:
            train_batch_sampler = seq2seq_loader.WeightedRandom_BatchSampler(
                num_samples, args.train_batch_size,
                args.sampling_alpha, num_batches=math.ceil(total_num_samples*args.sampling_beta
                    /args.train_batch_size),
                drop_last=False) #to-check
            # train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size,
            #     sampler=SequentialSampler(train_dataset), num_workers=args.num_workers, 
            #     collate_fn=batch_list_to_batch_tensors, pin_memory=True)
            #num_batch the total number of batch per epoch
        else:
            num_batches = math.ceil(total_num_samples*args.sampling_beta/torch.distributed.get_world_size()/args.train_batch_size)
            train_batch_sampler = seq2seq_loader.WeightedRandom_DistributedBatchSampler(
                num_samples, args.train_batch_size, 
                args.sampling_alpha, num_batches=num_batches,
                drop_last=False)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
            batch_sampler=train_batch_sampler, num_workers=args.num_workers,
            collate_fn=batch_list_to_batch_tensors, pin_memory=True)



#batch_sampler batch_sampler (Sampler or Iterable, optional) 
#– like sampler, but returns a batch of indices at a time. 
#Mutually exclusive with batch_size, shuffle, sampler, and drop_last.

    # note: args.train_batch_size has been changed to (/= args.gradient_accumulation_steps)
    t_total = int(len(train_dataloader) * args.num_train_epochs * 1. /
                  args.gradient_accumulation_steps) # number of backward steps

    amp_handle = None
    if  args.amp:
        from apex import amp
        # amp_handle = amp.init(enable_caching=True)
        # logger.info("enable fp16 with amp")

    # Prepare model
    recover_step = _get_max_epoch_model(args.output_dir)
    cls_num_labels = 2
    #type_vocab_size = 6 if args.new_segment_ids else 2
    type_vocab_size = 12 if args.new_segment_ids else 2
    relax_projection = 4 if args.relax_projection else 0
    task_idx_proj = 3 if args.tasks == 'img2txt' else 0
    mask_word_id, eos_word_ids, pad_word_ids = indexer(["[MASK]", "[SEP]", "[PAD]"]) # index in BERT vocab: 103, 102, 0

    if (recover_step is None) and (args.model_recover_path is None):
        # if _state_dict == {}, the parameters are randomly initialized
        # if _state_dict == None, the parameters are initialized with bert-init
        assert args.scst == False, 'must init from maximum likelihood training'
        _state_dict = {} if args.from_scratch else None
        model = BertForPreTrainingLossMask.from_pretrained(
            args.bert_model, state_dict=_state_dict, num_labels=cls_num_labels,
            vocab_size=len(indexer),
            type_vocab_size=type_vocab_size, relax_projection=relax_projection,
            config_path=args.config_path, task_idx=task_idx_proj,
            max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
            fp32_embedding=args.fp32_embedding, cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.local_rank),
            drop_prob=args.drop_prob, enable_butd=args.enable_butd,
            len_vis_input=args.len_vis_input, tasks=args.tasks)
        global_step = 0
    else:
        if recover_step:
            logger.info("***** Recover model: %d *****", recover_step)
            model_recover = torch.load(os.path.join(
                args.output_dir, "model.{0}.bin".format(recover_step)))
            # recover_step == number of epochs
            global_step = math.floor(
                recover_step * t_total * 1. / args.num_train_epochs)
        elif args.model_recover_path:
            logger.info("***** Recover model: %s *****",
                        args.model_recover_path)
            model_recover = torch.load(args.model_recover_path)
            global_step = 0
        if not args.scst:
            model = BertForPreTrainingLossMask.from_pretrained(
                args.bert_model, state_dict=model_recover, num_labels=cls_num_labels,
                vocab_size=len(indexer),
                type_vocab_size=type_vocab_size, relax_projection=relax_projection,
                config_path=args.config_path, task_idx=task_idx_proj,
                max_position_embeddings=args.max_position_embeddings, label_smoothing=args.label_smoothing,
                fp32_embedding=args.fp32_embedding, cache_dir=args.output_dir+'/.pretrained_model_{}'.format(args.local_rank),
                drop_prob=args.drop_prob, enable_butd=args.enable_butd,
                len_vis_input=args.len_vis_input, tasks=args.tasks)
        else:
            model = BertForSeq2SeqDecoder.from_pretrained(args.bert_model,
                max_position_embeddings=args.max_position_embeddings, config_path=args.config_path,
                state_dict=model_recover, num_labels=cls_num_labels, type_vocab_size=type_vocab_size,
                task_idx=task_idx_proj, mask_word_id=mask_word_id, search_beam_size=1,
                eos_id=eos_word_ids, enable_butd=args.enable_butd,
                len_vis_input=args.len_vis_input)

        del model_recover
        torch.cuda.empty_cache()

    # deprecated
    # from vlp.resnet import resnet
    # cnn = resnet(args.resnet_model, _num_layers=101, _fixed_block=4, pretrained=True) # no finetuning

    if args.amp:
        #model.half()
        # cnn.half()
        if args.fp32_embedding:
            raise NotImplementedError
            model.bert.embeddings.word_embeddings.float()
            model.bert.embeddings.position_embeddings.float()
            model.bert.embeddings.token_type_embeddings.float()
    model.to(device)


    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    if args.amp:
        try:
            # from apex.optimizers import FP16_Optimizer
            #from pytorch_pretrained_bert.optimization_fp16 import FP16_Optimizer_State
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False)
        # if args.loss_scale == 0:
        #     optimizer = FP16_Optimizer_State(
        #         optimizer, dynamic_loss_scale=True)
        # else:
        #     optimizer = FP16_Optimizer_State(
        #         optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             schedule=args.sche_mode,
                             t_total=t_total)

    if args.amp:
        model, optimizer = amp.initialize(model, optimizer, opt_level='O2')#'02')

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP 
        except ImportError:
            raise ImportError(
                'Please install apex from https://www.github.com/nvidia/apex to use distributed fp16 for training.')
        model = DDP(model)
    elif n_gpu > 1:
        model = DataParallelImbalance(model)



    if recover_step:
        logger.info("***** Recover optimizer: %d *****", recover_step)
        optim_recover = torch.load(os.path.join(
            args.output_dir, "optim.{0}.bin".format(recover_step)))
        if hasattr(optim_recover, 'state_dict'):
            optim_recover = optim_recover.state_dict()
        optimizer.load_state_dict(optim_recover)
        # if args.loss_scale == 0:
        #     logger.info("***** Recover optimizer: dynamic_loss_scale *****")
        #     optimizer.dynamic_loss_scale = True

    logger.info("***** CUDA.empty_cache() *****")
    torch.cuda.empty_cache()

    if args.do_train:
        logger.info("***** Running training *****")
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", t_total)
        logger.info("  Loader length = %d", len(train_dataloader))

        model.train()
        if recover_step:
            start_epoch = recover_step+1
        else:
            start_epoch = 1
        for i_epoch in trange(start_epoch, args.num_train_epochs+1, desc="Epoch"):
            if args.local_rank >= 0:
                train_batch_sampler.set_epoch(i_epoch-1)
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
            nbatches = len(train_dataloader)
            train_loss = []
            pretext_loss = []
            vqa2_loss = []
            scst_reward = []
            for step, iter_output in enumerate(iter_bar):
                info_, batch = iter_output[0], iter_output[1]
                # logger.info('rank {}, step {}'.format(torch.distributed.get_rank(), step))
                # logger.info(info_)
                batch = [t.to(device) for t in batch]

                if info_[0][0] in ['coco','aic']:
                    input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, is_next, task_idx, img, vis_masked_pos, vis_pe, ans_labels = batch
                    # if args.fp16:
                    #     img = img.half()
                    #     vis_pe = vis_pe.half()

                    if args.enable_butd:
                        conv_feats = img.data # Bx100x2048
                        vis_pe = vis_pe.data
                    else:
                        conv_feats, _ = cnn(img.data) # Bx2048x7x7
                        conv_feats = conv_feats.view(conv_feats.size(0), conv_feats.size(1),
                            -1).permute(0,2,1).contiguous()
                    # print('input_Ids \n')
                    # print(input_ids.shape)
                    # print(input_ids[0])
                    # input()
                    # print('segment_ids \n')
                    # print(segment_ids.shape)
                    # print(segment_ids[0])
                    # input()
                    # print('input_mask \n')
                    # print(input_mask.shape)
                    # torch.set_printoptions(profile='full')
                    # print(input_mask[0,:60])
                    # input()
                    # print('lm label ids \n')
                    # print(lm_label_ids.shape)
                    # print(lm_label_ids[0])
                    # input()
                    # print('masked pos \n')
                    # print(masked_pos.shape)
                    # print(masked_pos[0])
                    # input()


                    if not args.scst:
                        loss_tuple = model('img2txt',conv_feats, vis_pe, input_ids, segment_ids,
                            input_mask, lm_label_ids, ans_labels, is_next, masked_pos=masked_pos,
                            masked_weights=masked_weights, task_idx=task_idx,
                            vis_masked_pos=vis_masked_pos, mask_image_regions=args.mask_image_regions,
                            drop_worst_ratio=args.max_drop_worst_ratio if i_epoch > args.drop_after else 0)
                        mean_reward = loss_tuple[0].new(1).fill_(0)
                    else:
                        # scst training
                        model.eval()
                        position_ids = torch.arange(input_ids.size(1), dtype=input_ids.dtype,
                            device=input_ids.device).unsqueeze(0).expand_as(input_ids)
                        input_dummy = input_ids[:, :args.len_vis_input + 2] # +2 for [CLS] and [SEP]
                        greedy_res = input_ids.new(input_ids.size(0), input_ids.size(1)-args.len_vis_input-2).fill_(0)
                        gen_result = input_ids.new(input_ids.size(0), input_ids.size(1)-args.len_vis_input-2).fill_(0)

                        with torch.no_grad():
                            greedy_res_raw, _ = model(conv_feats, vis_pe, input_dummy, segment_ids,
                                position_ids, input_mask, task_idx=task_idx, sample_mode='greedy')
                            for b in range(greedy_res_raw.size(0)):
                                for idx in range(greedy_res_raw.size(1)):
                                    if greedy_res_raw[b][idx] not in [eos_word_ids, pad_word_ids]:
                                        greedy_res[b][idx] = greedy_res_raw[b][idx]
                                    else:
                                        if greedy_res_raw[b][idx] == eos_word_ids:
                                            greedy_res[b][idx] = eos_word_ids
                                        break
                        model.train()
                        gen_result_raw, sample_logprobs = model(conv_feats, vis_pe, input_dummy, segment_ids,
                            position_ids, input_mask, task_idx=task_idx, sample_mode='sample')
                        for b in range(gen_result_raw.size(0)):
                            for idx in range(gen_result_raw.size(1)):
                                if gen_result_raw[b][idx] not in [eos_word_ids, pad_word_ids]:
                                    gen_result[b][idx] = gen_result_raw[b][idx]
                                else:
                                    if gen_result_raw[b][idx] == eos_word_ids:
                                        gen_result[b][idx] = eos_word_ids
                                    break

                        gt_ids = input_ids[:, args.len_vis_input+2:]
                        reward = get_self_critical_reward(greedy_res, gt_ids, gen_result, gt_ids.size(0))
                        reward = torch.from_numpy(reward).float().to(gen_result.device)
                        mean_reward = reward.mean()
                        loss = rl_crit(sample_logprobs, gen_result.data, reward)

                        loss_tuple = [loss, loss.new(1).fill_(0.), loss.new(1).fill_(0.)]
                else:  #wmt
                    input_ids, segment_ids, input_mask, lm_label_ids, masked_pos, masked_weights, is_next, task_idx = batch
                    loss_tuple = model('txt2txt', 
                        input_ids=input_ids, token_type_ids=segment_ids,
                        attention_mask=input_mask, masked_lm_labels=lm_label_ids, 
                        ans_labels=None, next_sentence_label=is_next, 
                        masked_pos=masked_pos, masked_weights=masked_weights, task_idx=task_idx,
                        drop_worst_ratio=args.max_drop_worst_ratio if i_epoch > args.drop_after else 0)
                    mean_reward = loss_tuple[0].new(1).fill_(0)

                # disable pretext_loss_deprecated for now
                masked_lm_loss, pretext_loss_deprecated, ans_loss = loss_tuple
                if n_gpu > 1:    # mean() to average on multi-gpu. For dist, this is done through gradient addition.
                    masked_lm_loss = masked_lm_loss.mean()
                    pretext_loss_deprecated = pretext_loss_deprecated.mean()
                    ans_loss = ans_loss.mean()
                loss = masked_lm_loss #+ pretext_loss_deprecated + ans_loss

                # logging for each step (i.e., before normalization by args.gradient_accumulation_steps)
                iter_bar.set_description('Iter (loss=%5.3f)' % loss.item())
                train_loss.append(loss.item())
                pretext_loss.append(pretext_loss_deprecated.item())
                vqa2_loss.append(ans_loss.item())
                scst_reward.append(mean_reward.item())
                if step%100 == 0:
                    logger.info("Epoch {}, Iter {}, Loss {:.2f}, Pretext {:.2f}, VQA2 {:.2f}, Mean R {:.3f}\n".format(i_epoch, step, np.mean(train_loss), np.mean(pretext_loss), np.mean(vqa2_loss), np.mean(scst_reward)))

                if args.enable_visdom:
                    if vis_window['iter'] is None:
                        vis_window['iter'] = vis.line(
                            X=np.tile(np.arange((i_epoch-1)*nbatches+step,
                                      (i_epoch-1)*nbatches+step+1), (1,1)).T,
                            Y=np.column_stack((np.asarray([np.mean(train_loss)]),)),
                            opts=dict(title='Training Loss',
                                      xlabel='Training Iteration',
                                      ylabel='Loss',
                                      legend=['total'])
                        )
                    else:
                        vis.line(
                            X=np.tile(np.arange((i_epoch-1)*nbatches+step,
                                      (i_epoch-1)*nbatches+step+1), (1,1)).T,
                            Y=np.column_stack((np.asarray([np.mean(train_loss)]),)),
                            opts=dict(title='Training Loss',
                                      xlabel='Training Iteration',
                                      ylabel='Loss',
                                      legend=['total']),
                            win=vis_window['iter'],
                            update='append'
                        )

                # ensure that accumlated gradients are normalized
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    # if amp_handle:
                    #     amp_handle._clear_cache()
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    lr_this_step = args.learning_rate * \
                        warmup_linear(global_step/t_total,
                                      args.warmup_proportion)
                    if args.amp:
                        # modify learning rate with special warm up BERT uses
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()

                    if args.enable_tensorboard and global_step%args.summary_steps==0:
                        if args.local_rank in [-1,0]:
                            writer.add_scalar('Training_Loss_{}'.format(info_[0][0]), train_loss[-1], global_step)

                    global_step += 1

            # Save a trained model
            logger.info(
                "** ** * Saving fine-tuned model and optimizer ** ** * ")
            model_to_save = model.module if hasattr(
                model, 'module') else model  # Only save the model it-self
            output_model_file = os.path.join(
                args.output_dir, "model.{0}.bin".format(i_epoch))
            output_optim_file = os.path.join(
                args.output_dir, "optim.{0}.bin".format(i_epoch))
            if args.local_rank in (-1, 0): # save model if the first device or no dist
                torch.save(copy.deepcopy(model_to_save).cpu().state_dict(), output_model_file)
                # torch.save(optimizer.state_dict(), output_optim_file) # disable for now, need to sanitize state and ship everthing back to cpu

            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

            if args.local_rank >= 0:
                torch.distributed.barrier()


if __name__ == "__main__":
    main()
