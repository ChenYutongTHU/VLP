from random import randint, shuffle, choices
from random import random as rand
import pickle
import math
import json
from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from vlp.loader_utils import get_random_word, batch_list_to_batch_tensors, Pipeline

import torchvision.transforms as transforms
from PIL import Image
# https://stackoverflow.com/questions/12984426/python-pil-ioerror-image-file-truncated-with-big-images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import os
import imghdr
import numpy as np
import h5py
import tqdm, time

NEW_SEGMENT_IDS = {
    'bi_img':[0], 'bi_en_cap':[1], 'bi_zh_cap':[6],
    's2s_img':[4], 's2s_en_cap':[5], 's2s_zh_cap':[7],
    'bi_en':[8], 'bi_zh':[9],
    's2s_en':[10],'s2s_zh':[11]}
SEGMENT_IDS = {
    'img':[0], 'en':[1], 'zh':[6]}

def truncate_tokens_pair(tokens_a, tokens_b, max_len, max_len_a=0, max_len_b=0, trunc_seg=None, always_truncate_tail=False):
    num_truncated_a = [0, 0]
    num_truncated_b = [0, 0]
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if (max_len_a > 0) and len(tokens_a) > max_len_a:
            trunc_tokens = tokens_a
            num_truncated = num_truncated_a
        elif (max_len_b > 0) and len(tokens_b) > max_len_b:
            trunc_tokens = tokens_b
            num_truncated = num_truncated_b
        elif trunc_seg:
            # truncate the specified segment
            if trunc_seg == 'a':
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        else:
            # truncate the longer segment
            if len(tokens_a) > len(tokens_b):
                trunc_tokens = tokens_a
                num_truncated = num_truncated_a
            else:
                trunc_tokens = tokens_b
                num_truncated = num_truncated_b
        # whether always truncate source sequences
        if (not always_truncate_tail) and (rand() < 0.5):
            del trunc_tokens[0]
            num_truncated[0] += 1
        else:
            trunc_tokens.pop()
            num_truncated[1] += 1
    return num_truncated_a, num_truncated_b

class Txt2txtDataset(torch.utils.data.Dataset): #to-do
    def __init__(self, N_lines, split, batch_size, tokenizers, max_len, preprocessed=True,
        short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[], s2s_prob=1, bi_prob=0):
        super().__init__()
        self.tokenizers = tokenizers #{'zh': ,'en'}
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob #?
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        self.s2s_prob = s2s_prob
        self.bi_prob = bi_prob
        self.preprocessed = preprocessed
        print('Txt2txt Sample seq2seq {} and bidirectional {}'.format(self.s2s_prob, self.bi_prob))
        assert(self.s2s_prob + self.bi_prob == 1)
        self.N_lines = N_lines

        ''' # too slow
        if preprocessed:
            file_src = file_src.split('.')[0]+'_preprocessed.json'#'wmt_dataset.json' -> 'wmt_pretokenized'
            print('Loading preprocessed {}'.format(file_src))
            start = time.time()
            assert len(split)==1, split
            with open(file_src, 'r') as f:
                self.bilingual_corpus_preprocessed = json.load(f)
            self.bilingual_corpus_preprocessed = self.bilingual_corpus_preprocessed[split[0]]
            print('Finish Loading {}. Time cost {}'.format(file_src, round(time.time()-start, 2)))

            self.ex_list = []
            for en, zh in self.bilingual_corpus_preprocessed:
                self.ex_list.append([en, zh])



        else:
            print('Loading {} ...'.format(file_src))
            with open(file_src, 'r') as f:
                self.bilingual_corpus_raw = json.load(f)
                self.bilingual_corpus_raw = self.bilingual_corpus_raw[split[0]]  #only support training set now to-do

            print('Tokenizing Bilingual Corpus ...')
            self.ex_list = []
            for en, zh in tqdm.tqdm(self.bilingual_corpus_raw):
                en_t = self.tokenizers['en'].tokenize(en)
                zh_t = self.tokenizers['zh'].tokenize(zh)
                self.ex_list.append([en_t, zh_t])
        '''


    def __len__(self):
        return self.N_lines

    def __getitem__(self, idx):
        proc = choices(self.bi_uni_pipeline, weights=[self.s2s_prob, self.bi_prob])[0]
        instance = proc(idx)
        return instance
    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(self.N_lines / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, self.N_lines-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)

class Img2txtDataset(torch.utils.data.Dataset):
    """ Load image-sentence pairs """

    def __init__(self, file_src, image_root, split, batch_size, tokenizer, max_len, preprocessed=True, file_valid_jpgs='tmp.json', use_num_imgs=-1, short_sampling_prob=0.1, sent_reverse_order=False, bi_uni_pipeline=[], s2s_prob=1, bi_prob=0, enable_butd=False, tasks='img2txt'):
        super().__init__()
        self.tokenizer = tokenizer  # tokenize function
        self.max_len = max_len  # maximum length of tokens
        self.short_sampling_prob = short_sampling_prob
        self.bi_uni_pipeline = bi_uni_pipeline
        self.batch_size = batch_size
        self.sent_reverse_order = sent_reverse_order
        self.s2s_prob = s2s_prob
        self.bi_prob = bi_prob
        self.preprocessed = preprocessed
        print('Img2txt Sample seq2seq {} and bidirectional {}'.format(self.s2s_prob, self.bi_prob))
        assert(self.s2s_prob + self.bi_prob == 1)

        # read the file into memory
        self.ex_list = []

        if tasks == 'img2txt':
            if self.preprocessed:
                file_src = file_src.split('.')[0]+'_preprocessed.json'
            with open(file_src, "r", encoding='utf-8') as f_src:
                # raw inputs are given
                img_dat = json.load(f_src)['images']
                counter = 0

                if not os.path.isfile(file_valid_jpgs):
                    valid_img = {}
                    ii= 0
                    for src in img_dat:
                        if src['split'] in split:
                            if use_num_imgs == -1 or counter < use_num_imgs:
                                if enable_butd:
                                    src_tk = os.path.join(image_root, src.get('filepath', 'trainval'),
                                        src['filename'][:-4]+'.npy')
                                    for sent in src['sentences']:
                                        if self.preprocessed:
                                            tgt_tk = sent['indices']
                                        else:
                                            tgt_tk = tokenizer.tokenize(sent['raw'])
                                        assert len(tgt_tk) > 0
                                        self.ex_list.append((src_tk, tgt_tk, {'answers': ['dummy']}))
                                        #if counter%10000 == 0:
                                            #print(src_tk, tgt_tk)
                                    counter += 1
                                else:
                                    src_tk = os.path.join(image_root, src.get('filepath', ''),
                                        src['filename'])
                                    # check if the image is valid
                                    if os.stat(src_tk).st_size > 0 and imghdr.what(src_tk) == 'jpeg':
                                        try:
                                            Image.open(src_tk)
                                            for sent in src['sentences']:
                                                tgt_tk = tokenizer.tokenize(sent['raw'])
                                                assert len(tgt_tk) > 0
                                                self.ex_list.append((src_tk, tgt_tk, {'answers': ['dummy']}))
                                            valid_img[src['filename']] = src['filename']
                                            counter += 1
                                        except:
                                            pass
                    #json.dump(valid_img, open(file_valid_jpgs, 'w'))
                    #print('Saving {0} valid JPG IDs'.format(len(valid_img)))
                else:
                    valid_jpgs = set(json.load(open(file_valid_jpgs)))
                    print('Loading {0} valid JPG IDs!'.format(len(valid_jpgs)))
                    for src in tqdm.tqdm(img_dat):
                        if src['split'] in split:
                            if use_num_imgs == -1 or counter < use_num_imgs:
                                if enable_butd:
                                    src_tk = os.path.join(image_root, src.get('filepath', 'trainval'),
                                        src['filename'][:-4]+'.npy')
                                else:
                                    src_tk = os.path.join(image_root, src.get('filepath', ''),
                                        src['filename'])
                                # check if the image is valid
                                if src['filename'] in valid_jpgs:
                                    for sent in src['sentences']:
                                        if self.preprocessed:
                                            tgt_tk = sent['indices']
                                        else:
                                            tgt_tk = tokenizer.tokenize(sent['raw'])  #to-do Chinese Tokenizer.tokenize?
                                        assert len(tgt_tk) > 0
                                        self.ex_list.append((src_tk, tgt_tk, {'answers': ['dummy']}))
                                        # if counter%10000 == 0:
                                        #     print(src_tk, tgt_tk)
                                    counter += 1
        elif tasks == 'vqa2':
            counter = 0
            for file_s in file_src:
                img_dat = np.load(file_s, allow_pickle=True)
                assert(img_dat[0]['has_answer'] == True)
                for i in range(1, img_dat.shape[0]):
                    if use_num_imgs == -1 or counter < use_num_imgs:
                        if enable_butd:
                            src_tk = os.path.join(image_root, img_dat[i]['image_name'].split('_')[1],
                                img_dat[i]['feature_path'])
                        else:
                            raise NotImplementedError
                        tgt_tk = tokenizer.tokenize(img_dat[i]['question_str'])
                        ans_tk = {'answers': img_dat[i]['answers']}
                        self.ex_list.append((src_tk, tgt_tk, ans_tk))
                        counter += 1

        print('Load {0} documents'.format(len(self.ex_list)))

    def __len__(self):
        return len(self.ex_list)

    def __getitem__(self, idx):
        instance = self.ex_list[idx]
        proc = choices(self.bi_uni_pipeline, weights=[self.s2s_prob, self.bi_prob])[0]
        instance = proc(instance)
        return instance

    def __iter__(self):  # iterator to load data
        for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
            batch = []
            for __ in range(self.batch_size):
                idx = randint(0, len(self.ex_list)-1)
                batch.append(self.__getitem__(idx))
            # To Tensor
            yield batch_list_to_batch_tensors(batch)

class Preprocess4Seq2seqBilingual(Pipeline):
    def __init__(self, corpus, file_src, max_pred, mask_prob, vocab_words, tokenizers, indexer, max_len, split, preprocessed=True,
        new_segment_ids=False, truncate_config={}, mode="s2s", local_rank=-1):
        super().__init__()
        #default a-en b-zh
        self.corpus = corpus
        self.file_src = file_src
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.tokenizers = tokenizers
        self.indexer = indexer  # function from token to token index 
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)   
        self.max_len_a = truncate_config.get('max_len_a', None)  #
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.max_len = max_len
        self.trunc_seg = None  # truncate the longer segment   
        self.preprocessed = preprocessed
        assert len(split)==1, split
        self.split = split[0]

        assert mode in ("s2s", "bi")
        self.mode = mode  
        if mode == 's2s':
            self.task_idx = 3   # relax projection layer for different tasks
        elif mode == 'bi':
            self.task_idx = 0

    def __call__(self, idx):
        assert self.preprocessed, 'Only support preprocessed h5py now'
        if self.preprocessed:
            self.file_src_en = os.path.join(os.path.dirname(self.file_src),'corpus_en_preprocessed.hdf5')
            self.file_src_zh = os.path.join(os.path.dirname(self.file_src),'corpus_zh_preprocessed.hdf5')
            with h5py.File(self.file_src_en,'r')  as en_f, \
                h5py.File(self.file_src_zh,'r') as zh_f:
                tokens_a = list(en_f[self.split][idx])
                tokens_b = list(zh_f[self.split][idx])


        #tokens_a, tokens_b = instance[:2] #[CLS] [SEP] [SEP] unincluded #en zh
        #print(tokens_a, tokens_b)
        truncate_tokens_pair(tokens_a, tokens_b,
            self.max_len_a+self.max_len_b, max_len_a=self.max_len_a, max_len_b=self.max_len_b,
            trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)
        if self.preprocessed:
            tokens = self.indexer(['[CLS]']) + tokens_a + self.indexer(['[SEP]']) + tokens_b + self.indexer(['[SEP]'])
        else:
            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

        if self.new_segment_ids:
            if self.mode == 's2s':
                segment_ids = NEW_SEGMENT_IDS['s2s_en'] * (len(tokens_a)+2) + NEW_SEGMENT_IDS['s2s_zh'] * (len(tokens_b)+1)
            elif self.mode == 'bi':
                segment_ids = NEW_SEGMENT_IDS['bi_en'] * (len(tokens_a)+2) + NEW_SEGMENT_IDS['bi_zh'] * (len(tokens_b)+1)
        else:
            segment_ids = SEGMENT_IDS['en'] * (len(tokens_a)+2) + SEGMENT_IDS['zh'] * (len(tokens_b)+1)




        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()

        if rand()<0.5: #mask En a:
            mask_type = 'a'
        else: # mask en b
            mask_type = 'b'


        for i, tk in enumerate(tokens):
            #for bilingual dataset, we can mask both tokens
            if not self.preprocessed and tk in ['[CLS]', '[SEP]']:
                special_pos.add(i)
            elif self.preprocessed and tk in self.indexer(['[CLS]','[SEP]']):
                special_pos.add(i)
            else:
                if mask_type=='a' and i<1+len(tokens_a):
                    cand_pos.append(i)
                elif mask_type=='b' and i>=2+len(tokens_a):
                    cand_pos.append(i)
        shuffle(cand_pos)
        n_pred = min(self.max_pred, max(
            1, int(round(len(cand_pos) * self.mask_prob))))

        # if n_pred==3:
        #     assert len(cand_pos)>=3, cand_pos
        masked_pos = cand_pos[:n_pred]
        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] =  '[MASK]' 
                if self.preprocessed:
                    tokens[pos] = self.indexer(tokens[pos])
            elif rand() < 0.5:  # 10%
                if mask_type=='a':
                    tokens[pos] = get_random_word(list(self.tokenizers['en'].vocab.keys()))
                else:
                    tokens[pos] = get_random_word(list(self.tokenizers['zh'].vocab.keys()))
                if self.preprocessed:
                    tokens[pos] = self.indexer(tokens[pos])


        masked_weights = [1]*len(masked_tokens)

        # Token Indexing #to-check
        if not self.preprocessed:
            input_ids = self.indexer(tokens)
            masked_ids = self.indexer(masked_tokens)
        else:
            input_ids = tokens[:]
            masked_ids = masked_tokens[:]

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        #attention mask
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        if mask_type=='a':
            second_st, second_end = 0, len(tokens_a)+2
        else:
            second_st, second_end = len(tokens_a)+2, len(tokens_a)+len(tokens_b)+3

        if self.mode == "s2s": #to-check
            if mask_type=='a':
                input_mask[:, len(tokens_a)+2:len(tokens_a)+len(tokens_b)+3].fill_(1) #all attend to zh
            else:
                input_mask[:, :len(tokens_a)+2].fill_(1) #all attend to en 
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
        else:
            input_mask = torch.tensor([1] * len(tokens) + [0] * n_pad, dtype=torch.long) \
                .unsqueeze(0).expand(self.max_len, self.max_len).clone() #do not attend to zero_pad tokens

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)
        assert len(masked_ids)==3 and len(masked_pos)==3 and len(masked_weights)==3, [len(masked_ids),len(masked_pos),len(masked_weights),self.max_pred, n_pred]

        return (self.corpus,self.mode), (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, -1, self.task_idx)#, img, vis_masked_pos, vis_pe, ans_tk)

class Preprocess4Seq2seq(Pipeline):
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, corpus, max_pred, mask_prob, vocab_words, indexer, preprocessed=True, max_len=512, block_mask=False, new_segment_ids=False, truncate_config={}, mask_image_regions=False, mode="s2s", len_vis_input=49, vis_mask_prob=0.25, enable_butd=False, region_bbox_file='', region_det_file_prefix='', local_rank=-1, load_vqa_ann=False, lang='en'):
        super().__init__()
        self.corpus = corpus
        self.max_pred = max_pred  # max tokens of prediction
        self.mask_prob = mask_prob  # masking probability
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.always_truncate_tail = truncate_config.get(
            'always_truncate_tail', False)
        self.max_len_b = truncate_config.get('max_len_b', None)
        self.trunc_seg = truncate_config.get('trunc_seg', None)
        self.mask_image_regions = mask_image_regions
        assert mode in ("s2s", "bi")
        self.mode = mode
        self.region_bbox_file = region_bbox_file
        self.region_det_file_prefix = region_det_file_prefix
        self.lang = lang
        self.preprocessed = preprocessed

        if mode == 's2s':
            self.task_idx = 3   # relax projection layer for different tasks
        elif mode == 'bi':
            self.task_idx = 0

        self.len_vis_input = len_vis_input
        self.vis_mask_prob = vis_mask_prob

        # for images
        self.enable_butd = enable_butd
        if not enable_butd:
            self.Resize = transforms.Resize((255, 255))
            self.RandomCrop = transforms.RandomCrop((224, 224))
            self.ToTensor = transforms.ToTensor()
            self.res_Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        else:
            if load_vqa_ann:
                # import packages from pythia
                import pythia.tasks.processors as pythia_proc # VQAAnswerProcessor
                from pythia.utils.configuration import ConfigNode
                args = {'vocab_file': 'pythia/data/vocabs/answers_vqa.txt', 'num_answers':10, 'preprocessor':{'type':'simple_word', 'params':{}}}
                args = ConfigNode(args)
                self.ans_proc = pythia_proc.registry.get_processor_class('vqa_answer')(args)
            else:
                self.ans_proc = None


    def __call__(self, instance):
        img_path, tokens_b = instance[:2]
        
        tokens_a = ['[UNK]'] * self.len_vis_input
        if self.preprocessed:
            tokens_a = self.indexer(tokens_a)

        truncate_tokens_pair(tokens_a, tokens_b,
            self.len_vis_input + self.max_len_b, max_len_b=self.max_len_b,
            trunc_seg=self.trunc_seg, always_truncate_tail=self.always_truncate_tail)

        # Add Special Tokens
        if self.preprocessed:
            tokens = self.indexer(['[CLS]']) + tokens_a + self.indexer(['[SEP]']) + tokens_b + self.indexer(['[SEP]'])
        else:
            tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']


        if self.new_segment_ids:
            if self.mode == 's2s':
                segment_ids = NEW_SEGMENT_IDS['s2s_img'] * (len(tokens_a)+2) + NEW_SEGMENT_IDS['s2s_'+self.lang+'_cap'] * (len(tokens_b)+1)
            elif self.mode == 'bi':
                segment_ids = NEW_SEGMENT_IDS['bi_img'] * (len(tokens_a)+2) + NEW_SEGMENT_IDS['bi_'+self.lang+'_cap'] * (len(tokens_b)+1)
        else:
            segment_ids = SEGMENT_IDS['img'] * (len(tokens_a)+2) + SEGMENT_IDS[self.lang] * (len(tokens_b)+1)

        # For masked Language Models
        # the number of prediction is sometimes less than max_pred when sequence is short
        effective_length = len(tokens_b)
        n_pred = min(self.max_pred, max(
            1, int(round(effective_length * self.mask_prob))))
        # candidate positions of masked tokens
        cand_pos = []
        special_pos = set()
        for i, tk in enumerate(tokens):
            # only mask tokens_b (target sequence)
            # we will mask [SEP] as an ending symbol
            if (i >= len(tokens_a)+2): #and (tk != '[CLS]'):
                cand_pos.append(i)
            else:
                special_pos.add(i)
        shuffle(cand_pos)

        masked_pos = cand_pos[:n_pred]

        if self.mask_image_regions:
            vis_masked_pos = np.random.choice(self.len_vis_input,
                int(self.len_vis_input*self.vis_mask_prob), replace=False)+1 # +1 for [CLS], always of the same length, no need to pad
        else:
            vis_masked_pos = []

        masked_tokens = [tokens[pos] for pos in masked_pos]
        for pos in masked_pos:
            if rand() < 0.8:  # 80%
                tokens[pos] = '[MASK]'
                if self.preprocessed:
                    tokens[pos] = self.indexer(tokens[pos]) 
            elif rand() < 0.5:  # 10%
                tokens[pos] = get_random_word(self.vocab_words)
                if self.preprocessed:
                    tokens[pos] = self.indexer(tokens[pos]) 
        # when n_pred < max_pred, we only calculate loss within n_pred
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        if self.preprocessed:
            input_ids = tokens[:]
            masked_ids = masked_tokens[:]
        else:
            input_ids = self.indexer(tokens)
            masked_ids = self.indexer(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)

        # self-attention mask
        input_mask = torch.zeros(self.max_len, self.max_len, dtype=torch.long)
        second_st, second_end = len(tokens_a)+2, len(tokens_a)+len(tokens_b)+3

        if self.mode == "s2s":
            input_mask[:, :len(tokens_a)+2].fill_(1)
            input_mask[second_st:second_end, second_st:second_end].copy_(
                self._tril_matrix[:second_end-second_st, :second_end-second_st])
        else:
            input_mask = torch.tensor([1] * len(tokens) + [0] * n_pad, dtype=torch.long) \
                .unsqueeze(0).expand(self.max_len, self.max_len).clone()

        if self.mask_image_regions:
            input_mask[:, vis_masked_pos].fill_(0) # block the masked visual feature

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
            masked_weights.extend([0] * n_pad)

        if not self.enable_butd:
            # loading images
            img = Image.open(img_path).convert('RGB')
            img = self.Resize(img)
            img = self.RandomCrop(img)
            img = self.ToTensor(img)
            img = self.res_Normalize(img)
        else:
            # loading pre-processed features
            img_id = img_path.split('/')[-1].split('.')[0]
            if self.region_det_file_prefix != '':
                # read data from h5 files
                with h5py.File(self.region_det_file_prefix+'_feat'+img_id[-1:] +'.h5', 'r') as region_feat_f, \
                        h5py.File(self.region_det_file_prefix+'_cls'+img_id[-1:] +'.h5', 'r') as region_cls_f, \
                        h5py.File(self.region_bbox_file+img_id[-1:]+'.h5', 'r') as region_bbox_f:
                    img = torch.from_numpy(region_feat_f[img_id][:]).float()
                    cls_label = torch.from_numpy(region_cls_f[img_id][:]).float()
                    vis_pe = torch.from_numpy(region_bbox_f[img_id][:])
            else:
                # legacy, for some datasets, read data from numpy files
                img = torch.from_numpy(np.load(img_path))
                cls_label = torch.from_numpy(np.load(img_path.replace('.npy', '_cls_prob.npy')))
                with h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                    vis_pe = torch.from_numpy(region_bbox_f[img_id][:])

            # lazy normalization of the coordinates...
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
                F.layer_norm(cls_label, [1601])), dim=-1) # 1601 hard coded...

            # process answer
            if self.ans_proc:
                ans_tk = self.ans_proc(instance[2])['answers_scores']
            else:
                ans_tk = img.new(1)

        return (self.corpus, self.mode),(input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, -1, self.task_idx, img, vis_masked_pos, vis_pe, ans_tk)


class Preprocess4Seq2seqDecoder(Pipeline):  #to-do self.lang/ new_segment_id/ txt
    """ Pre-processing steps for pretraining transformer """

    def __init__(self, vocab_words, indexer, max_len=512, max_tgt_length=128, new_segment_ids=False, mode="s2s", enable_butd=False, len_vis_input=49, region_bbox_file='', region_det_file_prefix='', lang='en'):
        super().__init__()
        self.vocab_words = vocab_words  # vocabulary (sub)words
        self.indexer = indexer  # function from token to token index
        self.max_len = max_len
        self._tril_matrix = torch.tril(torch.ones(
            (max_len, max_len), dtype=torch.long))
        self.new_segment_ids = new_segment_ids
        self.task_idx = 3   # relax projection layer for different tasks
        self.mode = mode
        self.lang = lang
        if self.mode != "s2s":
            raise ValueError("Invalid mode for seq2seq decode: %s" % self.mode)
        self.max_tgt_length = max_tgt_length
        self.len_vis_input = len_vis_input
        self.region_bbox_file = region_bbox_file
        self.region_det_file_prefix = region_det_file_prefix

        # for images
        self.enable_butd = enable_butd
        if not enable_butd:
            self.Resize = transforms.Resize((255, 255))
            self.CenterCrop = transforms.CenterCrop((224, 224))
            self.ToTensor = transforms.ToTensor()
            self.res_Normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __call__(self, instance):
        img_path, max_a_len = instance[:2]
        tokens_a = ['[UNK]'] * self.len_vis_input

        # Add Special Tokens
        padded_tokens_a = ['[CLS]'] + tokens_a + ['[SEP]']
        assert len(padded_tokens_a) <= max_a_len + 2
        if max_a_len + 2 > len(padded_tokens_a):
            padded_tokens_a += ['[PAD]'] * \
                (max_a_len + 2 - len(padded_tokens_a))
        assert len(padded_tokens_a) == max_a_len + 2
        max_len_in_batch = min(self.max_tgt_length +
                               max_a_len + 2, self.max_len)
        tokens = padded_tokens_a
        if self.new_segment_ids:
            segment_ids = NEW_SEGMENT_IDS['s2s_img']*(len(padded_tokens_a)) \
                + NEW_SEGMENT_IDS['s2s_'+self.lang]*(max_len_in_batch - len(padded_tokens_a))
        else:
            segment_ids = SEGMENT_IDS['img']*(len(padded_tokens_a)) \
                + SEGMENT_IDS[self.lang]*(max_len_in_batch - len(padded_tokens_a))

        position_ids = []
        for i in range(len(tokens_a) + 2):
            position_ids.append(i)
        for i in range(len(tokens_a) + 2, max_a_len + 2):
            position_ids.append(0)
        for i in range(max_a_len + 2, max_len_in_batch):
            position_ids.append(i - (max_a_len + 2) + len(tokens_a) + 2)

        # Token Indexing
        input_ids = self.indexer(tokens)

        # Zero Padding
        input_mask = torch.zeros(
            max_len_in_batch, max_len_in_batch, dtype=torch.long)
        input_mask[:, :len(tokens_a)+2].fill_(1)
        second_st, second_end = len(padded_tokens_a), max_len_in_batch

        input_mask[second_st:second_end, second_st:second_end].copy_(
            self._tril_matrix[:second_end-second_st, :second_end-second_st])

        if not self.enable_butd:
            try:
                img = Image.open(img_path).convert('RGB')
                img = self.Resize(img)
                img = self.CenterCrop(img)
                img = self.ToTensor(img)
                img = self.res_Normalize(img)
            except:
                print('Unable to load image {}! Loading mean image instead...'.format(img_path))
                img = torch.Tensor(self.res_Normalize.mean).view(-1, 1, 1).expand(
                    (3, self.CenterCrop.size[0], self.CenterCrop.size[1]))
        else:
            img_id = img_path.split('/')[-1].split('.')[0]
            if self.region_det_file_prefix != '':
                # read data from h5 files
                with h5py.File(self.region_det_file_prefix+'_feat'+img_id[-1:] +'.h5', 'r') as region_feat_f, \
                        h5py.File(self.region_det_file_prefix+'_cls'+img_id[-1:] +'.h5', 'r') as region_cls_f, \
                        h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                    img = torch.from_numpy(region_feat_f[img_id][:]).float()
                    cls_label = torch.from_numpy(region_cls_f[img_id][:]).float()
                    vis_pe = torch.from_numpy(region_bbox_f[img_id][:])
            else:
                # legacy, for some datasets, read data from numpy files
                img = torch.from_numpy(np.load(img_path))
                cls_label = torch.from_numpy(np.load(img_path.replace('.npy', '_cls_prob.npy')))
                with h5py.File(self.region_bbox_file, 'r') as region_bbox_f:
                    vis_pe = torch.from_numpy(region_bbox_f[img_id][:])

            # lazy normalization of the coordinates...
            w_est = torch.max(vis_pe[:, [0, 2]])*1.+1e-5
            h_est = torch.max(vis_pe[:, [1, 3]])*1.+1e-5
            vis_pe[:, [0, 2]] /= w_est
            vis_pe[:, [1, 3]] /= h_est
            rel_area = (vis_pe[:, 3]-vis_pe[:, 1])*(vis_pe[:, 2]-vis_pe[:, 0])
            rel_area.clamp_(0)

            vis_pe = torch.cat((vis_pe[:, :4], rel_area.view(-1, 1), vis_pe[:, 5:]), -1) # confident score
            normalized_coord = F.normalize(vis_pe.data[:, :5]-0.5, dim=-1)
            vis_pe = torch.cat((F.layer_norm(vis_pe, [6]), \
                F.layer_norm(cls_label, [1601])), dim=-1) # 1601 hard coded...

        return (input_ids, segment_ids, position_ids, input_mask, self.task_idx, img, vis_pe)

class CombinedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets_dict):
        super().__init__() #sample strategy using a ratio
        self.datasets_dict = datasets_dict
        self.datasets = [self.datasets_dict[key] for key in self.datasets_dict]
        self.lens = [len(d) for d in self.datasets]
        print('Length of Combined Dataset {} sum:{}'.format(self.lens, sum(self.lens)))

    def __len__(self):
        return sum([len(d) for d in self.datasets])
    def __getitem__(self, idx):
        clen = 0
        for i, length in enumerate(self.lens):
            if idx>=clen and idx<clen+length:
                return self.datasets[i][idx-clen]
            clen += length
        raise
        return None

class WeightedRandom_DistributedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, corpus_size, batch_size, alpha, num_batches, drop_last=False, num_replicas=None, rank=None, seed=0):
        #note that here 
        #batch_size -> batch_size_per_gpu
        #num_batches -> num_batches_per_gpu
        self.drop_last=drop_last
        self.alpha = alpha
        self.corpus_size = corpus_size

        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size*self.num_replicas #batch_size -> batch_size per GPU
        self.num_batches = num_batches # num_batches_per_gpu
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        all_batches = []
        cnt = 0
        weights = []
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        for cs in self.corpus_size:
            shuffled_indices =(torch.randperm(cs, generator=g)+cnt).tolist()
            i = 0
            while i<cs:
                if i+self.batch_size>cs:
                    if self.drop_last:
                        break
                    else:
                        all_batches.append(shuffled_indices[i:]+shuffled_indices[0:self.batch_size-(cs-i)])
                        weights.append(math.pow(cs, self.alpha-1))      
                else:
                    all_batches.append(shuffled_indices[i:i+self.batch_size])
                    weights.append(math.pow(cs, self.alpha-1))  
                i += self.batch_size
            cnt += cs                                                        
        batch_indices = torch.multinomial(torch.tensor(weights,dtype=torch.float32), self.num_batches, 
            generator=g, replacement=False)

        for ind in batch_indices:
            yield all_batches[ind][self.rank::self.num_replicas]
        #subsample

    def __len__(self):
        return self.num_batches
    def set_epoch(self, epoch):
        self.epoch = epoch

class WeightedRandom_BatchSampler(torch.utils.data.Sampler):
    def __init__(self, corpus_size, batch_size, alpha, num_batches, drop_last=False):
        self.batch_size=batch_size
        self.drop_last=drop_last
        self.alpha = alpha
        self.num_batches = num_batches
        self.corpus_size = corpus_size
    def __iter__(self):
        all_batches = []
        cnt = 0
        weights = []
        for cs in self.corpus_size:
            shuffled_indices =(torch.randperm(cs)+cnt).tolist()
            i = 0
            while i<cs:
                if i+self.batch_size>cs:
                    if self.drop_last:
                        break
                    else:
                        all_batches.append(shuffled_indices[i:]+shuffled_indices[0:self.batch_size-(cs-i)])
                        weights.append(math.pow(cs, self.alpha-1))
                else:
                    all_batches.append(shuffled_indices[i:i+self.batch_size])
                    weights.append(math.pow(cs, self.alpha-1))

                i += self.batch_size
            cnt += cs
        batch_indices = torch.multinomial(torch.tensor(weights,dtype=torch.float32), self.num_batches, replacement=False)
        for ind in batch_indices:
            yield all_batches[ind] 

    def __len__(self):
        return self.num_batches     
