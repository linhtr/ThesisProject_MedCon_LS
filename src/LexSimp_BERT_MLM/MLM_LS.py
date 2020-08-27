"""
The original version of this script is taken from: https://github.com/qiang2100/BERT-LS/.
Modifications have been made in order to work with own data.
"""

import os
import argparse
import random
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz
from collections import OrderedDict
from tqdm import tqdm, trange
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from transformers import AutoTokenizer, AutoModel, AutoModelWithLMHead
from sklearn.metrics.pairwise import cosine_similarity as cosine
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_sentence_to_token(sentence, seq_length, tokenizer):
    """
    Define the words from the tokenized text and create a positional embedding for the words.
    For example:
    tokenized_text = ['den', '##mark', 'is', 'credited', 'as', 'co', '-', 'developer', 'of', 'the', 'per', '##tus', '##sis', 'vaccine', 'in', 'the', '1920s', 'and', '1930s', '.']
    words = ['denmark', 'is', 'credited', 'as', 'co-developer', 'of', 'the', 'pertussis', 'vaccine', 'in', 'the', '1920s', 'and', '1930s', '.']
    position = [[22, 23], 24, 25, 26, [27, 28, 29], 30, 31, [32, 33, 34], 35, 36, 37, 38, 39, 40, 41]
    """

    tokenized_text = tokenizer.tokenize(sentence)

    if len(tokenized_text) > seq_length - 2:
        tokenized_text = tokenized_text[0:(seq_length - 2)]

    special = []
    isSpecial = False
    whole_word = ''
    words = []
    position = []

    # Start position of S' sentence is moved 2 indexes due to [CLS] and [SEP]
    start_pos = len(tokenized_text) + 2

    for index in range(len(tokenized_text) - 1):

        # Dealing with words with a dash that are splitted. For example: "co", "-", "developer"
        if (tokenized_text[index + 1] == "-" and tokenized_text[index + 2] != "-") or \
                (tokenized_text[index + 1] == "–" and tokenized_text[index + 2] != "–") or \
                (tokenized_text[index + 1] == "'" and tokenized_text[index + 2] != "'"):
            special.append(start_pos + index)
            whole_word += tokenized_text[index]  # "co"
            continue

        if tokenized_text[index] == "-" or tokenized_text[index] == "–" or tokenized_text[index] == "'":
            special.append(start_pos + index)
            whole_word += tokenized_text[index]  # "co" + "-"
            if tokenized_text[index - 1] == "-" or tokenized_text[index - 1] == "–":
                words.append(whole_word)
                position.append(start_pos + index)
                special = []
                whole_word = ''
            continue

        if (tokenized_text[index] != "-" and tokenized_text[index - 1] == "-" and not tokenized_text[index - 2] == "-") or \
                (tokenized_text[index] != "–" and tokenized_text[index - 1] == "–" and not tokenized_text[index - 2] == "–") or \
                (tokenized_text[index] != "'" and tokenized_text[index - 1] == "'" and not tokenized_text[index - 2] == "'"):
            special.append(start_pos + index)
            whole_word += tokenized_text[index]  # "co" + "-" + "developer"
            whole_word = whole_word.replace('##', '')
            if (tokenized_text[index + 1][0:2] != "##"):
                words.append(whole_word)
                position.append(special)
                special = []
                whole_word = ''
                isSpecial = False
                continue
            else:
                isSpecial = True
                continue

        # Dealing with subword tokens. For example: 'per', '##tus', '##sis'
        if (tokenized_text[index + 1][0:2] == "##"):
            special.append(start_pos + index)
            whole_word += tokenized_text[index] # 'per'
            isSpecial = True
            continue
        else:
            if isSpecial:
                isSpecial = False
                special.append(start_pos + index)
                whole_word += tokenized_text[index] # 'per' + '##tus'
                whole_word = whole_word.replace('##', '')
                words.append(whole_word)
                position.append(special)
                special = []
                whole_word = ''
            else:
                position.append(start_pos + index)
                words.append(tokenized_text[index])

    # Dealing with the last token
    if isSpecial:
        isSpecial = False
        special.append(start_pos + index + 1)
        position.append(special)
        whole_word += tokenized_text[index + 1] # 'per' + '##tus' + '##sis'
        whole_word = whole_word.replace('##', '')
        words.append(whole_word)
    else:
        position.append(start_pos + index + 1)
        words.append(tokenized_text[index + 1])

    return tokenized_text, words, position


def convert_whole_word_to_feature(tokens_a, mask_position, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    # tokens_a = tokenizer.tokenize(sentence)
    # print(mask_position)

    """
    For a sentence pair, we need to specify to which sentence each token belongs to (segment_ids or input_type_ids).
    In our case, sentence S has a series of 0s, and sentence S' has a series of 1s.
    """
    tokens = []
    input_type_ids = [] #segment_ids
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)

    tokens.append("[SEP]")
    input_type_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(1)

    tokens.append("[SEP]")
    input_type_ids.append(1)

    count = 0
    mask_position_length = len(mask_position)

    while count in range(mask_position_length):
        index = mask_position_length - 1 - count
        pos = mask_position[index]
        if index == 0:
            tokens[pos] = '[MASK]'
        else:
            del tokens[pos]
            del input_type_ids[pos]

        count += 1

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return InputFeatures(unique_id=0, tokens=tokens, input_ids=input_ids, input_mask=input_mask,
                         input_type_ids=input_type_ids)


def convert_token_to_feature(tokens_a, mask_position, seq_length, tokenizer):
    """Loads a data file into a list of `InputFeature`s."""

    # tokens_a = tokenizer.tokenize(sentence)
    # print(mask_position)

    """
    For a sentence pair, we need to specify to which sentence each token belongs to (segment_ids or input_type_ids).
    In our case, sentence S has a series of 0s, and sentence S' has a series of 1s.
    """
    tokens = []
    input_type_ids = []
    tokens.append("[CLS]")
    input_type_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(0)

    tokens.append("[SEP]")
    input_type_ids.append(0)

    for token in tokens_a:
        tokens.append(token)
        input_type_ids.append(1)

    tokens.append("[SEP]")
    input_type_ids.append(1)

    true_word = ''
    if isinstance(mask_position, list):
        for pos in mask_position:
            true_word = true_word + tokens[pos]
            tokens[pos] = '[MASK]'
    else:
        true_word = tokens[mask_position]
        tokens[mask_position] = '[MASK]'

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < seq_length:
        input_ids.append(0)
        input_mask.append(0)
        input_type_ids.append(0)

    assert len(input_ids) == seq_length
    assert len(input_mask) == seq_length
    assert len(input_type_ids) == seq_length

    return InputFeatures(unique_id=0, tokens=tokens, input_ids=input_ids, input_mask=input_mask,
                         input_type_ids=input_type_ids)


def getWordmap(wordVecPath):
    words = []
    We = []
    f = open(wordVecPath, 'r')
    lines = f.readlines()

    for (n, line) in enumerate(lines):
        if (n == 0):
            print(line)
            continue
        word, vect = line.rstrip().split(' ', 1)
        vect = np.fromstring(vect, sep=' ')
        We.append(vect)
        words.append(word)

        # if(n==200000):
        #    break
    f.close()
    return (words, We)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def substitution_selection(source_word, pre_tokens, pre_scores, stopwords, ps, num_selection=10):
    cur_tokens = []

    source_stem = ps.stem(source_word)

    assert num_selection <= len(pre_tokens)

    punctuationMarks = [',', '.', ';', ':', '(', '-', ')', '{', '}', '[', ']', '\\', '/', '!', '?', '<', '>', '"', "'"]

    for i in range(len(pre_tokens)):
        token = pre_tokens[i]

        if token[0:2] == "##":
            continue

        if (token == source_word):
            continue

        if str(token).isdigit():
            continue

        if token.isalpha() and len(token) == 1:
            continue

        if token in punctuationMarks:
            continue

        if token.lower() in stopwords:
            continue

        token_stem = ps.stem(token)

        if (token_stem.lower() == source_stem.lower()):
            continue

        # if (len(token_stem) >= 3) and (token_stem[:3].lower() == source_stem[:3].lower()):
        #     continue

        cur_tokens.append(token)

        if (len(cur_tokens) == num_selection):
            break

    if (len(cur_tokens) == 0):
        cur_tokens = pre_tokens[0:num_selection + 1]

    assert len(cur_tokens) > 0

    return cur_tokens

def extract_context(words, mask_index, window):
    #extract 7 words around the content word

    length = len(words)
    half = int(window/2)

    assert mask_index>=0 and mask_index<length

    context = ""

    if length<=window:
        context = words
    elif mask_index<length-half and mask_index>=half:
        context = words[mask_index-half:mask_index+half+1]
    elif mask_index<half:
        context = words[0:window]
    elif mask_index>=length-half:
        context = words[length-window:length]
    else:
        print("Wrong!")

    return context

def get_score(sentence, tokenizer, maskedLM):
    tokenize_input = tokenizer.tokenize(sentence)
    tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
    tensor_input = tensor_input.to('cuda')
    sentence_loss = 0

    for i, word in enumerate(tokenize_input):
        original_word = tokenize_input[i]
        tokenize_input[i] = '[MASK]'
        # print(tokenize_input)
        mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        mask_input = mask_input.to('cuda')
        with torch.no_grad():
            word_loss = maskedLM(mask_input, masked_lm_labels=tensor_input)[0].data.cpu().numpy()
        sentence_loss += word_loss
        tokenize_input[i] = original_word

    return np.exp(sentence_loss / len(tokenize_input))


def LM_score(source_word, source_context, substitution_selection, tokenizer, maskedLM):
    # source_index = source_context.index(source_word)

    source_sentence = ''

    for context in source_context:
        source_sentence += context + " "

    source_sentence = source_sentence.strip()
    # print(source_sentence)
    LM = []

    for substibution in substitution_selection:
        sub_sentence = source_sentence.replace(source_word, substibution)

        # print(sub_sentence)
        score = get_score(sub_sentence, tokenizer, maskedLM)
        LM.append(score)

    return LM

def preprocess_SR(source_word, substitution_selection, fasttext_dico, fasttext_emb, word_count):
    ss = []
    # ss_score=[]
    sis_scores=[]
    count_scores=[]
    # source_count = 10
    # if source_word in word_count:
    #     source_count = word_count[source_word]

    isFast = True

    if (source_word not in fasttext_dico):
        isFast = False
    else:
        source_emb = fasttext_emb[fasttext_dico.index(source_word)].reshape(1,-1)

    if isFast == False and source_word.lower() in fasttext_dico:
        isFast = True
        source_emb = fasttext_emb[fasttext_dico.index(source_word.lower())].reshape(1,-1)

    # ss.append(source_word)

    for sub in substitution_selection:

        if sub.lower() not in word_count:
            continue
        else:
            sub_count = word_count[sub.lower()]

        # if sub_count<source_count:
        #     continue
        if isFast:
            if sub not in fasttext_dico:
                if sub.lower() not in fasttext_dico:
                    continue
                else:
                    sub_emb = fasttext_emb[fasttext_dico.index(sub.lower())].reshape(1, -1)
            else:
                sub_emb = fasttext_emb[fasttext_dico.index(sub)].reshape(1, -1)

            sis = cosine(source_emb, sub_emb)[0][0]

            # if sis<0.35:
            #    continue
            sis_scores.append(sis)

        ss.append(sub)
        count_scores.append(sub_count)

    return ss, sis_scores, count_scores

def compute_context_sis_score(source_word, sis_context, substitution_selection, fasttext_dico, fasttext_emb):
    context_sis = []

    word_context = []

    for con in sis_context:
        if con == source_word or (con not in fasttext_dico):
            continue

        word_context.append(con)

    if len(word_context) != 0:
        for sub in substitution_selection:
            sub_emb = fasttext_emb[fasttext_dico.index(sub)].reshape(1, -1)
            all_sis = 0
            for con in word_context:
                token_index_fast = fasttext_dico.index(con)
                all_sis += cosine(sub_emb, fasttext_emb[token_index_fast].reshape(1, -1))

            context_sis.append(all_sis / len(word_context))
    else:
        for i in range(len(substitution_selection)):
            context_sis.append(len(substitution_selection) - i)

    return context_sis

def substitution_ranking(source_word, source_context, substitution_selection, fasttext_dico, fasttext_emb, word_count,
                         tokenizer, maskedLM):
    ss, sis_scores, count_scores = preprocess_SR(source_word, substitution_selection, fasttext_dico, fasttext_emb,
                                                 word_count)

    # print(ss)
    if len(ss) == 0:
        return source_word

    if len(sis_scores) > 0:
        seq = sorted(sis_scores, reverse=True)
        sis_rank = [seq.index(v) + 1 for v in sis_scores]

    rank_count = sorted(count_scores, reverse=True)
    count_rank = [rank_count.index(v) + 1 for v in count_scores]

    lm_score = LM_score(source_word, source_context, ss, tokenizer, maskedLM)
    rank_lm = sorted(lm_score)
    lm_rank = [rank_lm.index(v) + 1 for v in lm_score]

    bert_rank = []
    for i in range(len(ss)):
        bert_rank.append(i + 1)

    if len(sis_scores) > 0:
        all_ranks = [bert + sis + count + LM for bert, sis, count, LM in zip(bert_rank, sis_rank, count_rank, lm_rank)]
    else:
        all_ranks = [bert + count + LM for bert, count, LM in zip(bert_rank, count_rank, lm_rank)]
    # all_ranks = [con for con in zip(context_rank)]

    print("bert_rank: ", bert_rank)
    if len(sis_scores) > 0:
        print("sis_rank: ", sis_rank)
    print("count_rank: ", count_rank)
    print("lm_rank: ", lm_rank)
    print("all_ranks: ", all_ranks)

    pre_index = all_ranks.index(min(all_ranks))
    pre_word = ss[pre_index]

    return pre_word

def process_string(string):
    stopword_list = set(stopwords.words('english'))
    ps = PorterStemmer()
    tokens = nltk.word_tokenize(string)
    rootWords = []
    for w in tokens:
        if w.lower() in stopword_list:
            continue
        rootWord = ps.stem(w)
        rootWords.append(rootWord)

    return ' '.join(rootWords)

def fuzzy_match(string1, string2):
    processed_string1 = process_string(string1)
    processed_string2 = process_string(string2)

    sim_score = fuzz.token_sort_ratio(processed_string1, processed_string2)
    return sim_score

def read_file(input_file):
    """Read a list of `InputExample`s from an input file."""
    sentences = []

    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()

            sentences.append(line)

    return sentences

def read_df(input_file):
    df = pd.read_csv(input_file, sep=';')
    # df = df.tail(len(df) - 1326).reset_index()
    origin = df.origin
    sent_id = df.sent_id
    eval_sentences = df.sentence
    mask_words = df.source_term
    CHV_selections = df.CHV_selection
    CHV_substitutions = df.CHV_substitution
    CHV_sim_scores = df.sim_scores
    return origin, sent_id, eval_sentences, mask_words, CHV_selections, CHV_substitutions, CHV_sim_scores

def save_output(output_path, df):
    df = pd.DataFrame(df).set_index(['origin', 'sent_id'])
    # If csv doesn't exist yet, create one, and append to it in the next iterations
    with open(output_path, 'a') as f:
        df.to_csv(f, header=f.tell() == 0, sep=';')

def setup_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--eval_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The evaluation data dir.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-large-cased-whole-word-masking, "
                             "emilyalsentzer/Bio_Discharge_Summary_BERT, emilyalsentzer/Bio_ClinicalBERT.")
    parser.add_argument("--output_path",
                        default=None,
                        type=str,
                        required=True,
                        help="The output path of writing substitution selection.")
    parser.add_argument("--word_embeddings",
                        default=None,
                        type=str,
                        required=True,
                        help="The path of word embeddings")
    parser.add_argument("--word_frequency",
                        default=None,
                        type=str,
                        required=True,
                        help="The path of word frequency.")
    parser.add_argument("--stopwords",
                        default=None,
                        type=str,
                        required=True,
                        help="The path of stopwords.")

    # Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from AWS S3?")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--num_selections",
                        default=10,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_eval_epochs",
                        default=1,
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
                        default=-1,
                        type=int,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")

    return parser

def main():
    parser = setup_parser()
    args = parser.parse_args()

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_eval:
        raise ValueError("At least `do_eval` must be True.")

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None

    # Prepare model. Load pre-trained model weights.
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                   'distributed_{}'.format(args.local_rank))
    model = BertForMaskedLM.from_pretrained(args.bert_model, cache_dir=cache_dir)
    if args.fp16:
        model.half()
    model.to(device)

    # output_sr_file = open(args.output_SR_file, "a+")

    # Load fastText word embeddings
    print("Loading embeddings ...")
    wordVecPath = args.word_embeddings
    # wordVecPath = "./fastText/crawl-300d-2M-subword.vec"
    fasttext_dico, fasttext_emb = getWordmap(wordVecPath)

    # Load word frequency
    word_count_path = args.word_frequency
    with open(word_count_path, 'rb') as f:
        word_count = pickle.load(f)
    # with open('../word_frequency/counter_Tokens.p', 'rb') as f:
    #     word_count = pickle.load(f)

    stopword_list1 = set(stopwords.words('english'))
    with open(args.stopwords, "r") as f:
        stopword_list2 = set(eval(f.read()))
    stopword_list = stopword_list1.union(stopword_list2)

    ps = PorterStemmer()

    SS = []
    substitution_words = []
    source_words = []

    num_selection = args.num_selections

    window_context = 11

    if args.do_eval and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Read dataset
        origin, sent_id, eval_examples, mask_words, CHV_selections, CHV_substitutions, CHV_sim_scores = read_df(args.eval_dir)
        print(sent_id)
        print(eval_examples)

        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_examples))
        # logger.info("  Batch size = %d", args.eval_batch_size)

        # Put the model in "evaluation" mode, meaning feed-forward operation.
        model.eval()

        eval_size = len(eval_examples)

        for i in tqdm(range(eval_size)):
            substitution_df = []

            # print(f'{origin[i]}, sentence {sent_id[i]}: ')
            print(f'sentence {sent_id[i]}:\n{eval_examples[i]}')

            tokens, words, position = convert_sentence_to_token(
                sentence=eval_examples[i], tokenizer=tokenizer, seq_length=128)
            print("tokens: ", tokens)
            print("words: ", words)
            print("position: ", position)

            assert len(words) == len(position)

            # len_tokens = len(tokens)
            # print("len_tokens: ", len_tokens)
            try:
                mask_index = words.index(mask_words[i].lower()) #use lower case if do_lower_case == True
            except ValueError:
                print(f'"{mask_words[i]}" is not in list of words')
                try:
                    mask_index = words.index(mask_words[i].lower() + "'s")
                except ValueError:
                    print(f'"{mask_words[i]}" + "\'s" is also not in list of words\nThis sentence will be skipped.\n')
                    continue

            mask_position = position[mask_index]
            mask_context = extract_context(words, mask_index, window_context)
            # print("mask_index: ", mask_index)
            # print("mask_position: ", mask_position)
            # print("mask_context: ", mask_context)

            if isinstance(mask_position, list):
                feature = convert_whole_word_to_feature(tokens_a=tokens,
                                                        mask_position=mask_position,
                                                        seq_length=args.max_seq_length,
                                                        tokenizer=tokenizer)
            else:
                feature = convert_token_to_feature(tokens_a=tokens,
                                                   mask_position=mask_position,
                                                   seq_length=args.max_seq_length,
                                                   tokenizer=tokenizer)

            print("feature.tokens: ", feature.tokens)
            # print("feature.input_ids: ", feature.input_ids)
            # print("feature.input_type_ids: ", feature.input_type_ids)
            # print("feature.input_mask: ", feature.input_mask)

            # Convert inputs to PyTorch tensors
            tokens_tensor = torch.tensor([feature.input_ids])
            segments_tensor = torch.tensor([feature.input_type_ids])
            attention_mask = torch.tensor([feature.input_mask])

            # If we have a GPU, put the tensors on cuda
            tokens_tensor = tokens_tensor.to('cuda')
            segments_tensor = segments_tensor.to('cuda')
            attention_mask = attention_mask.to('cuda')

            # Predict all tokens
            with torch.no_grad():
                output = model(tokens_tensor,
                               token_type_ids=segments_tensor,
                               attention_mask=attention_mask)
                prediction_scores = output[0]
                # print("predictions: ", prediction_scores)

            if isinstance(mask_position, list):
                predicted_top = prediction_scores[0, mask_position[0]].topk(40)
            else:
                predicted_top = prediction_scores[0, mask_position].topk(40)

            pre_tokens = tokenizer.convert_ids_to_tokens(predicted_top[1].cpu().numpy())
            print("pre_tokens: ", pre_tokens)
            pre_prob_values = predicted_top[0].cpu().numpy()
            print("pre_prob_values: ", pre_prob_values)

            ss = substitution_selection(source_word=mask_words[i],
                                        pre_tokens=pre_tokens,
                                        pre_scores=pre_prob_values,
                                        stopwords=stopword_list,
                                        ps=ps,
                                        num_selection=num_selection)

            # SS.append(ss)
            # source_words.append(mask_words[i])

            pre_word = substitution_ranking(source_word=mask_words[i],
                                            source_context=mask_context,
                                            substitution_selection=ss,
                                            fasttext_dico=fasttext_dico,
                                            fasttext_emb=fasttext_emb,
                                            word_count=word_count,
                                            tokenizer=tokenizer,
                                            maskedLM=model)

            MLM_sim_score = fuzzy_match(mask_words[i], pre_word)

            print('---------------------------------------')
            print("Sentence: ", eval_examples[i])
            print("Source word: ", mask_words[i])
            print("Substitution selection: ", ss)
            print("Model substitution: ", pre_word)
            print("Model sim score: ", MLM_sim_score)
            print("CHV substitution: ", CHV_substitutions[i])
            print("CHV sim score: ", CHV_sim_scores[i])
            print(" ")

            # substitution_words.append(pre_word)

            substitution_df.append(OrderedDict({"origin": origin[i],
                                                "sent_id": sent_id[i],
                                                "sentence": eval_examples[i],
                                                "source_term": mask_words[i],
                                                "CHV_selection": CHV_selections[i],
                                                "CHV_substitution": CHV_substitutions[i],
                                                "CHV_sim_score": CHV_sim_scores[i],
                                                "MLM_selection": ss,
                                                "MLM_substitution": pre_word,
                                                "MLM_sim_score": MLM_sim_score
                                                }))

            save_output(args.output_path, substitution_df)

if __name__ == "__main__":
    main()
