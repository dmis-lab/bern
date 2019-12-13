import pandas as pd
from collections import Counter

BEST_PLUS_DICT_PATH = 'dictionary/best_dict_Gene.txt'
BEST_DICT_PATH = 'BEST_v1_dict/WordNetChecked_bossNewDic_2016-03-23.txt'

def find_best_plus_largest_index(best_plus_dict_path):
    best_plus_dict = pd.read_csv(best_plus_dict_path, sep='\|\|', header=None, index_col=0)
    largest_index = max([index // 100 for index in best_plus_dict.index]) # index // 100 : delete last 2 digit because that is type code
    return largest_index

def find_largest_index(uid_list):
    return max([uid // 100 for uid in uid_list])

def make_best_dict(best_dict_path, start_index, type_code=2):
    best_dict = dict()
    registered_id = set()
    with open(best_dict_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('\t')
            best_id, mention, mention_type = line[0], line[1], line[2]
            if mention_type in ['Gene', 'Target']:
                registered_id.add(best_id)
                uid = (start_index + (len(registered_id)-1)) * 100 + type_code
                if uid not in best_dict:
                    best_dict[uid] = []
                best_dict[uid].append(mention)
    return best_dict

def write_auxiliary_dict(aux_dict, output_path):
    with open(output_path, 'w', encoding='utf-8') as f:
        for uid, mentions in aux_dict.items():
            for mention in mentions:
                f.write('{}||{}\n'.format(uid, mention))

def load_auxiliary_dict(path):
    auxiliary_dict = dict()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.split('||')
            uid, mention = line[0].strip(), line[1].strip()
            assert mention not in auxiliary_dict
            auxiliary_dict[mention] = uid
    return auxiliary_dict

def make_freq_dict(total_sample_path, start_index, threshold, type_code=2):
    freq_dict = dict()
    registered_mention = set()
    total_df = pd.read_csv(total_sample_path, sep='\t', index_col=None, header=0)
    gene_df = total_df.query('entity_type == "gene"')
    gene_cuiless_df = gene_df.query('entity_id == "CUI-less"')
    mention_counter = Counter(gene_cuiless_df['mention'])
    top_mentions = [mention for mention, freq in mention_counter.items() if freq >= threshold]
    for mention in top_mentions:
        registered_mention.add(mention)
        uid = (start_index + len(registered_mention) - 1) * 100 + type_code
        if uid not in freq_dict:
            freq_dict[uid] = []
        freq_dict[uid].append(mention)
    return freq_dict

if __name__ == '__main__':
    # make best dict (not best plus dict)
    BEST_DICT_OUT_PATH = 'dictionary/best_dict_Gene_oldbest.txt'
    FREQ_DICT_OUT_PATH = 'dictionary/best_dict_Gene_freq.txt'
    TOTAL_SAMPLE_PATH  = 'C:/Users/HS/Downloads/gene_drug_disease_species_normalization_result_sample100p.tsv'
    start_index = find_best_plus_largest_index(BEST_PLUS_DICT_PATH) + 100
    best_dict = make_best_dict(BEST_DICT_PATH, start_index)
    write_auxiliary_dict(best_dict, BEST_DICT_OUT_PATH)
    start_index = find_largest_index(best_dict.keys()) + 100
    freq_dict = make_freq_dict(TOTAL_SAMPLE_PATH, start_index, threshold=50)
    write_auxiliary_dict(freq_dict, FREQ_DICT_OUT_PATH)