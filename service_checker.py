from datetime import datetime
import math
import numpy as np
import random
from utils \
    import is_good, is_get_good, send_mail, test_bern_get, test_bern_post, query

FROM_GMAIL_ADDR = 'YOUR_GMAIL_ADDR'
FROM_GMAIL_ACCOUNT_PASSWORD = 'YOUR_GMAIL_PASSWORD'
TO_EMAIL_ADDR = 'TO_EMAIL_ADDR'


def check_bern(from_gmail, to_email, from_google_account, from_google_password):
    results = list()

    # 0. raw text
    results.append(is_good())

    # 1. pmid, json
    results.append(is_get_good(29446767, 'json', 3, 10))

    # 2. pmid, pubtator
    results.append(is_get_good(29446767, 'pubtator', 3, 10))

    # 3. mutiple pmid
    results.append(is_get_good([29446767, 25681199], 'json', 4, 32))

    acceptables = ['success', 'tmtool error']

    problems = list()
    for ridx, r in enumerate(results):

        if r in acceptables:
            continue

        problems.append('{}: {}'.format(ridx, r))

    if len(problems) == 0:
        print(datetime.now(), 'No problem')
    else:
        problems_total = ', '.join(problems)
        print(datetime.now(), 'Found', problems_total)
        send_mail(from_gmail, to_email,
                  '[BERN] Error(s) {}'.format(problems_total),
                  '\n'.join(problems),
                  from_google_account, from_google_password)


def benchmark(tries, batch_size=None, log_interval=100):
    mutation_times = list()
    ner_times = list()
    normalization_times = list()
    total_times = list()

    pmids = random.sample(range(0, 31113013), tries)
    print('pmids[:10]', pmids[:min(10, tries)])

    if batch_size is not None:
        batch_pmids = list()
        num_batches = math.ceil(len(pmids) / batch_size)

        for i in range(num_batches):

            # last
            if i == num_batches - 1:
                batch_pmids.append(pmids[i * batch_size:])
            else:
                batch_pmids.append(pmids[i * batch_size:(i+1) * batch_size])

        pmids = batch_pmids

    num_na = 0
    num_not_list = 0
    num_not_dict = 0
    ooi_list = list()
    num_error_dict = dict()
    with open('benchmark.tsv', 'w', encoding='utf-8') as f:
        for pidx, pmid in enumerate(pmids):
            res_dict_list = query(pmid)
            if type(res_dict_list) is not list:
                print('not list', pmid, sep='\t')
                num_not_list += 1
                continue

            if type(res_dict_list[0]) is not dict:
                print('not dict', pmid, sep='\t')
                num_not_dict += 1
                continue

            if 'text' in res_dict_list[0]:
                if 'out of index range' in res_dict_list[0]['text']:
                    ooi_list.append(pmid)
                    print('out of index range', pmid, sep='\t')
                elif 'BioC.key' in res_dict_list[0]['text']:
                    num_na += 1
                    # print(res_dict_list[0]['text'], pmid, sep='\t')
                elif 'error: ' in res_dict_list[0]['text'] \
                        and 'elapsed_time' not in res_dict_list[0]:
                    if res_dict_list[0]['text'] in num_error_dict:
                        num_error_dict[res_dict_list[0]['text']] += 1
                    else:
                        num_error_dict[res_dict_list[0]['text']] = 1

            if 'elapsed_time' not in res_dict_list[0]:
                # print('no elapsed_time', pmid, sep='\t')
                continue

            elapsed_time_dict = res_dict_list[0]['elapsed_time']
            mutation_times.append(elapsed_time_dict['tmtool'])
            ner_times.append(elapsed_time_dict['ner'])
            normalization_times.append(elapsed_time_dict['normalization'])
            total_times.append(elapsed_time_dict['total'])

            valid_results = len(mutation_times)

            if pidx > 0 and (pidx + 1) % log_interval == 0:
                print(datetime.now(), '{}/{}'.format(pidx + 1, tries),
                      '#valid_results', valid_results, '#N/A', num_na,
                      '#not_list', num_not_list, '#not_dict', num_not_dict,
                      '#ooi', len(ooi_list), ooi_list, '#err', num_error_dict)

            if valid_results > 0 and valid_results % log_interval == 0:
                print(datetime.now(), '#valid_results', valid_results)
                mutation_res = \
                    '\t'.join(['{:.3f}'.format(v)
                               for v in get_stats(mutation_times,
                                                  batch_size=batch_size)])
                ner_res = \
                    '\t'.join(['{:.3f}'.format(v)
                               for v in get_stats(ner_times,
                                                  batch_size=batch_size)])
                normalization_res = \
                    '\t'.join(['{:.3f}'.format(v)
                               for v in get_stats(normalization_times,
                                                  batch_size=batch_size)])
                total_res = \
                    '\t'.join(['{:.3f}'.format(v)
                               for v in get_stats(total_times,
                                                  batch_size=batch_size)])
                print(valid_results, 'mutation', mutation_res, sep='\t')
                print(valid_results, 'ner', ner_res, sep='\t')
                print(valid_results, 'normalization', normalization_res,
                      sep='\t')
                print(valid_results, 'total', total_res, sep='\t')
                f.write('{}\t{}\t{}\n'.format(valid_results, 'mutation NER',
                                              mutation_res))
                f.write('{}\t{}\t{}\n'.format(valid_results, 'NER',
                                              ner_res))
                f.write('{}\t{}\t{}\n'.format(valid_results, 'normalization',
                                              normalization_res))
                f.write('{}\t{}\t{}\n'.format(valid_results, 'total',
                                              total_res))
                f.flush()

    print('#valid_results', len(mutation_times))
    print('mutation',
          '\t'.join(['{:.3f}'.format(v)
                     for v in get_stats(mutation_times,
                                        batch_size=batch_size)]), sep='\t')
    print('ner',
          '\t'.join(['{:.3f}'.format(v)
                     for v in get_stats(ner_times,
                                        batch_size=batch_size)]), sep='\t')
    print('normalization',
          '\t'.join(['{:.3f}'.format(v)
                     for v in get_stats(normalization_times,
                                        batch_size=batch_size)]), sep='\t')
    print('total',
          '\t'.join(['{:.3f}'.format(v)
                     for v in get_stats(total_times,
                                        batch_size=batch_size)]), sep='\t')


def get_stats(lst, batch_size=None):
    if not lst:
        return None

    if batch_size is None:
        return sum(lst) / len(lst), np.std(lst), min(lst), max(lst)
    else:
        return (sum(lst) / len(lst)) / batch_size, \
               np.std(lst), min(lst) / batch_size, max(lst) / batch_size


def stress_test(num_threads, wait_seconds, num_try):
    test_bern_get(num_threads, wait_seconds, num_try)
    test_bern_post('CLAPO syndrome: identification of somatic activating '
                   'PIK3CA mutations and delineation of the natural history '
                   'and phenotype. Purpose CLAPO syndrome is a rare vascular '
                   'disorder characterized by capillary malformation of the '
                   'lower lip, lymphatic malformation predominant on the face'
                   ' and neck, asymmetry and partial/generalized overgrowth. '
                   'Here we tested the hypothesis that, although the genetic '
                   'cause is not known, the tissue distribution of the '
                   'clinical manifestations in CLAPO seems to follow a '
                   'pattern of somatic mosaicism. Methods We clinically '
                   'evaluated a cohort of 13 patients with CLAPO and screened'
                   ' 20 DNA blood/tissue samples from 9 patients using '
                   'high-throughput, deep sequencing. Results We identified '
                   'five activating mutations in the PIK3CA gene in affected '
                   'tissues from 6 of the 9 patients studied; one of the '
                   'variants (NM_006218.2:c.248T>C; p.Phe83Ser) has not been '
                   'previously described in developmental disorders. '
                   'Conclusion We describe for the first time the presence '
                   'of somatic activating PIK3CA mutations in patients with '
                   'CLAPO. We also report an update of the phenotype and '
                   'natural history of the syndrome.',
                   num_threads, wait_seconds, num_try)


if __name__ == '__main__':
    check_bern(FROM_GMAIL_ADDR, TO_EMAIL_ADDR,
               FROM_GMAIL_ADDR, FROM_GMAIL_ACCOUNT_PASSWORD)
