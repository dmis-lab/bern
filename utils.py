from email.mime.text import MIMEText
import random
import requests
import smtplib
import subprocess
import time
import threading
from convert import pubtator2pubannotation


def query_raw(text, url='https://bern.korea.ac.kr/plain'):
    return requests.post(url, data={'sample_text': text}).json()


def query(pmid, url='https://bern.korea.ac.kr/pubmed', output_format='json',
          verbose=False):
    res = None
    if type(pmid) is str or type(pmid) is int:
        res = requests.get('{}/{}/{}'.format(url, pmid, output_format))
    elif type(pmid) is list:
        if len(pmid) == 0:
            print('No pmid')
            return res

        pmid = [str(p) for p in pmid if type(p) is not str]
        res = requests.get('{}/{}/{}'.format(url, ','.join(pmid),
                                             output_format))

    if verbose:
        print('pmid', pmid, 'result', res.text)

    if output_format == 'pubtator':
        return res.text
    return res.json()


def test_bern_get(num_thread, period_delay_seconds, tries,
                  url='https://bern.korea.ac.kr/pubmed'):
    for _ in range(tries):
        pmids = random.sample(range(0, 30400000), num_thread)

        print(url, pmids)

        threads = list()
        for pmid in pmids:
            t = threading.Thread(target=query,
                                 args=(pmid, url, 'json', True))
            t.daemon = True
            t.start()
            threads.append(t)

        # block until all tasks are done
        for t in threads:
            t.join()

        time.sleep(period_delay_seconds)


def test_bern_post(base_text, num_thread, period_delay_seconds, tries,
                   url='https://bern.korea.ac.kr/plain'):
    for _ in range(tries):
        random_numbers = random.sample(range(0, 99999999), num_thread)

        print(url, random_numbers)

        threads = list()
        for random_number in random_numbers:
            t = threading.Thread(target=query_raw,
                                 args=('{} {}'.format(base_text, random_number),
                                       url))
            t.daemon = True
            t.start()
            threads.append(t)

        # block until all tasks are done
        for t in threads:
            t.join()

        time.sleep(period_delay_seconds)


# Ref.
# https://www.endpoint.com/blog/2015/01/28/getting-realtime-output-using-python
def run_command(command, cwd):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, cwd=cwd)
    while True:
        output = process.stdout.readline().decode('utf-8')
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
    rc = process.poll()
    return rc


def ps_grep(q):
    ps_process = subprocess.Popen(["ps", "aux"], stdout=subprocess.PIPE)
    grep_process = subprocess.Popen(["egrep", q], stdin=ps_process.stdout,
                                    stdout=subprocess.PIPE)
    ps_process.stdout.close()
    output = grep_process.communicate()[0]
    return output.decode('utf-8')


def get_bern_status(ps_grep_res):
    status = 0
    off_list = list()

    # mutation taggers
    mutation_taggers = ['tmVar2Server.jar', 'GNormPlusServer.jar']
    for mt in mutation_taggers:
        if mt in ps_grep_res:
            status += 1
        else:
            off_list.append(mt)

    # normalizers
    normalizers = ['GNormPlus_180921.jar',
                   'disease_normalizer_181030.jar', 'chemical_normalizer.py',
                   'mutation_normalizer.py', 'species_normalizer.py']
    for norm in normalizers:
        if norm in ps_grep_res:
            status += 1
        else:
            off_list.append(norm)

    # back-end and front-end
    be_fe = ['python3 -u server.py', 'node bern_server.js']
    for bf in be_fe:
        if bf in ps_grep_res:
            status += 1
        else:
            off_list.append(bf)

    return status, off_list


def send_mail(from_addr, to, subject, content, gmail_id, password):
    msg = MIMEText(content)
    msg['Subject'] = subject
    msg['From'] = from_addr
    msg['To'] = to

    # Send the message via our own SMTP server.
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.ehlo()
    s.starttls()
    s.ehlo()
    s.login(gmail_id, password)
    s.send_message(msg)
    print('Mail has been sent')
    s.quit()


def is_good(num_type_set=3, normal_id_cnt=13):
    try:
        # A part of PMID:29446767
        post_res = query_raw('{} {}'.format(
            'CLAPO syndrome: identification of somatic activating PIK3CA '
            'mutations '
            'and delineation of the natural history and phenotype. Purpose '
            'CLAPO '
            'syndrome is a rare vascular disorder characterized by capillary '
            'malformation of the lower lip, lymphatic malformation predominant '
            'on the face and neck, asymmetry, and partial/generalized '
            'overgrowth. '
            'Here we tested the hypothesis that, although the genetic cause is '
            'not known, the tissue distribution of the clinical manifestations '
            'in CLAPO seems to follow a pattern of somatic mosaicism.'
            ' Methods We '
            'clinically evaluated a cohort of 13 patients with CLAPO and '
            'screened '
            '20 DNA blood/tissue samples from 9 patients using '
            'high-throughput, '
            'deep sequencing. Results We identified five activating mutations '
            'in the PIK3CA gene in affected tissues from 6 of the 9 patients '
            'studied; one of the variants (NM_006218.2:c.248T>C; p.Phe83Ser) '
            'has not been previously described in developmental disorders. '
            'Conclusion We describe for the first time the presence of somatic '
            'activating PIK3CA mutations in patients with CLAPO. We also '
            'report '
            'an update of the phenotype and natural history of the syndrome. '
            'Imatinib is a asdf of homo sapiens ...', 2019.8))
    except requests.exceptions.ConnectionError:
        return 'ConnectionError'

    if 'denotations' not in post_res:
        print('No denotations')
        return 'No denotations'

    # all entity types
    id_cnt = 0
    type_set = set()
    for d in post_res['denotations']:
        type_set.add(d['obj'])
        if 'CUI-less' == d['id'][0]:
            continue
        id_cnt += 1

    if len(type_set) != num_type_set:
        print('Found an NER problem:', num_type_set, '!=', len(type_set))
        return 'NER problem #types: {} != {}'.format(num_type_set,
                                                     len(type_set))

    if id_cnt != normal_id_cnt:
        print('Found a normalization problem:', normal_id_cnt, '!=', id_cnt)
        return 'Normalization problem #norm. ids: {} != {}'.format(
            normal_id_cnt, id_cnt)

    return 'success'


def is_get_good(pmid, output_format, num_type_set, normal_id_cnt):
    get_res = query(pmid, url='https://bern.korea.ac.kr/pubmed',
                    output_format=output_format, verbose=False)

    if output_format.lower() == 'pubtator':
        get_res = pubtator2pubannotation(get_res)

    if type(get_res) is str:
        return get_res

    # get_res should be a list
    if type(get_res) is not list:
        return 'no list: {}'.format(type(get_res))

    if not get_res:
        return 'no result'

    if 'error: tmtool:' in get_res[0]['text']:
        return 'tmtool error'

    id_cnt = 0
    type_set = set()

    for gr in get_res:
        if 'denotations' not in gr:
            if 'sourceid' in gr:
                print('No denotations, sourceid:', gr['sourceid'])
            else:
                print('No denotations, gr:', gr)
            return 'pmid: {}, No denotations {}'.format(pmid, gr)

        for d in gr['denotations']:
            type_set.add(d['obj'])
            if 'CUI-less' == d['id'][0]:
                continue
            id_cnt += 1

        # print(gr['sourceid'], '#types', len(type_set))
        # print(gr['sourceid'], '#ids', id_cnt)

    if len(type_set) != num_type_set:
        print('Found an NER problem #types: {} != {}'.format(
            num_type_set, len(type_set)))
        return 'pmid: {}, {}, NER problem,  #types: {} != {}'.format(
            pmid, output_format.lower(), num_type_set, len(type_set))

    if id_cnt != normal_id_cnt:
        print('Found a normalization problem: got', id_cnt,
              'expected', normal_id_cnt)
        return 'pmid: {}, {}, normalizer problem, got {} expected {}'.format(
            pmid, output_format.lower(), id_cnt, normal_id_cnt)

    return 'success'


# Ref. dict of SR4GN
species_human_excl_homo_sapiens = \
    'person|infant|Child|people|participants|woman|' \
    'Girls|Man|Peoples|Men|Participant|Patients|' \
    'humans|Persons|mans|participant|Infants|Boys|' \
    'Human|Humans|Women|children|Mans|child|Participants|Girl|' \
    'Infant|girl|patient|patients|boys|men|infants|' \
    'man|girls|Children|Boy|women|persons|human|Woman|' \
    'peoples|Patient|People|boy|Person'.split('|')


def filter_entities(ner_results, is_raw_text):
    num_filtered_species_per_doc = list()

    for idx, paper in enumerate(ner_results):

        if is_raw_text:
            content = paper['abstract']
        else:
            if len(paper['abstract']) > 0:
                content = paper['title'] + ' ' + paper['abstract']
            else:
                content = paper['title']

        valid_species = list()
        species = paper['entities']['species']
        for spcs in species:
            entity_mention = content[spcs['start']:spcs['end']+1]
            if entity_mention in species_human_excl_homo_sapiens:
                spcs['end'] += 1
                continue
            valid_species.append(spcs)

        num_filtered_species = len(species) - len(valid_species)
        if num_filtered_species > 0:
            paper['entities']['species'] = valid_species

        num_filtered_species_per_doc.append((paper['pmid'],
                                             num_filtered_species))

    return num_filtered_species_per_doc
