import ftplib
import html
import json
import os
import requests
import ssl
import tarfile
import time
import urllib.request
from urllib.parse import urlparse
import xmltodict
from collections import OrderedDict


def get_pubmed_xml(pmid, output_path=None):
    # https://www.codementor.io/aviaryan/downloading-files-from-urls-in-python-77q3bs0un
    try:
        r = requests.get('https://www.ncbi.nlm.nih.gov/pubmed/'
                         '{}?report=xml&format=text'.format(pmid),
                         allow_redirects=True)
    except requests.exceptions.ConnectionError as ce:
        print('ConnectionError', pmid, ce)
        return None

    text = r.text
    text = html.unescape(text)

    if '<PMID' not in text and '</PMID>' not in text:
        print('Not found pmid:', pmid)
        return None

    real_output_path = './pubmed/{}.xml'.format(pmid)
    if output_path is not None:
        real_output_path = output_path

    if not os.path.exists(os.path.dirname(real_output_path)):
        os.makedirs(os.path.dirname(real_output_path))

    with open(real_output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    return real_output_path


def pubmed_xml2pubtator(pmid, xml_path, output_path=None):
    # print(xml_path)
    with open(xml_path, 'r', encoding='utf-8') as fd:
        doc = xmltodict.parse(fd.read())
    pre = doc['pre']

    if 'PubmedArticle' in pre:
        article = pre['PubmedArticle']['MedlineCitation']['Article']
    elif 'PubmedBookArticle' in pre:
        article = pre['PubmedBookArticle']['BookDocument']
    else:
        raise ValueError(pmid)

    title = article.get('ArticleTitle')
    if title is None:
        if 'PubmedBookArticle' in pre:
            title = \
                pre['PubmedBookArticle']['BookDocument']['Book']['BookTitle']
    if type(title) is OrderedDict:
        title = title['#text']

    abstract_zone = article.get('Abstract')
    if abstract_zone is None:
        print('No abstract pmid:', pmid)
        return None

    abstract = ''
    for abstract_text in abstract_zone:
        if abstract_text == 'CopyrightInformation':
            continue

        abst_element = abstract_zone[abstract_text]

        if type(abst_element) is str:
            if len(abstract) > 0:
                abstract += ' '

            abstract += abst_element
        elif type(abst_element) is list:
            for od in abst_element:
                if type(od) is str:
                    if len(abstract) > 0:
                        abstract += ' '
                    abstract += od
                elif type(od) is OrderedDict:
                    if '#text' in od:
                        if len(abstract) > 0:
                            abstract += ' '
                        abstract += od['#text']
                else:
                    print('Unknown abstract element type', type(od))
                    if len(abstract) > 0:
                        abstract += ' '
                    abstract += str(od)

    abstract = replace_wspaces(abstract)

    real_output_path = './pubmed_pubtator/{}.PubTator'.format(pmid)
    if output_path is not None:
        real_output_path = output_path
    if not os.path.exists(os.path.dirname(real_output_path)):
        os.makedirs(os.path.dirname(real_output_path))

    with open(real_output_path, 'w', encoding='utf-8') as f_pub:
        f_pub.write('{}|t|{}\n'.format(pmid, title))
        f_pub.write('{}|a|{}\n'.format(pmid, abstract))
        f_pub.write('\n')
    print('Saved {}'.format(real_output_path))

    return real_output_path


def get_pmc_xml(pmcid, output_path=None):
    # get a ftp address from
    # https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=<pmcid>
    r = requests.get('https://www.ncbi.nlm.nih.gov/pmc/'
                     'utils/oa/oa.fcgi?id={}'.format(pmcid),
                     allow_redirects=True)
    text = r.text

    real_output = './pmc/{}.xml'.format(pmcid)
    if output_path is not None:
        real_output = output_path

    if not os.path.exists(os.path.dirname(real_output)):
        os.makedirs(os.path.dirname(real_output))

    doc = xmltodict.parse(text)
    oa = doc['OA']
    if 'records' not in oa:
        print('Not found pmcid:', pmcid)
        return None

    with open(real_output, 'w') as f:
        f.write(text)

    xml_path = None
    real_output_dir = os.path.dirname(real_output)
    records = oa['records']
    record = records['record']
    link = record['link']
    href = None
    if type(link) is OrderedDict:
        href = link['@href']
    elif type(link) is list:
        for l in link:
            if l['@format'] != 'tgz':
                continue
            href = l['@href']

    if href is None:
        print('Not found href: pmcid:', pmcid)
        return None

    print('Download', href)
    tar_filename = download_ftp(href, real_output_dir)

    tar = tarfile.open('./pmc/{}'.format(tar_filename))
    tar.extractall(path='./pmc/')
    tar.close()

    # retain only .nxml to save disk space
    tar_dir = tar_filename.replace('.tar.gz', '')
    for f in os.listdir('./pmc/{}'.format(tar_dir)):
        f_path = os.path.join(real_output_dir, tar_dir, f)
        fname, ext = os.path.splitext(f)
        if ext != '.nxml':
            os.remove(f_path)
        else:
            xml_path = f_path

    return xml_path


def pmc_xml2pubtator(pmcid, xml_path, output_path=None):
    with open(xml_path, 'r') as fd:
        doc = xmltodict.parse(fd.read())

    article = doc['article']
    front = article['front']
    title = front['article-meta']['title-group']['article-title']

    body_text = ''

    if 'abstract' in front['article-meta']:
        abstract = front['article-meta']['abstract']
        if 'p' in abstract:
            if type(abstract['p']) is str:
                if len(body_text) > 0:
                    body_text += ' '
                body_text += abstract['p']
            elif type(abstract['p']) is OrderedDict:
                if '#text' in abstract['p']:
                    if len(body_text) > 0:
                        body_text += ' '
                    body_text += abstract['p']['#text']
                else:
                    print('No #text abstract p:', pmcid)
            elif type(abstract['p']) is list:
                for ap in abstract['p']:
                    if type(ap) is OrderedDict:
                        if '#text' in ap:
                            if len(body_text) > 0:
                                body_text += ' '
                            body_text += ap['#text']
                        else:
                            print('No #text abstract p list', pmcid)
                    elif type(ap) is str:
                        if len(body_text) > 0:
                            body_text += ' '
                        body_text += ap
            else:
                print('Unknown abstract type:', type(abstract['p']), pmcid)

        if 'sec' in abstract:
            if type(abstract['sec']) is list:
                for sec in abstract['sec']:
                    if 'p' in sec:
                        if type(sec['p']) is OrderedDict:
                            if '#text' in sec['p']:
                                if len(body_text) > 0:
                                    body_text += ' '
                                body_text += sec['p']['#text']
                            else:
                                print('No #text abst sec p:', pmcid)
                        elif type(sec['p']) is str:
                            if len(body_text) > 0:
                                body_text += ' '
                            body_text += sec['p']
                        else:
                            print('No #text abst sec p:', pmcid)
            elif type(abstract['sec']) is OrderedDict:
                if 'p' in abstract['sec']:
                    if len(body_text) > 0:
                        body_text += ' '
                    body_text += abstract['sec']['p']
                else:
                    print('No p in abstract sec:', pmcid)

        # PMC2930000
        if type(abstract) is list:
            for abst in abstract:
                if 'p' in abst:
                    if type(abst['p']) is str:
                        if len(body_text) > 0:
                            body_text += ' '
                        body_text += abst['p']
                    elif type(abst['p']) is OrderedDict:
                        if '#text' in abst['p']:
                            if len(body_text) > 0:
                                body_text += ' '
                            body_text += abst['p']['#text']
                        elif 'list' in abst['p']:
                            abst_p_lst = abst['p']['list']
                            if type(abst_p_lst) is OrderedDict:
                                if 'list-item' in abst_p_lst:
                                    for abst_p_lst_e in abst_p_lst['list-item']:
                                        if len(body_text) > 0:
                                            body_text += ' '
                                        body_text += abst_p_lst_e['p']

                        else:
                            print('No #text abst list p:', pmcid)
                    elif type(abst['p']) is list:
                        # PMC5050000
                        pass
                    else:
                        print('Unknown type', type(abst['p']), pmcid)
                elif 'sec' in abst:
                    if type(abst['sec']) is list:
                        for abstsec in abst['sec']:
                            if type(abstsec) is OrderedDict:
                                if 'p' in abstsec:
                                    if type(abstsec['p']) is str:
                                        if len(body_text) > 0:
                                            body_text += ' '
                                        body_text += abstsec['p']
                                    elif type(abstsec['p']) is OrderedDict:
                                        if '#text' in abstsec['p']:
                                            if len(body_text) > 0:
                                                body_text += ' '
                                            body_text += abstsec['p']['#text']
                                        else:
                                            print('No #text abstsec p:',
                                                  pmcid)
        if 'p' not in abstract and 'sec' not in abstract \
                and len(body_text) == 0:
            print('No sec and p abstract', pmcid)

    if 'body' in article:
        # PMC2500000 (WIP)
        if 'sec' in article['body']:
            if type(article['body']['sec']) is list:
                for sec in article['body']['sec']:
                    if 'p' in sec:
                        for p in sec['p']:
                            # PMC3600000
                            if type(p) is OrderedDict:
                                if '#text' in p:
                                    if len(body_text) > 0:
                                        body_text += ' '
                                    body_text += p['#text']
                                else:
                                    if 'ext-link' not in p \
                                            and 'disp-formula' not in p \
                                            and 'bold' not in p \
                                            and 'table-wrap' not in p \
                                            and 'fig' not in p:
                                        print('No sec p #text', type(p), pmcid)
                            elif type(p) is str:
                                if len(body_text) > 0:
                                    body_text += ' '
                                body_text += p
                            else:
                                print('Unknown type', type(p), pmcid)
                    elif 'sec' in sec:
                        if type(sec['sec']) is list:
                            for ss in sec['sec']:
                                if 'p' in ss:
                                    if type(ss['p']) is OrderedDict:
                                        if '#text' in ss['p']:
                                            if len(body_text) > 0:
                                                body_text += ' '
                                            body_text += ss['p']['#text']
                                        else:
                                            print('No ss p #text', pmcid)
                                    elif type(ss['p']) is list:
                                        for ssp in ss['p']:
                                            if type(ssp) is OrderedDict:
                                                if '#text' in ssp:
                                                    if len(body_text) > 0:
                                                        body_text += ' '
                                                    body_text += ssp['#text']
                                                elif 'table-wrap' in ssp:
                                                    pass
                                                elif 'fig' in ssp:
                                                    pass
                                                else:
                                                    print('No ssp #text', pmcid)
                                            elif type(ssp) is str:
                                                if len(body_text) > 0:
                                                    body_text += ' '
                                                body_text += ssp
                                            else:
                                                print('Unknown ssp type:',
                                                      type(ssp), pmcid)
                        elif type(sec['sec']) is OrderedDict:
                            # PMC3790000
                            if 'sec' in sec['sec']:
                                if type(sec['sec']['sec']) is OrderedDict:
                                    # TODO
                                    pass
                        else:
                            print('Unknown sec type:', type(sec['sec']), pmcid)
                    elif 'supplementary-material' in sec:
                        # PMC1940000
                        print('No sec and p but supplementary-material', pmcid)
                        # TODO supp
                    else:
                        print('Unknown type: No sec, p, supp', pmcid)
            elif type(article['body']['sec']) is OrderedDict:
                if 'sec' in article['body']['sec']:
                    if type(article['body']['sec']['sec']) is list:
                        for bss in article['body']['sec']['sec']:
                            if 'p' in bss:
                                if type(bss['p']) is list:
                                    for bssp in bss['p']:
                                        if '#text' in bssp:
                                            if len(body_text) > 0:
                                                body_text += ' '
                                            body_text += bssp['#text']
                                        else:
                                            print('No bssp #text', pmcid)

        elif 'p' in article['body']:
            paragraphs = article['body']['p']
            for p in paragraphs:
                if type(p) is OrderedDict:
                    if '#text' in p:
                        p_text = p['#text']
                    else:
                        print('No p #text', type(p), pmcid)
                        p_text = ''
                elif type(p) is str:
                    p_text = p
                else:
                    print('Unknown type', type(p), pmcid)
                    p_text = str(p)

                if len(body_text) > 0 and len(p_text) > 0:
                    body_text += ' '
                body_text += p_text
        elif 'supplementary-material' in article['body']:
            # PMC1140000, PMC1520000, PMC 2010000
            print('No sec and p but supplementary-material', pmcid)
        else:
            print('Unknown doc type', pmcid)
    else:
        # PMC2200000
        print('Unknown doc type: no body', pmcid)

    body_text = replace_wspaces(body_text)

    if len(title) == 0:
        print('Empty title', pmcid)

    if len(body_text) == 0:
        print('Empty body', pmcid)

    real_output_path = './pmc_pubtator/{}.PubTator'.format(pmcid)
    if output_path is not None:
        real_output_path = output_path
    if not os.path.exists(os.path.dirname(real_output_path)):
        os.makedirs(os.path.dirname(real_output_path))

    with open(real_output_path, 'w', encoding='utf-8') as f_pub:
        f_pub.write('{}|t|{}\n'.format(pmcid, title))
        f_pub.write('{}|a|{}\n'.format(pmcid, body_text))
        f_pub.write('\n')
    print('Saved {}'.format(real_output_path))


def get_pmc_archive(non_comm_use_pdf, non_comm_output_dir,
                    comm_use_file_list, comm_output_dir, overwrite=False,
                    n_threads=8):
    import threading
    from queue import Queue

    # ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_non_comm_use_pdf.txt
    # ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_comm_use_file_list.txt

    if not os.path.exists(non_comm_output_dir):
        os.makedirs(non_comm_output_dir)

    if not os.path.exists(comm_output_dir):
        os.makedirs(comm_output_dir)

    num_non_comm = 0
    num_comm = 0

    # start_t = time.time()

    ftp = ftplib.FTP('ftp.ncbi.nlm.nih.gov')
    ftp.login()

    #
    def worker():
        while True:
            item = q.get(block=True, timeout=None)
            if item is None:
                break
            download_ftp(item[0], item[1], overwrite=overwrite)
            q.task_done()

    q = Queue()
    threads = list()
    for _ in range(n_threads):
        t = threading.Thread(target=worker)
        t.daemon = True
        t.start()
        threads.append(t)

    with open(non_comm_use_pdf, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            cols = line.split()
            if len(cols) < 5:
                continue
            q.put(('ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/' + cols[0],
                   non_comm_output_dir))

    q.join()

    # stop workers
    for i in range(n_threads):
        q.put(None)

    # block until all tasks are done
    for t in threads:
        t.join()

    start_t = time.time()
    with open(comm_use_file_list, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            cols = line.split()
            if len(cols) < 5:
                continue

            href = 'ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/' + cols[0]
            if num_comm % 10 == 0:
                print(idx, num_comm, 'comm Download', href,
                      '{:.2f} doc/sec'.format(
                          num_comm/(time.time()-start_t)))

            pr = urlparse(href)
            # print(pr.netloc, pr.path)
            path2file, filename = os.path.split(pr.path)
            target_file_name = os.path.join(comm_output_dir, filename)
            if not overwrite and os.path.exists(target_file_name):
                continue
            ftp.cwd(path2file)
            with open(target_file_name, 'wb') as fhandle:
                ftp.retrbinary('RETR %s' % filename, fhandle.write,
                               blocksize=8192*4)

            num_comm += 1

    ftp.quit()

    print(num_non_comm)
    print(num_comm)


def download_ftp(url, output_dir, overwrite=True):
    pr = urlparse(url)
    ftp = ftplib.FTP(pr.netloc)
    ftp.login()
    path2file, filename = os.path.split(pr.path)
    ftp.cwd(path2file)
    target_file_name = os.path.join(output_dir, filename)
    if not overwrite and os.path.exists(target_file_name):
        return filename
    with open(target_file_name, 'wb') as fhandle:
        ftp.retrbinary('RETR %s' % filename, fhandle.write)
    return filename


def get_pubmed(pmid, out_format='json', encoding='unicode'):
    # https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pubmed.cgi/BioC_json/17299597/unicode
    try:
        r = requests.get('https://www.ncbi.nlm.nih.gov/research/bionlp/'
                         'RESTful/pubmed.cgi/BioC_{}/{}/{}'
                         .format(out_format, pmid, encoding),
                         allow_redirects=True)
    except requests.exceptions.ConnectionError as ce:
        print('ConnectionError', pmid, ce)
        return None

    return r.text


def query_tmtool(pmids, entity_type='Mutation', out_format='JSON'):
    if type(pmids) is list:
        pmids = ','.join([str(pmid) for pmid in pmids])

    try:
        r = requests.get('https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/'
                         'RESTful/tmTool.cgi/{}/{}/{}/'
                         .format(entity_type, pmids, out_format),
                         allow_redirects=True)
    except requests.exceptions.ConnectionError as ce:
        print('ConnectionError', pmids, ce)
        return None

    return r.text


def query_pubtator2(pmids, out_format='pubtator', escape_html=True):
    if type(pmids) is list:
        pmids = ','.join([str(pmid) for pmid in pmids])

    # https://www.ncbi.nlm.nih.gov/research/bionlp/pubtator2/api/v1/publications/export/pubtator?pmids=29446767,25681199
    try:
        r = requests.get('https://www.ncbi.nlm.nih.gov/research/bionlp/'
                         'pubtator2/api/v1/publications/export/'
                         '{}?pmids={}'
                         .format(out_format, pmids),
                         allow_redirects=True)
    except requests.exceptions.ConnectionError as ce:
        print('ConnectionError', pmids, ce)
        return None

    if escape_html:
        return html.unescape(r.text)
    return r.text


def query_pubtator2biocxml(pmids):
    if type(pmids) is list:
        pmids = ','.join([str(pmid) for pmid in pmids])

    # https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/biocxml?pmids=29446767,25681199
    try:
        r = requests.get('https://www.ncbi.nlm.nih.gov/research/'
                         'pubtator-api/publications/export/'
                         'biocxml?pmids={}'
                         .format(pmids),
                         allow_redirects=True)
    except requests.exceptions.ConnectionError as ce:
        print('ConnectionError', pmids, ce)
        return None

    r.encoding = 'utf-8'
    return html.unescape(r.text), r.text


# Ref.
# https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/RESTfulAPI.client.zip
def query_raw_tmtool(input_str, trigger='tmVar'):
    url_submit = "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/" \
                 "RESTful/tmTool.cgi/" + trigger + "/Submit/"
    urllib_submit = urllib.request.urlopen(url_submit,
                                           input_str.encode('utf-8'))
    # urllib_result = urllib.request.urlopen(url_submit,
    #                                        input_str.encode('utf-8'))
    session_number = urllib_submit.read().decode('utf-8')
    url_receive = "https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/" \
                  "RESTful/tmTool.cgi/" + session_number + "/Receive/"

    urllib_result = None
    try_cnt = 1

    code = 404
    while code == 404 or code == 501:
        time.sleep(0.5)
        print('{}\t{}'.format(try_cnt, url_receive))
        try_cnt += 1
        try:
            urllib_result = urllib.request.urlopen(url_receive)
        except urllib.request.HTTPError as e:
            code = e.code
        except urllib.request.URLError:
            code = urllib_submit.code
        else:
            code = urllib_result.getcode()

    if urllib_result is not None:
        return urllib_result.read().decode('utf-8')

    return None


def query_raw_bern(input_str, bern_post_url='https://bern.korea.ac.kr/plain'):
    input_dict = {'text': input_str}

    # context = ssl._create_unverified_context()

    # http://www.hanul93.com/malwares-api-urllib2/
    ctxt = ssl.create_default_context()
    ctxt.check_hostname = False
    ctxt.verify_mode = ssl.CERT_NONE

    urllib_result = \
        urllib.request.urlopen(bern_post_url,
                               json.dumps(input_dict).encode('utf-8'),
                               context=ctxt)

    return urllib_result.read().decode('utf-8')


def replace_wspaces(t):
    t = t.replace('\r', '')
    t = t.replace('\n', ' ')
    t = t.replace('\t', ' ')
    t = t.replace(' \u2028 ', ' ')
    t = t.replace('\u2029', ' ')
    return t


if __name__ == '__main__':
    from convert import pubtator_biocxml2dict_list

    # testpmids = [25681199, 29446767]
    testpmids = [29446767, 25681199]
    # print(query_pubtator2([29446767,25681199]))
    # print(query_pubtator2([25681199]))
    print(pubtator_biocxml2dict_list(testpmids))

    import sys
    sys.exit(0)

    get_pmc_archive(
        os.path.expanduser('~') + '/bestplus/pmc/oa_non_comm_use_pdf.txt',
        os.path.expanduser('~') + '/bestplus/pmc/non_comm',
        os.path.expanduser('~') + '/bestplus/pmc/oa_comm_use_file_list.txt',
        os.path.expanduser('~') + '/bestplus/pmc/comm'
    )

    from convert import pubtatorstr2dict_list

    # raw texts & demo
    bern_raw_res = query_raw_bern('Results We identified five activating mutations in the PIK3CA gene in affected tissues from 6 of the 9 patients studied; one of the variants (NM_006218.2:c.248T>C; p.Phe83Ser) has not been previously described in developmental disorders.')
    print(bern_raw_res)

    # api
    tmtool_format = 'PubTator'
    tmtool_res = query_tmtool([29446767,25681199], out_format=tmtool_format)
    dict_list = pubtatorstr2dict_list(tmtool_res, is_raw_text=False)
    print(dict_list)

    # # demo
    # tmtool_res = query_raw_tmtool('CLAPO syndrome: identification of somatic activating PIK3CA mutations and delineation of the natural history and phenotype. Purpose CLAPO syndrome is a rare vascular disorder characterized by capillary malformation of the lower lip, lymphatic malformation predominant on the face and neck, asymmetry, and partial/generalized overgrowth. Here we tested the hypothesis that, although the genetic cause is not known, the tissue distribution of the clinical manifestations in CLAPO seems to follow a pattern of somatic mosaicism. Methods We clinically evaluated a cohort of 13 patients with CLAPO and screened 20 DNA blood/tissue samples from 9 patients using high-throughput, deep sequencing. Results We identified five activating mutations in the PIK3CA gene in affected tissues from 6 of the 9 patients studied; one of the variants (NM_006218.2:c.248T>C; p.Phe83Ser) has not been previously described in developmental disorders. Conclusion We describe for the first time the presence of somatic activating PIK3CA mutations in patients with CLAPO. We also report an update of the phenotype and natural history of the syndrome. GENETICS in MEDICINE advance online publication, 15 February 2018; doi:10.1038/gim.2017.200.')
    # dict_list = pubtatorstr2dict_list(tmtool_res, is_raw_text=True)
    # print(dict_list)

    # tmtool_res, tmtool_format = query_tmtool(29446767, out_format='PubTator')
    # tmtool2bern(tmtool_res, tmtool_format=tmtool_format, is_raw_text=True)



    # pubmed_baseid = 626800
    # pubmed_path = get_pubmed_xml(pubmed_baseid)
    # if pubmed_path is not None:
    #     pubmed_xml2pubtator(pubmed_baseid, pubmed_path)
    # for pubmedid in range(pubmed_baseid, 29999999, 100):
    #     pubmed_path = get_pubmed_xml(pubmedid)
    #     if pubmed_path is not None:
    #         pubmed_xml2pubtator(pubmedid, pubmed_path)
    # get_pubmed_xml('sadrfasd')

    # pmc_path = get_pmc_xml('PMC6137033')
    # pmc_xml2pubtator('PMC6137033', pmc_path)

    # pmc_baseid = 6050000
    # for pmcidbase in range(pmc_baseid, 6299999, 10000):
    #     pmc_id = 'PMC{}'.format(pmcidbase)
    #     pmc_path = get_pmc_xml(pmc_id)
    #     if pmc_path is not None:
    #         pmc_xml2pubtator(pmc_id, pmc_path)
    # get_pmc_xml('PMC131')
