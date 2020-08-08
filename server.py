from http.server import HTTPServer, BaseHTTPRequestHandler

import cgi
from datetime import datetime
import hashlib
import json
import numpy as np
from biobert_ner.run_ner import BioBERT, FLAGS
from convert import pubtator2dict_list, pubtator_biocxml2dict_list, \
    get_pub_annotation, get_pubtator
from normalize import Normalizer
from utils import filter_entities
import os
import random
import shutil
import string
import socket
import struct
import time
import tensorflow as tf
import threading
import urllib.parse as urlparse
# if hasattr(os, "fork"):
#     from socketserver import ForkingMixIn
# else:
#     from socketserver import ThreadingMixIn
from socketserver import ThreadingMixIn


class GetHandler(BaseHTTPRequestHandler):
    stm_dict = None
    normalizer = None

    def do_GET(self):
        get_start_t = time.time()
        parsed_path = urlparse.urlparse(self.path)
        cur_thread_name = threading.current_thread().getName()

        message = '\n'.join([
            'CLIENT VALUES:',
            'client_address=%s (%s)' % (self.client_address,
                                        self.address_string()),
            'command=%s' % self.command,
            'path=%s' % self.path,
            'real path=%s' % parsed_path.path,
            'query=%s' % parsed_path.query,
            'request_version=%s' % self.request_version,
            '',
            'SERVER VALUES:',
            'server_version=%s' % self.server_version,
            'sys_version=%s' % self.sys_version,
            'protocol_version=%s' % self.protocol_version,
            'thread_name=%s' % cur_thread_name,
        ])
        self.send_response(200)
        self.end_headers()

        elapsed_time_dict = dict()

        time_format = self.stm_dict['time_format']
        available_formats = self.stm_dict['available_formats']

        if parsed_path.query is None:
            err_msg = 'No url query'
            print(datetime.now().strftime(time_format),
                  '[' + cur_thread_name + ']', err_msg)
            message += '\n' + err_msg
            self.wfile.write(message.encode('utf-8'))
            return

        indent = None

        # print(datetime.now().strftime(time_format),
        #       'query', parsed_path.query)

        qs_dict = urlparse.parse_qs(parsed_path.query)
        # print(datetime.now().strftime(time_format), 'qs_dict', qs_dict)

        if 'pmid' not in qs_dict or len(qs_dict['pmid']) == 0:
            err_msg = 'No pmid param'
            print(datetime.now().strftime(time_format),
                  '[' + cur_thread_name + ']', err_msg)
            message += '\n' + err_msg
            self.wfile.write(message.encode('utf-8'))
            return

        pmid_list = qs_dict['pmid'][0].split(',')
        # print(datetime.now().strftime(time_format), 'pmid', pmid_list)

        if len(pmid_list) > self.stm_dict['n_pmid_limit']:
            err_msg = 'Too many (> {}) pmids: {}'.format(
                self.stm_dict['n_pmid_limit'], len(pmid_list))
            print(datetime.now().strftime(time_format),
                  '[' + cur_thread_name + ']', err_msg)
            message += '\n' + err_msg
            self.wfile.write(message.encode('utf-8'))
            return

        out_format = available_formats[0]
        if 'format' in qs_dict and len(qs_dict['format']) > 0:
            if qs_dict['format'][0] in available_formats:
                out_format = qs_dict['format'][0]
            else:
                print('Unavailable format', qs_dict['format'][0])

        # print(datetime.now().strftime(time_format),
        #       'pmid:', pmid_list, ', format:', out_format)

        if 'indent' in qs_dict and len(qs_dict['indent']) > 0:
            indent = qs_dict['indent'][0]
            if 'true' == indent.lower():
                indent = 4
            else:
                indent = None

        text_hash = \
            hashlib.sha224(qs_dict['pmid'][0].encode('utf-8')).hexdigest()
        print(datetime.now().strftime(time_format),
              '[{}] text_hash: {}'.format(cur_thread_name, text_hash))

        # bern_output_path = './output/bern_api_{}.{}'.format(text_hash,
        #                                                     out_format)

        # # Re-use prev. outputs
        # if os.path.exists(bern_output_path):
        #     with open(bern_output_path, 'r', encoding='utf-8') as f_out:
        #         if out_format == 'json':
        #             message = \
        #                 json.dumps(json.load(f_out), indent=indent,
        #                            sort_keys=indent is not None)
        #         elif out_format == 'pubtator':
        #             message = f_out.read()
        #         else:
        #             raise ValueError('Wrong format: {}'.format(out_format))
        #
        #     self.wfile.write(message.encode('utf-8'))
        #     print(datetime.now().strftime(time_format),
        #           '[{}] Done. Found prev. output. Total {:.3f} sec\n'.
        #           format(cur_thread_name, time.time() - get_start_t))
        #     return

        is_raw_text = False

        tmtool_start_t = time.time()
        dict_list = pubtator_biocxml2dict_list(pmid_list)
        tmtool_time = time.time() - tmtool_start_t
        elapsed_time_dict['tmtool'] = round(tmtool_time, 3)
        if dict_list is None:
            error_dict = self.get_err_dict()
            error_dict['pmid'] = pmid_list[0] if len(pmid_list) == 1 else ''
            error_dict['abstract'] = 'error: tmtool: no response'
            print(datetime.now().strftime(time_format),
                  '[' + cur_thread_name + ']', error_dict['abstract'])

            if out_format == available_formats[0]:
                self.wfile.write(
                    json.dumps([get_pub_annotation(error_dict,
                                                   is_raw_text=is_raw_text)],
                               indent=indent,
                               sort_keys=indent is not None).encode('utf-8'))
            elif out_format == available_formats[1]:
                self.wfile.write(get_pubtator([error_dict]).encode('utf-8'))

            return
        elif type(dict_list) is str:
            error_dict = self.get_err_dict()
            error_dict['pmid'] = pmid_list[0] if len(pmid_list) == 1 else ''
            if 'currently unavailable' in dict_list:
                error_dict['abstract'] = 'error: tmtool: currently unavailable'
            elif 'invalid version format' in dict_list:
                error_dict['abstract'] = 'error: tmtool: invalid version format'
            else:
                error_dict['abstract'] = 'error: tmtool: {}'.format(
                    dict_list.replace('\n', ''))
            print(datetime.now().strftime(time_format),
                  '[' + cur_thread_name + ']', error_dict['abstract'])

            if out_format == available_formats[0]:
                self.wfile.write(
                    json.dumps([get_pub_annotation(error_dict,
                                                   is_raw_text=is_raw_text)],
                               indent=indent,
                               sort_keys=indent is not None).encode('utf-8'))
            elif out_format == available_formats[1]:
                self.wfile.write(get_pubtator([error_dict]).encode('utf-8'))

            return

        print(datetime.now().strftime(time_format),
              '[{}] tmTool: PubMed & GNormPlus & tmVar {:.3f} sec'
              .format(cur_thread_name, tmtool_time))

        # Run BioBERT NER models of Lee et al., 2019
        ner_start_time = time.time()
        tagged_docs, num_entities = \
            self.biobert_recognize(dict_list, is_raw_text, cur_thread_name)
        ner_time = time.time() - ner_start_time
        elapsed_time_dict['ner'] = round(ner_time, 3)
        if tagged_docs is None:
            error_dict = self.get_err_dict()
            error_dict['pmid'] = pmid_list[0] if len(pmid_list) == 1 else ''
            error_dict['abstract'] = 'error: BioBERT NER, out of index range'

            if out_format == available_formats[0]:
                self.wfile.write(
                    json.dumps([get_pub_annotation(error_dict,
                                                   is_raw_text=is_raw_text)],
                               indent=indent,
                               sort_keys=indent is not None).encode('utf-8'))
            elif out_format == available_formats[1]:
                self.wfile.write(get_pubtator([error_dict]).encode('utf-8'))

            return
        print(datetime.now().strftime(time_format),
              '[%s] NER %.3f sec, #entities: %d, #articles: %d'
              % (cur_thread_name, ner_time, num_entities, len(tagged_docs)))

        # Normalization models
        normalization_time = 0.
        if num_entities > 0:
            # print(datetime.now().strftime(time_format),
            #       '[{}] Normalization models..'.format(cur_thread_name))
            normalization_start_time = time.time()
            tagged_docs = self.normalizer.normalize(text_hash, tagged_docs,
                                                    cur_thread_name,
                                                    is_raw_text=is_raw_text)
            normalization_time = time.time() - normalization_start_time
        elapsed_time_dict['normalization'] = round(normalization_time, 3)

        # apply output format
        if out_format == available_formats[0]:

            elapsed_time_dict['total'] = round(time.time() - get_start_t, 3)

            # PubAnnotation JSON
            pubannotation_res = list()
            for d in tagged_docs:
                pubannotation_res.append(
                    get_pub_annotation(d, is_raw_text=is_raw_text,
                                       elapsed_time_dict=elapsed_time_dict))
            self.wfile.write(
                json.dumps(pubannotation_res, indent=indent,
                           sort_keys=indent is not None).encode('utf-8'))

            # # Save a BERN result
            # with open(bern_output_path, 'w', encoding='utf-8') as f_out:
            #     json.dump(pubannotation_res, f_out)

        elif out_format == available_formats[1]:
            # PubTator
            self.wfile.write(get_pubtator(tagged_docs).encode('utf-8'))

            # # Save a BERN result
            # with open(bern_output_path, 'w', encoding='utf-8') as f_out:
            #     f_out.write(pubtator_res)

        print(datetime.now().strftime(time_format),
              '[{}] Done. Total {:.3f} sec\n'.format(cur_thread_name,
                                                     time.time() - get_start_t))
        return

    def do_POST(self):
        post_start_t = time.time()
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST',
                     'CONTENT_TYPE': self.headers['Content-Type'],
                     }
        )

        self.send_response(200)
        self.end_headers()

        cur_thread_name = threading.current_thread().getName()

        time_format = self.stm_dict['time_format']

        # input
        if 'param' not in form:
            err_msg = [{"error": "no param"}]
            print(datetime.now().strftime(time_format),
                  '[' + cur_thread_name + ']', err_msg)
            self.wfile.write(json.dumps(err_msg).encode('utf-8'))
            return

        data = json.loads(form['param'].value)

        if 'text' not in data:
            err_msg = [{"error": "no text"}]
            print(datetime.now().strftime(time_format),
                  '[' + cur_thread_name + ']', err_msg)
            self.wfile.write(json.dumps(err_msg).encode('utf-8'))
            return

        text = str(data['text'])
        # print(datetime.now().strftime(time_format), 'Input:', text)

        if text == '':
            err_msg = [{"error": "empty text"}]
            print(datetime.now().strftime(time_format),
                  '[' + cur_thread_name + ']', err_msg)
            self.wfile.write(json.dumps(err_msg).encode('utf-8'))
            return
        elif not text.strip() or text.isspace():
            err_msg = [{"error": "only whitespace letters"}]
            print(datetime.now().strftime(time_format),
                  '[' + cur_thread_name + ']', err_msg)
            self.wfile.write(json.dumps(err_msg).encode('utf-8'))
            return

        text = self.preprocess_input(text, cur_thread_name)

        # NER
        result_dict = self.tag_entities(
            text, cur_thread_name, is_raw_text=True, reuse=False)

        if result_dict is None:
            err_msg = [{"error": "NER crash"}]
            print(datetime.now().strftime(time_format),
                  '[' + cur_thread_name + ']', err_msg)
            self.wfile.write(json.dumps(err_msg).encode('utf-8'))
            return

        result_dict_str = json.dumps(result_dict)

        # # output, pretty json
        # print(datetime.now().strftime(time_format), 'output:',
        #       json.dumps(result_dict, indent=4, sort_keys=True))

        # send message
        self.wfile.write(result_dict_str.encode('utf-8'))
        print(datetime.now().strftime(self.stm_dict['time_format']),
              '[{}] Done. total {:.3f} sec\n'
              .format(cur_thread_name, time.time() - post_start_t))
        return

    def preprocess_input(self, text, cur_thread_name):

        if '\r\n' in text:
            print(datetime.now().strftime(self.stm_dict['time_format']),
                  '[{}] Found a CRLF -> replace it w/ a space'
                  .format(cur_thread_name))
            text = text.replace('\r\n', ' ')

        if '\n' in text:
            print(datetime.now().strftime(self.stm_dict['time_format']),
                  '[{}] Found a line break -> replace it w/ a space'
                  .format(cur_thread_name))
            text = text.replace('\n', ' ')

        if '\t' in text:
            print(datetime.now().strftime(self.stm_dict['time_format']),
                  '[{}] Found a tab -> replace w/ a space'
                  .format(cur_thread_name))
            text = text.replace('\t', ' ')

        found_too_long_words = 0
        tokens = text.split(' ')
        for idx, tk in enumerate(tokens):
            if len(tk) > self.stm_dict['max_word_len']:
                tokens[idx] = tk[:self.stm_dict['max_word_len']]
                found_too_long_words += 1
        if found_too_long_words > 0:
            print(datetime.now().strftime(self.stm_dict['time_format']),
                  '[{}] Found a too long word -> cut the suffix of the word'
                  .format(cur_thread_name))
            text = ' '.join(tokens)

        return text

    def tag_entities(self, text, cur_thread_name, is_raw_text, reuse=False):
        assert self.stm_dict is not None

        n_ascii_letters = 0
        for l in text:
            if l not in string.ascii_letters:
                continue
            n_ascii_letters += 1

        if n_ascii_letters == 0:
            text = 'No ascii letters. Please enter your text in English.'

        text_hash = hashlib.sha224(text.encode('utf-8')).hexdigest()
        print(datetime.now().strftime(self.stm_dict['time_format']),
              '[{}] text_hash: {}'.format(cur_thread_name, text_hash))

        bern_output_path = './output/bern_demo_{}.json'.format(text_hash)

        if reuse and os.path.exists(bern_output_path):
            print(datetime.now().strftime(self.stm_dict['time_format']),
                  f'[{cur_thread_name}] Found prev. output')
            with open(bern_output_path, 'r', encoding='utf-8') as f_out:
                return json.load(f_out)

        pubtator_file = f'{text_hash}-{cur_thread_name}.PubTator'
        home_gnormplus = self.stm_dict['gnormplus_home']
        input_gnormplus = os.path.join(home_gnormplus, 'input', pubtator_file)
        output_gnormplus = os.path.join(home_gnormplus, 'output', pubtator_file)

        home_tmvar2 = self.stm_dict['tmvar2_home']
        input_dir_tmvar2 = os.path.join(home_tmvar2, 'input')
        input_tmvar2 = os.path.join(input_dir_tmvar2, pubtator_file)
        output_tmvar2 = os.path.join(home_tmvar2, 'output',
                                     f'{pubtator_file}.PubTator')

        # Write input str to a .PubTator format file
        with open(input_gnormplus, 'w', encoding='utf-8') as f:
            # only abstract
            f.write(f'{text_hash}-{cur_thread_name}|t|\n')
            f.write(f'{text_hash}-{cur_thread_name}|a|{text}\n\n')

        # Run GNormPlus
        gnormplus_start_time = time.time()
        gnormplus_resp = tell_inputfile(self.stm_dict['gnormplus_host'],
                                        self.stm_dict['gnormplus_port'],
                                        pubtator_file)
        if gnormplus_resp is None:
            os.remove(input_gnormplus)
            return None

        print(datetime.now().strftime(self.stm_dict['time_format']),
              '[{}] GNormPlus {:.3f} sec'
              .format(cur_thread_name, time.time() - gnormplus_start_time))

        # Move a GNormPlus output file to the tmVar2 input directory
        shutil.move(output_gnormplus, input_tmvar2)

        # Run tmVar 2.0
        tmvar2_start_time = time.time()
        tmvar2_resp = tell_inputfile(self.stm_dict['tmvar2_host'],
                                     self.stm_dict['tmvar2_port'],
                                     pubtator_file)
        if tmvar2_resp is None:
            os.remove(input_gnormplus)
            os.remove(input_tmvar2)
            return None

        print(datetime.now().strftime(self.stm_dict['time_format']),
              '[{}] tmVar 2.0 {:.3f} sec'
              .format(cur_thread_name, time.time() - tmvar2_start_time))

        # Convert tmVar 2.0 outputs (?.PubTator.PubTator) to python dict
        dict_list = pubtator2dict_list(output_tmvar2, is_raw_text=True)

        # Delete temp files
        os.remove(input_gnormplus)
        os.remove(input_tmvar2)
        os.remove(output_tmvar2)

        # error
        if type(dict_list) is str:
            print(dict_list)
            return None

        # Run BioBERT of Lee et al., 2019
        start_time = time.time()
        tagged_docs, num_entities = \
            self.biobert_recognize(dict_list, is_raw_text, cur_thread_name)
        if tagged_docs is None:
            return None

        assert len(tagged_docs) == 1
        print(datetime.now().strftime(self.stm_dict['time_format']),
              '[%s] NER %.3f sec, #entities: %d' %
              (cur_thread_name, time.time() - start_time, num_entities))

        # Normalization models
        if num_entities > 0:
            # print(datetime.now().strftime(time_format),
            #       '[{}] Normalization models..'.format(cur_thread_name))
            tagged_docs = self.normalizer.normalize(text_hash, tagged_docs,
                                                    cur_thread_name,
                                                    is_raw_text=is_raw_text)

        # Convert to PubAnnotation JSON
        tagged_docs[0] = get_pub_annotation(tagged_docs[0],
                                            is_raw_text=is_raw_text)

        # # Save a BERN result
        # with open(bern_output_path, 'w', encoding='utf-8') as f_out:
        #     json.dump(tagged_docs[0], f_out, sort_keys=True)

        return tagged_docs[0]

    def biobert_recognize(self, dict_list, is_raw_text, cur_thread_name):
        res = self.stm_dict['biobert'].recognize(dict_list,
                                                 is_raw_text=is_raw_text,
                                                 thread_id=cur_thread_name)
        if res is None:
            return None, 0

        num_filtered_species_per_doc = filter_entities(res, is_raw_text)
        for n_f_spcs in num_filtered_species_per_doc:
            if n_f_spcs[1] > 0:
                print(datetime.now().strftime(self.stm_dict['time_format']),
                      '[{}] Filtered {} species{}'
                      .format(cur_thread_name, n_f_spcs[1],
                              '' if is_raw_text
                              else ' in PMID:%s' % n_f_spcs[0]))
        num_entities = count_entities(res)
        return res, num_entities

    @staticmethod
    def get_err_dict():
        return {
            'pmid': '',
            'title': '',
            'abstract': '',
            'entities': {
                'mutation': [],
                'drug': [],
                'gene': [],
                'disease': [],
                'species': [],
            }
        }


# https://docs.python.org/3.6/library/socketserver.html#asynchronous-mixins
# https://stackoverflow.com/a/14089457
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    request_queue_size = 4  # to match our server's memory size (default: 5)


def count_entities(data):
    num_entities = 0
    for d in data:
        if 'entities' not in d:
            continue
        doc_ett = d['entities']
        num_entities += len(doc_ett['gene'])
        num_entities += len(doc_ett['disease'])
        num_entities += len(doc_ett['drug'])
        num_entities += len(doc_ett['species'])
        if 'mutation' in doc_ett:
            num_entities += len(doc_ett['mutation'])
    return num_entities


def tell_inputfile(host, port, inputfile):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        sock.connect((host, port))
        input_str = inputfile
        input_stream = struct.pack('>H', len(input_str)) + input_str.encode(
            'utf-8')
        sock.send(input_stream)

        output_stream = sock.recv(512)
        resp = output_stream.decode('utf-8')[2:]

        sock.close()
        return resp
    except ConnectionRefusedError as e:
        print(e)
        # from utils import send_mail
        # from service_checker import FROM_GMAIL_ADDR, \
        #     FROM_GMAIL_ACCOUNT_PASSWORD, \
        #     TO_EMAIL_ADDR
        # send_mail(FROM_GMAIL_ADDR, TO_EMAIL_ADDR,
        #           '[BERN] Error: {}\n'.format(type(e)),
        #           'host:port= {}:{}\ninputfile: {}'.format(
        #               host, port, inputfile),
        #           FROM_GMAIL_ADDR, FROM_GMAIL_ACCOUNT_PASSWORD)
        return None
    except TimeoutError as e:
        print(e)
        return None
    except ConnectionResetError as e:
        print(e)
        return None


def delete_files(dirname):
    if not os.path.exists(dirname):
        return

    n_deleted = 0
    for f in os.listdir(dirname):
        f_path = os.path.join(dirname, f)
        if not os.path.isfile(f_path):
            continue
        # print('Delete', f_path)
        os.remove(f_path)
        n_deleted += 1
    print(dirname, n_deleted)


class Main:
    def __init__(self, params):
        print(datetime.now().strftime(params.time_format), 'Starting..')
        # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # verbose off(info, warning)
        random.seed(params.seed)
        np.random.seed(params.seed)
        tf.set_random_seed(params.seed)

        print("A GPU is{} available".format(
            "" if tf.test.is_gpu_available() else " NOT"))

        stm_dict = dict()
        stm_dict['params'] = params

        FLAGS.model_dir = './biobert_ner/pretrainedBERT/'
        FLAGS.bert_config_file = './biobert_ner/conf/bert_config.json'
        FLAGS.vocab_file = './biobert_ner/conf/vocab.txt'
        FLAGS.init_checkpoint = \
            './biobert_ner/pretrainedBERT/pubmed_pmc_470k/biobert_model.ckpt'

        FLAGS.ip = params.ip
        FLAGS.port = params.port

        FLAGS.gnormplus_home = params.gnormplus_home
        FLAGS.gnormplus_host = params.gnormplus_host
        FLAGS.gnormplus_port = params.gnormplus_port

        FLAGS.tmvar2_home = params.tmvar2_home
        FLAGS.tmvar2_host = params.tmvar2_host
        FLAGS.tmvar2_port = params.tmvar2_port

        # import pprint
        # pprint.PrettyPrinter().pprint(FLAGS.__flags)

        stm_dict['biobert'] = BioBERT(FLAGS)

        stm_dict['gnormplus_home'] = params.gnormplus_home
        stm_dict['gnormplus_host'] = params.gnormplus_host
        stm_dict['gnormplus_port'] = params.gnormplus_port

        stm_dict['tmvar2_home'] = params.tmvar2_home
        stm_dict['tmvar2_host'] = params.tmvar2_host
        stm_dict['tmvar2_port'] = params.tmvar2_port

        stm_dict['max_word_len'] = params.max_word_len
        stm_dict['ner_model'] = params.ner_model
        stm_dict['n_pmid_limit'] = params.n_pmid_limit
        stm_dict['time_format'] = params.time_format
        stm_dict['available_formats'] = params.available_formats

        if not os.path.exists('./output'):
            os.mkdir('output')
        else:
            # delete prev. version outputs
            delete_files('./output')

        delete_files(os.path.join(params.gnormplus_home, 'input'))
        delete_files(os.path.join(params.tmvar2_home, 'input'))
        delete_files(os.path.join('./biobert_ner', 'tmp'))

        print(datetime.now().strftime(params.time_format),
              'Starting server at http://{}:{}'.format(params.ip, params.port))

        # https://stackoverflow.com/a/18445168
        GetHandler.stm_dict = stm_dict
        GetHandler.normalizer = Normalizer()

        # https://docs.python.org/3.6/library/socketserver.html#asynchronous-mixins
        # https://stackoverflow.com/a/14089457
        server = ThreadedHTTPServer((params.ip, params.port), GetHandler)
        server.serve_forever()


if __name__ == '__main__':
    import argparse

    argparser = argparse.ArgumentParser()
    argparser.add_argument('--ip', default='0.0.0.0')
    argparser.add_argument('--port', type=int, default=8888)
    argparser.add_argument('--ner_model', default='BioBERT')
    argparser.add_argument('--max_word_len', type=int, help='word max chars',
                           default=50)
    argparser.add_argument('--seed', type=int, help='seed value', default=2019)

    argparser.add_argument('--gnormplus_home',
                           help='GNormPlus home',
                           default=os.path.join(os.path.expanduser('~'),
                                                'bern', 'GNormPlusJava'))
    argparser.add_argument('--gnormplus_host',
                           help='GNormPlus host', default='localhost')
    argparser.add_argument('--gnormplus_port', type=int,
                           help='GNormPlus port', default=18895)
    argparser.add_argument('--tmvar2_home',
                           help='tmVar 2.0 home',
                           default=os.path.join(os.path.expanduser('~'),
                                                'bern', 'tmVarJava'))
    argparser.add_argument('--tmvar2_host',
                           help='tmVar 2.0 host', default='localhost')
    argparser.add_argument('--tmvar2_port', type=int,
                           help='tmVar 2.0 port', default=18896)

    argparser.add_argument('--n_pmid_limit', type=int,
                           help='max # of pmids', default=10)
    argparser.add_argument('--available_formats', type=list,
                           help='output formats', default=['json', 'pubtator'])
    argparser.add_argument('--time_format',
                           help='time format', default='[%d/%b/%Y %H:%M:%S.%f]')

    args = argparser.parse_args()

    Main(args)
