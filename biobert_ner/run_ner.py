#! usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Copyright 2018 The Google AI Language Team Authors.
BASED ON Google_BERT.
@Author:zhoukaiyin
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading
import time

from biobert_ner.modeling import *
from biobert_ner.tokenization import *
from biobert_ner.ops import *
from biobert_ner.utils import Profile, show_prof_data
from biobert_ner.fast_predict2 import FastPredict
from convert import preprocess


flags = tf.flags

flags.DEFINE_string(
    "task_name", "NER", "The name of the task to train."
)

flags.DEFINE_string(
    "data_dir", None,
    "The input datadir.",
)

flags.DEFINE_string(
    "model_dir", './pretrainedBERT/',
    "The input datadir.",
)

flags.DEFINE_string(
    "bert_config_file", './conf/bert_config.json',
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string("vocab_file", './conf/vocab.txt',
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "init_checkpoint", './pretrainedBERT/pubmed_pmc_470k/biobert_model.ckpt',
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "rep_ent", False,
    "Off entity type decision rules? Whether to print all rep outputs"
)

flags.DEFINE_bool(
    "do_lower_case", False,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 256,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_bool("do_predict", True,
                  "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("predict_batch_size", 2, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

# BERN settings
flags.DEFINE_string("ip", '0.0.0.0', "")
flags.DEFINE_integer("port", 8888, "")
flags.DEFINE_string("gnormplus_home", os.path.join(os.path.expanduser('~'),
                                                   'bern', 'GNormPlusJava'), "")
flags.DEFINE_string("gnormplus_host", 'localhost', "")
flags.DEFINE_integer("gnormplus_port", 18895, "")
flags.DEFINE_string("tmvar2_home", os.path.join(os.path.expanduser('~'),
                                                'bern', 'tmVarJava'), "")
flags.DEFINE_string("tmvar2_host", 'localhost', "")
flags.DEFINE_integer("tmvar2_port", 18896, "")
# BERN settings

FLAGS = flags.FLAGS


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids,):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        # self.label_mask = label_mask


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, data, pmids):
        """Reads a BIO data."""
        lines = []
        words = []
        labels = []
        for pmid in pmids:
            for sent in data[pmid]['words']:
                words = sent[:]
                labels = ['O'] * len(words)

                if len(words) >= 30:
                    while len(words) >= 30:
                        tmplabel = labels[:30]
                        l = ' '.join([label for label
                                      in labels[:len(tmplabel)]
                                      if len(label) > 0])
                        w = ' '.join([word for word
                                      in words[:len(tmplabel)]
                                      if len(word) > 0])
                        lines.append([l, w])
                        words = words[len(tmplabel):]
                        labels = labels[len(tmplabel):]
                if len(words) == 0:
                    continue

                l = ' '.join([label for label in labels if len(label) > 0])
                w = ' '.join([word for word in words if len(word) > 0])
                lines.append([l, w])
                words = []
                labels = []
                continue

        return lines


class NerProcessor(DataProcessor):
    def get_test_examples(self, data_dir):
        data = list()
        pmids = list()
        with open(data_dir, 'r') as in_:
            for line in in_:
                line = line.strip()
                tmp = json.loads(line)
                tmp['title'] = preprocess(tmp['title'])
                tmp['abstract'] = preprocess(tmp['abstract'])
                data.append(tmp)
                pmids.append(tmp["pmid"])

        json_file = input_form(json_to_sent(data))

        return \
            self._create_example(self._read_data(json_file, pmids), "test"), \
            json_file, data

    def get_test_dict_list(self, dict_list, is_raw_text=False):
        pmids = list()
        for d in dict_list:
            pmids.append(d["pmid"])
            # d['title'] = preprocess(d['title'])
            # d['abstract'] = preprocess(d['abstract'])

        json_file = input_form(json_to_sent(dict_list, is_raw_text=is_raw_text))

        return \
            self._create_example(self._read_data(json_file, pmids), "test"), \
            json_file

    def get_labels(self):
        return ["B", "I", "O", "X", "[CLS]", "[SEP]"]

    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = convert_to_unicode(line[1])
            label = convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples


def file_based_input_fn_builder(input_file, seq_length, drop_remainder):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        # "label_ids":tf.VarLenFeature(tf.int64),
        # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.cast(t, dtype=tf.int32)
            example[name] = t
        return example

    def input_fn(params):
        batch_size = params["batch_size"]
        d = tf.data.TFRecordDataset(input_file)
        # if is_training:
        #     d = d.repeat()
        #     d = d.shuffle(buffer_size=100)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder
        ))
        return d
    return input_fn


def create_model(bert_config, is_training, input_ids, input_mask,
                 segment_ids, labels, num_labels, use_one_hot_embeddings):
    model = BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings
    )

    output_layer = model.get_sequence_output()

    hidden_size = output_layer.shape[-1].value

    output_weight = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("loss"):
        if is_training:
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
        output_layer = tf.reshape(output_layer, [-1, hidden_size])
        logits = tf.matmul(output_layer, output_weight, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [-1, FLAGS.max_seq_length, 7])
        # mask = tf.cast(input_mask,tf.float32)
        # loss = tf.contrib.seq2seq.sequence_loss(logits,labels,mask)
        # return (loss, logits, predict)
        #
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_sum(per_example_loss)
        probabilities = tf.nn.softmax(logits, axis=-1)
        predict = tf.argmax(probabilities, axis=-1)
        return loss, per_example_loss, logits, log_probs, predict
        #


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        # label_mask = features["label_mask"]
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, log_probs, predicts) = \
            create_model(bert_config, is_training, input_ids, input_mask,
                         segment_ids, label_ids, num_labels,
                         use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names) = \
                get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:
                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint,
                                                  assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"

        assert mode == tf.estimator.ModeKeys.PREDICT

        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={"prediction": predicts, "log_probs": log_probs},
            scaffold_fn=scaffold_fn
        )
        return output_spec
    return model_fn


class BioBERT:
    def __init__(self, _):
        init_start_t = time.time()

        self.FLAGS = FLAGS

        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        tf.logging.set_verbosity(tf.logging.INFO)

        bert_config = BertConfig.from_json_file(FLAGS.bert_config_file)

        if FLAGS.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (FLAGS.max_seq_length, bert_config.max_position_embeddings))

        self.processor = NerProcessor()
        self.label_list = self.processor.get_labels()
        self.idx2label = dict()
        self.label2idx = dict()
        for lidx, l in enumerate(self.label_list):
            self.idx2label[lidx + 1] = l
            self.label2idx[l] = lidx + 1

        self.tokenizer = FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

        tpu_cluster_resolver = None
        if FLAGS.use_tpu and FLAGS.tpu_name:
            tpu_cluster_resolver = \
                tf.contrib.cluster_resolver.TPUClusterResolver(
                    FLAGS.tpu_name, zone=FLAGS.tpu_zone,
                    project=FLAGS.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        session_config = tf.ConfigProto()
        session_config.gpu_options.allow_growth = True

        self.estimator_dict = dict()
        self.entity_types = ['gene', 'disease', 'drug', 'species']
        for etype in self.entity_types:
            num_train_steps = None
            num_warmup_steps = None

            run_config = tf.contrib.tpu.RunConfig(
                cluster=tpu_cluster_resolver,
                master=FLAGS.master,
                model_dir=os.path.join(FLAGS.model_dir, etype),
                session_config=session_config,
                save_checkpoints_steps=FLAGS.save_checkpoints_steps,
                tpu_config=tf.contrib.tpu.TPUConfig(
                    iterations_per_loop=FLAGS.iterations_per_loop,
                    num_shards=FLAGS.num_tpu_cores,
                    per_host_input_for_training=is_per_host))

            model_fn = model_fn_builder(
                bert_config=bert_config,
                num_labels=len(self.label_list) + 1,
                init_checkpoint=FLAGS.init_checkpoint,
                learning_rate=FLAGS.learning_rate,
                num_train_steps=num_train_steps,
                num_warmup_steps=num_warmup_steps,
                use_tpu=FLAGS.use_tpu,
                use_one_hot_embeddings=FLAGS.use_tpu)

            estimator = tf.contrib.tpu.TPUEstimator(
                use_tpu=FLAGS.use_tpu,
                model_fn=model_fn,
                config=run_config,
                predict_batch_size=FLAGS.predict_batch_size)

            # self.estimator_dict[etype] = \
            #     FastPredict(estimator, self.fast_input_fn_builder_gen)

            # self.recognize()
            self.estimator_dict[etype] = \
                FastPredict(estimator, self.fast_input_fn_builder_gen_batch)

        self.counter = 0

        init_end_t = time.time()
        print('BioBERT init_t {:.3f} sec.'.format(init_end_t - init_start_t))

    @Profile(__name__)
    def recognize(self, input_dl, is_raw_text=False, thread_id=None,
                  indent=None):
        if thread_id is None:
            self.counter += 1
            req_id = self.counter
        else:
            req_id = thread_id

        if type(input_dl) is str:
            predict_examples, json_dict, data_list = \
                self.processor.get_test_examples(input_dl)
        elif type(input_dl) is list:
            predict_examples, json_dict = \
                self.processor.get_test_dict_list(input_dl, is_raw_text)
            data_list = input_dl
        else:
            raise ValueError('Wrong type')

        token_path = os.path.join("biobert_ner", "tmp",
                                  "token_test_{}.txt".format(req_id))

        if os.path.exists(token_path):
            os.remove(token_path)

        predict_example_list = list()
        for ex_index, example in enumerate(predict_examples):
            feature = self.convert_single_example(
                example, self.FLAGS.max_seq_length, req_id, "test")
            feature_dict = dict()
            # feature_dict["input_ids"] = [feature.input_ids]
            # feature_dict["input_mask"] = [feature.input_mask]
            # feature_dict["segment_ids"] = [feature.segment_ids]
            # feature_dict["label_ids"] = [feature.label_ids]
            feature_dict["input_ids"] = feature.input_ids
            feature_dict["input_mask"] = feature.input_mask
            feature_dict["segment_ids"] = feature.segment_ids
            feature_dict["label_ids"] = feature.label_ids
            predict_example_list.append(feature_dict)

        tokens = list()
        tot_tokens = list()
        with open(token_path, 'r') as reader:
            for line in reader:
                tok = line.strip()
                tot_tokens.append(tok)
                if tok == '[CLS]':
                    tmp_toks = [tok]
                elif tok == '[SEP]':
                    tmp_toks.append(tok)
                    tokens.append(tmp_toks)
                else:
                    tmp_toks.append(tok)

        # predict_example_list = self.get_inputs(predict_examples, req_id)

        predict_dict = dict()
        logits_dict = dict()

        # Threading
        threads = list()
        out_tag_dict = dict()
        for etype in self.entity_types:
            out_tag_dict[etype] = (False, None)
            t = threading.Thread(target=self.recognize_etype,
                                 args=(etype, predict_example_list,
                                       tokens, tot_tokens,
                                       predict_dict, logits_dict, data_list,
                                       json_dict, out_tag_dict))
            t.daemon = True
            t.start()
            threads.append(t)

        # block until all tasks are done
        for t in threads:
            t.join()

        for etype in self.entity_types:
            if out_tag_dict[etype][0]:
                if type(input_dl) is str:
                    print(os.path.split(input_dl)[1],
                          'Found an error:', out_tag_dict[etype][1])
                else:
                    print('Found an error:', out_tag_dict[etype][1])
                if os.path.exists(token_path):
                    os.remove(token_path)
                return None

        data_list = merge_results(data_list, json_dict, predict_dict,
                                  logits_dict, FLAGS.rep_ent,
                                  is_raw_text=is_raw_text)

        if type(input_dl) is str:
            output_path = os.path.join('result/', os.path.splitext(
                os.path.basename(input_dl))[0] + '_NER_{}.json'.format(req_id))
            gold_output_path = os.path.join('result/', os.path.splitext(
                os.path.basename(input_dl))[0] + '_NER.json')
            print('pred', output_path)
            print('gold', gold_output_path)

            with open(output_path, 'w') as resultf:
                for paper in data_list:
                    paper['ner_model'] = "BioBERT NER v.20190603"
                    resultf.write(
                        json.dumps(paper, sort_keys=True, indent=indent) + '\n')

            if os.path.exists(gold_output_path):
                with open(gold_output_path, 'r') as f_gold:
                    for lidx, l in enumerate(f_gold):
                        line_dict = json.loads(l)
                        gold_dict = data_list[lidx]
                        for etype in sorted(gold_dict['entities']):

                            if len(gold_dict['entities'][etype]) != \
                                   len(line_dict['entities'][etype]):
                                print('{} {} != {}'.format(
                                    etype, len(gold_dict['entities'][etype]),
                                    len(line_dict['entities'][etype])))
                                print('gold:', gold_dict['entities'][etype])
                                print('pred:', line_dict['entities'][etype])
                                continue

                            # assert len(gold_dict['entities'][etype]) == \
                            #        len(line_dict['entities'][etype]), \
                            #         '{} {} != {}'.format(
                            #         etype, len(gold_dict['entities'][etype]),
                            #         len(line_dict['entities'][etype]))

                            for (e_gold, e_pred) in zip(
                                    gold_dict['entities'][etype],
                                    line_dict['entities'][etype]):
                                # print(etype, e_gold, e_pred)
                                assert e_gold['start'] == e_pred['start']
                                assert e_gold['end'] == e_pred['end']
            else:
                print('Not found gold output', gold_output_path)

            print()

        # delete temp files
        if os.path.exists(token_path):
            os.remove(token_path)

        return data_list

    @Profile(__name__)
    def recognize_etype(self, etype, predict_example_list, tokens, tot_tokens,
                        predict_dict, logits_dict, data_list, json_dict,
                        out_tag_dict):
        # result = list()
        # for e in predict_example_list:
        #     result.append(self.estimator_dict[etype].predict(e))

        result = self.estimator_dict[etype].predict(predict_example_list)

        predicts = list()
        logits = list()
        for pidx, prediction in enumerate(result):
            slen = len(tokens[pidx])
            for p in prediction['prediction'][:slen]:
                if p in [0, 4, 5, 6]:
                    predicts.append(self.idx2label[3])
                else:
                    predicts.append(self.idx2label[p])
            for l in prediction['log_probs'][:slen]:
                logits.append(l)

        de_toks, de_labels, de_logits = detokenize(tot_tokens, predicts, logits)

        predict_dict[etype] = dict()
        logits_dict[etype] = dict()
        piv = 0
        for data in data_list:
            pmid = data['pmid']
            predict_dict[etype][pmid] = list()
            logits_dict[etype][pmid] = list()

            sent_lens = list()
            for sent in json_dict[pmid]['words']:
                sent_lens.append(len(sent))
            sent_idx = 0
            de_i = 0
            overlen = False
            while True:
                if overlen:
                    try:
                        predict_dict[etype][pmid][-1].extend(
                            de_labels[piv + de_i])
                    except Exception as e:
                        out_tag_dict[etype] = (True, e)
                        break
                    logits_dict[etype][pmid][-1].extend(de_logits[piv + de_i])
                    de_i += 1
                    if len(predict_dict[etype][pmid][-1]) == len(
                            json_dict[pmid]['words'][
                                len(predict_dict[etype][pmid]) - 1]):
                        sent_idx += 1
                        overlen = False

                else:
                    predict_dict[etype][pmid].append(de_labels[piv + de_i])
                    logits_dict[etype][pmid].append(de_logits[piv + de_i])
                    de_i += 1
                    if len(predict_dict[etype][pmid][-1]) == len(
                            json_dict[pmid]['words'][
                                len(predict_dict[etype][pmid]) - 1]):
                        sent_idx += 1
                        overlen = False
                    else:
                        overlen = True

                if sent_idx == len(json_dict[pmid]['words']):
                    piv += de_i
                    break
            if out_tag_dict[etype][0]:
                break

    def filed_based_convert_examples_to_features(self, examples, max_seq_length,
                                                 output_file, req_id,
                                                 mode='test'):
        features_list = list()
        writer = tf.python_io.TFRecordWriter(output_file)
        for (ex_index, example) in enumerate(examples):
            feature = self.convert_single_example(example, max_seq_length,
                                                  req_id, mode)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(
                    value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(feature.input_ids)
            features["input_mask"] = create_int_feature(feature.input_mask)
            features["segment_ids"] = create_int_feature(feature.segment_ids)
            features["label_ids"] = create_int_feature(feature.label_ids)
            # features["label_mask"] = create_int_feature(feature.label_mask)
            tf_example = tf.train.Example(features=tf.train.Features(
                feature=features))
            writer.write(tf_example.SerializeToString())

            feature_dict = dict()
            # feature_dict["input_ids"] = [feature.input_ids]
            # feature_dict["input_mask"] = [feature.input_mask]
            # feature_dict["segment_ids"] = [feature.segment_ids]
            # feature_dict["label_ids"] = [feature.label_ids]
            feature_dict["input_ids"] = feature.input_ids
            feature_dict["input_mask"] = feature.input_mask
            feature_dict["segment_ids"] = feature.segment_ids
            feature_dict["label_ids"] = feature.label_ids
            features_list.append(feature_dict)

        writer.close()

        return features_list

    def get_input_generator(self, predict_examples, req_id, mode='test'):
        for (ex_index, example) in enumerate(predict_examples):
            feature = self.convert_single_example(example,
                                                  self.FLAGS.max_seq_length,
                                                  req_id, mode)

            feature_dict = dict()
            feature_dict["input_ids"] = [feature.input_ids]
            feature_dict["input_mask"] = [feature.input_mask]
            feature_dict["segment_ids"] = [feature.segment_ids]
            feature_dict["label_ids"] = [feature.label_ids]
            yield feature_dict

    def get_inputs(self, predict_examples, req_id, mode='test'):
        features = list()
        for (ex_index, example) in enumerate(predict_examples):
            feature = self.convert_single_example(example,
                                                  self.FLAGS.max_seq_length,
                                                  req_id, mode)

            feature_dict = dict()
            feature_dict["input_ids"] = feature.input_ids
            feature_dict["input_mask"] = feature.input_mask
            feature_dict["segment_ids"] = feature.segment_ids
            feature_dict["label_ids"] = feature.label_ids
            features.append(feature_dict)
        return features

    # https://hanxiao.github.io/2019/01/02/Serving-Google-BERT-in-Production-using-Tensorflow-and-ZeroMQ/
    def fast_input_fn_builder_gen(self, gen_predict_examples):

        def input_fn(params):
            ds = tf.data.Dataset.from_generator(
                gen_predict_examples, output_types={
                    'input_ids': tf.int32,
                    'input_mask': tf.int32,
                    'segment_ids': tf.int32,
                    'label_ids': tf.int32
                }, output_shapes={
                    'input_ids': (self.FLAGS.max_seq_length),
                    'input_mask': (self.FLAGS.max_seq_length),
                    'segment_ids': (self.FLAGS.max_seq_length),
                    'label_ids': (self.FLAGS.max_seq_length)
                    # 'input_ids': (None, self.FLAGS.max_seq_length),
                    # 'input_mask': (None, self.FLAGS.max_seq_length),
                    # 'segment_ids': (None, self.FLAGS.max_seq_length),
                    # 'label_ids': (None, self.FLAGS.max_seq_length)
                }).batch(1)
            return ds

        return input_fn

    def fast_input_fn_builder_gen_batch(self, gen_predict_examples):
        seq_length = self.FLAGS.max_seq_length

        def input_fn(params):
            # batch_size = params["batch_size"]
            output_types = {
                'input_ids': tf.int32,
                'input_mask': tf.int32,
                'segment_ids': tf.int32,
                'label_ids': tf.int32
            }
            output_shapes = {
                # 'input_ids': (None, seq_length),
                # 'input_mask': (None, seq_length),
                # 'segment_ids': (None, seq_length),
                # 'label_ids': (None, seq_length)
                'input_ids': (seq_length,),
                'input_mask': (seq_length,),
                'segment_ids': (seq_length,),
                'label_ids': (seq_length,)
            }
            d = tf.data.Dataset.from_generator(
                gen_predict_examples, output_types,
                output_shapes=output_shapes)
            # d = d.prefetch(batch_size)  # error
            # d = d.batch(batch_size)  # error
            d = d.batch(1)
            return d

        return input_fn

    def convert_single_example(self, example, max_seq_length, req_id, mode):
        label_map = {}
        for (i, label) in enumerate(self.label_list, 1):
            label_map[label] = i
        textlist = example.text.split(' ')
        labellist = example.label.split(' ')
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        # tokens = tokenizer.tokenize(example.text)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
            labels = labels[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        label_ids.append(label_map["[SEP]"])
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        # label_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")
            # label_mask.append(0)
        # print(len(input_ids))
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        # assert len(label_mask) == max_seq_length

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=label_ids,
            # label_mask = label_mask
        )
        self.write_tokens(ntokens, mode, req_id)
        return feature

    @staticmethod
    def write_tokens(tokens, mode, req_id):
        if mode == "test":
            path = os.path.join("biobert_ner", "tmp",
                                "token_{}_{}.txt".format(mode, req_id))
            with open(path, 'a') as wf:
                for token in tokens:
                    if token != "**NULL**":
                        wf.write(token + '\n')

    def close(self):
        for etype in self.entity_types:
            self.estimator_dict[etype].close()


def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    biobert = BioBERT(FLAGS)

    from convert import pubtator_biocxml2dict_list
    import json

    dl = pubtator_biocxml2dict_list(
        [26658955, 24189420, 22579007, 29185436])
    for d in dl:
        print(d['pmid'])
        with open('/media/donghyeon/f7c53837-2156-4793-b2b1-4b0578dffef1'
                  '/biobert/BioBert_NER/BioBERTNER/data/' + d['pmid'] + '.json',
                  'w', encoding='utf-8') as f_out:
            json.dump(d, f_out)
        biobert.recognize([d])

    biobert.close()

    show_prof_data()


if __name__ == "__main__":
    tf.app.run()
