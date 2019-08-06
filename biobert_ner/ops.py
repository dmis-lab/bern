import numpy as np
import re


tokenize_regex = re.compile(r'([0-9a-zA-Z]+|[^0-9a-zA-Z])')


def json_to_sent(data, is_raw_text=False):
    '''data: list of json file [{pmid,abstract,title}, ...] '''
    out = dict()
    for paper in data:
        sentences = list()
        if is_raw_text:
            # assure that paper['abstract'] is not empty
            abst = sentence_split(paper['abstract'])
            if len(abst) != 1 or len(abst[0].strip()) > 0:
                sentences.extend(abst)
        else:
            # assure that paper['title'] is not empty
            if len(CoNLL_tokenizer(paper['title'])) < 50:
                title = [paper['title']]
            else:
                title = sentence_split(paper['title'])
            if len(title) != 1 or len(title[0].strip()) > 0:
                sentences.extend(title)

            if len(paper['abstract']) > 0:
                abst = sentence_split(' ' + paper['abstract'])
                if len(abst) != 1 or len(abst[0].strip()) > 0:
                    sentences.extend(abst)

        out[paper['pmid']] = dict()
        out[paper['pmid']]['sentence'] = sentences
    return out


def input_form(sent_data):
    '''sent_data: dict of sentence, key=pmid {pmid:[sent,sent, ...], pmid: ...}'''
    for pmid in sent_data:
        sent_data[pmid]['words'] = list()
        sent_data[pmid]['wordPos'] = list()
        doc_piv = 0
        for sent in sent_data[pmid]['sentence']:
            wids = list()
            wpos = list()
            sent_piv = 0
            tok = CoNLL_tokenizer(sent)

            for w in tok:
                if len(w) > 20:
                    wids.append(w[:10])
                else:
                    wids.append(w)

                start = doc_piv + sent_piv + sent[sent_piv:].find(w)
                end = start + len(w) - 1
                sent_piv = end - doc_piv + 1
                wpos.append((start, end))
            doc_piv += len(sent)
            sent_data[pmid]['words'].append(wids)
            sent_data[pmid]['wordPos'].append(wpos)

    return sent_data


def isInt(string):
    try:
        int(string)
        return True
    except ValueError:
        return False


def isFloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False


def softmax(logits):
    out = list()
    for logit in logits:
        temp = np.subtract(logit, np.max(logit))
        p = np.exp(temp) / np.sum(np.exp(temp))
        out.append(np.max(p))
    return out


def CoNLL_tokenizer(text):
    rawTok = [t for t in tokenize_regex.split(text) if t]
    assert ''.join(rawTok) == text
    tok = [t for t in rawTok if t != ' ']
    return tok


def sentence_split(text):
    sentences = list()
    sent = ''
    piv = 0
    for idx, char in enumerate(text):
        if char in "?!":
            if idx > len(text) - 3:
                sent = text[piv:]
                piv = -1
            else:
                sent = text[piv:idx + 1]
                piv = idx + 1

        elif char == '.':
            if idx > len(text) - 3:
                sent = text[piv:]
                piv = -1
            elif (text[idx + 1] == ' ') and (
                    text[idx + 2] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ-"' + "'"):
                sent = text[piv:idx + 1]
                piv = idx + 1

        if sent != '':
            toks = CoNLL_tokenizer(sent)
            if len(toks) > 100:
                while True:
                    rawTok = [t for t in tokenize_regex.split(sent) if t]
                    cut = ''.join(rawTok[:200])
                    sent = ''.join(rawTok[200:])
                    sentences.append(cut)

                    if len(CoNLL_tokenizer(sent)) < 100:
                        if sent.strip() == '':
                            sent = ''
                            break
                        else:
                            sentences.append(sent)
                            sent = ''
                            break
            else:
                sentences.append(sent)
                sent = ''

            if piv == -1:
                break

    if piv != -1:
        sent = text[piv:]
        toks = CoNLL_tokenizer(sent)
        if len(toks) > 100:
            while True:
                rawTok = [t for t in tokenize_regex.split(sent) if t]
                cut = ''.join(rawTok[:200])
                sent = ''.join(rawTok[200:])
                sentences.append(cut)

                if len(CoNLL_tokenizer(sent)) < 100:
                    if sent.strip() == '':
                        sent = ''
                        break
                    else:
                        sentences.append(sent)
                        sent = ''
                        break
        else:
            sentences.append(sent)
            sent = ''

    return sentences


def merge_results(data, sent_data, predicDict, logitsDict, rep_ent,
                  is_raw_text=False):
    # b 2   i 3   e 4   s 1   o 0
    for idx, paper in enumerate(data):
        pmid = paper['pmid']
        if is_raw_text:
            content = paper['abstract']
        else:
            if len(paper['abstract']) > 0:
                content = paper['title'] + ' ' + paper['abstract']
            else:
                content = paper['title']

        paper['entities']['drug'] = []
        paper['entities']['gene'] = []
        paper['entities']['disease'] = []
        paper['entities']['species'] = []
        paper['logits'] = dict()

        for dtype in ['disease', 'gene', 'drug', 'species']:
            for sentidx, tags in enumerate(predicDict[dtype][pmid]):
                B_flag = False
                for widx, tag in enumerate(tags):
                    if tag == 'O':
                        if B_flag:
                            tmpSE["end"] = \
                            sent_data[pmid]['wordPos'][sentidx][widx - 1][1]
                            paper['entities'][dtype].append(tmpSE)
                        B_flag = False
                        continue
                    elif tag == 'B':
                        if B_flag:
                            tmpSE["end"] = \
                            sent_data[pmid]['wordPos'][sentidx][widx - 1][1]
                            paper['entities'][dtype].append(tmpSE)
                        tmpSE = {
                            "start": sent_data[pmid]['wordPos'][sentidx][widx][
                                0]}
                        B_flag = True
                    elif tag == "I":
                        continue
                if B_flag:
                    tmpSE["end"] = sent_data[pmid]['wordPos'][sentidx][-1][1]
                    paper['entities'][dtype].append(tmpSE)

            logs = list()
            for t_sent in logitsDict[dtype][pmid]:
                logs.extend(t_sent)
            paper['logits'][dtype] = list()
            for pos in paper['entities'][dtype]:
                if pos['start'] == pos['end']:
                    soft = softmax(logs[len(
                        CoNLL_tokenizer(content[:pos['start']])):len(
                        CoNLL_tokenizer(content[:pos['end']])) + 1])
                    paper['logits'][dtype].append(
                        (pos, float(np.average(soft))))
                else:
                    soft = softmax(logs[len(
                        CoNLL_tokenizer(content[:pos['start']])):len(
                        CoNLL_tokenizer(content[:pos['end']]))])
                    paper['logits'][dtype].append(
                        (pos, float(np.average(soft))))

    if rep_ent:
        return data
    else:
        for idx, paper in enumerate(data):
            pmid = paper['pmid']
            if is_raw_text:
                content = paper['abstract']
            else:
                if len(paper['abstract']) > 0:
                    content = paper['title'] + ' ' + paper['abstract']
                else:
                    content = paper['title']

            dlog = list()
            glog = list()
            clog = list()
            slog = list()

            for t_sent in logitsDict['disease'][pmid]:
                dlog.extend(t_sent)
            for t_sent in logitsDict['gene'][pmid]:
                glog.extend(t_sent)
            for t_sent in logitsDict['drug'][pmid]:
                clog.extend(t_sent)
            for t_sent in logitsDict['species'][pmid]:
                slog.extend(t_sent)

            # d_ent = paper['entities']['disease'][:]
            # g_ent = paper['entities']['gene'][:]
            # c_ent = paper['entities']['drug'][:]
            # s_ent = paper['entities']['species'][:]

            d_ent = paper['entities']['disease'][:]
            g_ent = paper['entities']['gene'][:]
            s_ent = paper['entities']['species'][:]
            for d_e in d_ent:

                removed_d_e = False

                for g_e in g_ent:
                    if d_e['end'] == g_e['end'] and d_e['start'] == g_e[
                        'start']:
                        if d_e['end'] == d_e['start']:
                            d_soft = softmax(
                                dlog[len(CoNLL_tokenizer(
                                    content[:d_e['start']])):len(
                                    CoNLL_tokenizer(content[:d_e['end']])) + 1])
                            g_soft = softmax(
                                glog[len(CoNLL_tokenizer(
                                    content[:g_e['start']])):len(
                                    CoNLL_tokenizer(content[:g_e['end']])) + 1])
                        else:
                            d_soft = softmax(
                                dlog[len(CoNLL_tokenizer(
                                    content[:d_e['start']])):len(
                                    CoNLL_tokenizer(content[:d_e['end']]))])
                            g_soft = softmax(
                                glog[len(CoNLL_tokenizer(
                                    content[:g_e['start']])):len(
                                    CoNLL_tokenizer(content[:g_e['end']]))])
                        if np.average(d_soft) < np.average(g_soft):
                            paper['entities']['disease'].remove(d_e)
                            removed_d_e = True
                            break
                        elif np.average(d_soft) > np.average(g_soft):
                            paper['entities']['gene'].remove(g_e)
                            break

                    elif d_e['end'] < g_e['start']:
                        break

                for s_e in s_ent:
                    if d_e['end'] == s_e['end'] and d_e['start'] == s_e[
                        'start']:
                        if d_e['end'] == d_e['start']:
                            d_soft = softmax(
                                dlog[len(CoNLL_tokenizer(
                                    content[:d_e['start']])):len(
                                    CoNLL_tokenizer(content[:d_e['end']])) + 1])
                            s_soft = softmax(
                                slog[len(CoNLL_tokenizer(
                                    content[:s_e['start']])):len(
                                    CoNLL_tokenizer(content[:s_e['end']])) + 1])
                        else:
                            d_soft = softmax(
                                dlog[len(CoNLL_tokenizer(
                                    content[:d_e['start']])):len(
                                    CoNLL_tokenizer(content[:d_e['end']]))])
                            s_soft = softmax(
                                slog[len(CoNLL_tokenizer(
                                    content[:s_e['start']])):len(
                                    CoNLL_tokenizer(content[:s_e['end']]))])
                        if np.average(d_soft) < np.average(s_soft):
                            if not removed_d_e:
                                paper['entities']['disease'].remove(d_e)
                            break
                        elif np.average(d_soft) > np.average(s_soft):
                            paper['entities']['species'].remove(s_e)
                            break

                    elif d_e['end'] < s_e['start']:
                        break

            g_ent = paper['entities']['gene'][:]
            c_ent = paper['entities']['drug'][:]
            s_ent = paper['entities']['species'][:]
            for g_e in g_ent:

                removed_g_e = False

                for c_e in c_ent:
                    if c_e['end'] == g_e['end'] and c_e['start'] == g_e[
                        'start']:
                        if c_e['end'] == c_e['start']:
                            c_soft = softmax(
                                clog[len(CoNLL_tokenizer(
                                    content[:c_e['start']])):len(
                                    CoNLL_tokenizer(content[:c_e['end']])) + 1])
                            g_soft = softmax(
                                glog[len(CoNLL_tokenizer(
                                    content[:g_e['start']])):len(
                                    CoNLL_tokenizer(content[:g_e['end']])) + 1])
                        else:
                            c_soft = softmax(
                                clog[len(CoNLL_tokenizer(
                                    content[:c_e['start']])):len(
                                    CoNLL_tokenizer(content[:c_e['end']]))])
                            g_soft = softmax(
                                glog[len(CoNLL_tokenizer(
                                    content[:g_e['start']])):len(
                                    CoNLL_tokenizer(content[:g_e['end']]))])
                        if np.average(c_soft) < np.average(g_soft):
                            paper['entities']['drug'].remove(c_e)
                            break
                        elif np.average(c_soft) > np.average(g_soft):
                            paper['entities']['gene'].remove(g_e)
                            removed_g_e = True
                            break

                    elif g_e['end'] < c_e['start']:
                        break

                for s_e in s_ent:
                    if g_e['end'] == s_e['end'] and g_e['start'] == s_e[
                        'start']:
                        if g_e['end'] == g_e['start']:
                            g_soft = softmax(
                                glog[len(CoNLL_tokenizer(
                                    content[:g_e['start']])):len(
                                    CoNLL_tokenizer(content[:g_e['end']])) + 1])
                            s_soft = softmax(
                                slog[len(CoNLL_tokenizer(
                                    content[:s_e['start']])):len(
                                    CoNLL_tokenizer(content[:s_e['end']])) + 1])
                        else:
                            g_soft = softmax(
                                glog[len(CoNLL_tokenizer(
                                    content[:g_e['start']])):len(
                                    CoNLL_tokenizer(content[:g_e['end']]))])
                            s_soft = softmax(
                                slog[len(CoNLL_tokenizer(
                                    content[:s_e['start']])):len(
                                    CoNLL_tokenizer(content[:s_e['end']]))])
                        if np.average(g_soft) < np.average(s_soft):
                            if not removed_g_e:
                                paper['entities']['gene'].remove(g_e)
                            break
                        elif np.average(g_soft) > np.average(s_soft):
                            paper['entities']['species'].remove(s_e)
                            break

                    elif g_e['end'] < s_e['start']:
                        break

            c_ent = paper['entities']['drug'][:]
            d_ent = paper['entities']['disease'][:]
            s_ent = paper['entities']['species'][:]
            for c_e in c_ent:

                removed_c_e = False

                for d_e in d_ent:
                    if c_e['end'] == d_e['end'] and c_e['start'] == d_e[
                        'start']:
                        if d_e['end'] == d_e['start']:
                            d_soft = softmax(
                                dlog[len(CoNLL_tokenizer(
                                    content[:d_e['start']])):len(
                                    CoNLL_tokenizer(content[:d_e['end']])) + 1])
                            c_soft = softmax(
                                clog[len(CoNLL_tokenizer(
                                    content[:c_e['start']])):len(
                                    CoNLL_tokenizer(content[:c_e['end']])) + 1])
                        else:
                            c_soft = softmax(
                                clog[len(CoNLL_tokenizer(
                                    content[:c_e['start']])):len(
                                    CoNLL_tokenizer(content[:c_e['end']]))])
                            d_soft = softmax(
                                dlog[len(CoNLL_tokenizer(
                                    content[:d_e['start']])):len(
                                    CoNLL_tokenizer(content[:d_e['end']]))])
                        if np.average(c_soft) < np.average(d_soft):
                            paper['entities']['drug'].remove(c_e)
                            removed_c_e = True
                            break
                        elif np.average(c_soft) > np.average(d_soft):
                            paper['entities']['disease'].remove(d_e)
                            break

                    elif c_e['end'] < d_e['start']:
                        break

                for s_e in s_ent:
                    if c_e['end'] == s_e['end'] and c_e['start'] == s_e[
                        'start']:
                        if c_e['end'] == c_e['start']:
                            c_soft = softmax(
                                clog[len(CoNLL_tokenizer(
                                    content[:c_e['start']])):len(
                                    CoNLL_tokenizer(content[:c_e['end']])) + 1])
                            s_soft = softmax(
                                slog[len(CoNLL_tokenizer(
                                    content[:s_e['start']])):len(
                                    CoNLL_tokenizer(content[:s_e['end']])) + 1])
                        else:
                            c_soft = softmax(
                                clog[len(CoNLL_tokenizer(
                                    content[:c_e['start']])):len(
                                    CoNLL_tokenizer(content[:c_e['end']]))])
                            s_soft = softmax(
                                slog[len(CoNLL_tokenizer(
                                    content[:s_e['start']])):len(
                                    CoNLL_tokenizer(content[:s_e['end']]))])
                        if np.average(c_soft) < np.average(s_soft):
                            if not removed_c_e:
                                paper['entities']['drug'].remove(c_e)
                            break
                        elif np.average(c_soft) > np.average(s_soft):
                            paper['entities']['species'].remove(s_e)
                            break

                    elif c_e['end'] < s_e['start']:
                        break
        return data


def detokenize(tokens, predicts, logits):
    pred = dict({
        'toks': tokens[:],
        'labels': predicts[:],
        'logits': logits[:]
    })  # dictionary for predicted tokens and labels.

    bert_toks = list()
    bert_labels = list()
    bert_logits = list()
    tmp_p = list()
    tmp_l = list()
    tmp_s = list()
    for t, l, s in zip(pred['toks'], pred['labels'], pred['logits']):
        if t == '[CLS]':  # non-text tokens will not be evaluated.
            continue
        elif t == '[SEP]':  # newline
            bert_toks.append(tmp_p)
            bert_labels.append(tmp_l)
            bert_logits.append(tmp_s)
            tmp_p = list()
            tmp_l = list()
            tmp_s = list()
            continue
        elif t[:2] == '##':  # if it is a piece of a word (broken by Word Piece tokenizer)
            tmp_p[-1] = tmp_p[-1] + t[2:]  # append pieces
        else:
            tmp_p.append(t)
            tmp_l.append(l)
            tmp_s.append(s)

    return bert_toks, bert_labels, bert_logits
