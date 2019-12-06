import re


class MiRNAFinder:
    def __init__(self, dict_path):

        # config
        prefix_list = ['mir', 'mir-',
                       'mirna', 'mirna-',
                       'microrna', 'microrna-',
                       'micro rna', 'micro-rna-',
                       'let-', 'mirna-let-', 'mir-let-',
                       'hsa-mir-', 'mmu-', "hsa-let-"]
        suffix_list = ['[-]?[a-c]-[12]-[35]p', '[-]?[a-c]-[35]p', '-[12]-[35]p',
                       '[a-c]', '-[12]', '-[35]p', '[*]', '[0-9]+']

        ten_species = ['bta-mir-', 'rno-mir-', 'gga-mir-', 'cel-mir-',
                       'ssc-mir-', 'ebv-mir-', 'mdv1-mir-', 'hcmv-mir-']
        prefix_list = ten_species + prefix_list

        # make regex
        p_list = list()
        for pre in prefix_list:
            for suf in suffix_list:
                p_list.append(r'\b%s[0-9]{1,3}%s\b' % (pre, suf))
        self.mirna_pattern = re.compile(r'|'.join(p_list), re.I)

        self.code2mirs = dict()
        self.mir2id = dict()
        self.mirna_id2accession = dict()

        with open(dict_path, 'r') as f:
            for l in f:
                cols = l[:-1].split('\t')

                if 3 > len(cols):
                    print('Invalid line:', l[:-1])
                    continue

                mirna_id = int(cols[0])
                mirbase_ids = cols[1].split('|')
                accession = cols[2]

                if '' != accession:
                    self.mirna_id2accession[mirna_id] = 'miRBase:' + accession

                for mirbase_id in mirbase_ids:
                    code = re.search(r"[0-9]+.*", mirbase_id)
                    if code is None:
                        continue
                    else:
                        code = code.group()

                    if code not in self.code2mirs:
                        self.code2mirs[code] = list()

                    self.code2mirs[code].append(mirbase_id)
                    self.mir2id[mirbase_id] = mirna_id

        print('code2mirs size', len(self.code2mirs))
        print('mirbase_id2mirna_id size', len(self.mir2id))
        print('mirna_id2accession size', len(self.mirna_id2accession))

    def normalize(self, miRNA):
        code = re.search(r"[0-9]+.*", miRNA).group()
        code = code.replace("-3p", "").replace("-5p", "").replace("-3P",
                                                                  "").replace(
            "-5P",
            "")

        # code_candidates = [code, code.lower()]

        selected_code = None

        if code in self.code2mirs:
            selected_code = code
        elif code.lower() in self.code2mirs:
            selected_code = code.lower()
        elif re.match("[0-9]+", code).group() in self.code2mirs:
            selected_code = re.match("[0-9]+", code).group()
        elif code.lstrip("0") in self.code2mirs:
            selected_code = code.lstrip("0")
        else:
            for ref_code in self.code2mirs:
                if re.match(code + "[a-z]+.*", ref_code):
                    selected_code = ref_code
                    break  # first matched code

        if selected_code is None:
            # print(miRNA)
            return ""

        # normalized_mir = ""

        if len(self.code2mirs[
                   selected_code]) == 1:  # Only one reference mirna for the code.
            normalized_mir = self.code2mirs[selected_code][0]

        elif (miRNA.lower()[:3] == "mir") or (
                miRNA.lower()[
                :5] == "micro"):  # No species such as hsa-, mmu-, ...
            if "hsa-mir-" + selected_code in self.code2mirs[selected_code]:
                normalized_mir = "hsa-mir-" + selected_code
            elif "mmu-mir-" + selected_code in self.code2mirs[selected_code]:
                normalized_mir = "mmu-mir-" + selected_code
            else:
                normalized_mir = self.code2mirs[selected_code][0]

        elif miRNA.lower()[:3] == "let":  # Starts with let-
            if "hsa-let-" + selected_code in self.code2mirs[selected_code]:
                normalized_mir = "hsa-let-" + selected_code
            elif "mmu-let-" + selected_code in self.code2mirs[selected_code]:
                normalized_mir = "mmu-let-" + selected_code
            else:
                normalized_mir = self.code2mirs[selected_code][0]

        elif miRNA.lower()[:3] == "hsa":  # hsa-xxx
            if "hsa-mir-" + selected_code in self.code2mirs[selected_code]:
                normalized_mir = "hsa-mir-" + selected_code
            else:
                normalized_mir = self.code2mirs[selected_code][0]

        elif miRNA.lower()[:3] == "mmu":  # mmu-xxx
            if "hsa-mir-" + selected_code in self.code2mirs[selected_code]:
                normalized_mir = "mmu-mir-" + selected_code
            else:
                normalized_mir = self.code2mirs[selected_code][0]

        else:
            # print(miRNA, code2mirs[selected_code])
            return ""

        return self.mir2id[normalized_mir]

    def tag(self, text):
        mirna_list = list()
        for idx, m in enumerate(self.mirna_pattern.finditer(text)):
            mirid = self.normalize(m.group())
            m_span = m.span()

            bern_mirid = 'BERN:{}'.format(mirid)
            id_fin = \
                self.mirna_id2accession[mirid] + '\t' + bern_mirid \
                if mirid in self.mirna_id2accession else bern_mirid

            mirna_list.append({
                "start": m_span[0],
                "id": id_fin,
                "end": m_span[1]
            })
        return mirna_list


if __name__ == '__main__':
    mf = MiRNAFinder('../normalization/resources/dictionary/best_dict_miRNA.txt')
    print(mf.tag('Evaluation of salivary and plasma microRNA expression in patients with Sjogren\'s syndrome, and correlations with clinical and ultrasonographic outcomes. To correlate the expression of microRNAs (miRNAs) 146a/b, 16, the 17-92 cluster and 181a in salivary and plasma samples taken from primary Sjogren\'s syndrome (pSS) patients with clinical, laboratory and ultrasound findings. Plasma and salivary samples were collected from 28 patients with pSS according to 2012 ACR and/or 2016 ACR/EULAR criteria (27 females, mean age 64.4 10.1 years, mean disease duration 10.7 6.9 years), and from 23 healthy subjects used as controls. The following patient data were recorded: ESSDAI and ESSPRI scores, anti-SSA and anti-SSB antibody status and laboratory data, Schirmer\'s test, ultrasound scores of the four major salivary glands according to Cornec et al., and concomitant treatments. The retro-transcribed and quantified miRNAs were: miR16-5p, miR17-5p, miR18a-5p, miR19a-5p, miR19b-1-5p, miR20a, miR92-5p, miR146a-5p, miR146b-5p, miR181a-5p. SS patients had higher expression of salivary miR146a than gender- and age-matched controls (p=0.01). Spearman\'s regression analysis revealed that salivary miR146b was significantly more expressed in the patients with worse ESSPRI scores (p=0.02), whereas salivary miR17 and 146b and plasma miR17 expression was lower in the patients with higher ultrasound scores (respectively p=0.01, p=0.01 and p=0.04). Salivary miR18a expression was significantly increased in the patients who were anti-La/SSB positive (p=0.04). Neither salivary nor plasma miRNAs correlated with disease duration or concomitant therapies. Our data show that salivary mi146a may represent a marker of the disease, and that the expression of salivary miR17, 18a and 146b may be altered in patients with pSS, and associated with worse ultrasound and ESSPRI scores and anti-La/SSB positivity.'))
