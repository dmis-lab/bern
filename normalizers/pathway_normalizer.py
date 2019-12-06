import re


class PathwayFinder:
    def __init__(self, dict_path):

        self.pathway2id = dict()
        self.pathway_id2keggid = dict()

        with open(dict_path, 'r') as f:
            for l in f:
                cols = l[:-1].split('\t')

                if 2 > len(cols):
                    print('Invalid line:', l[:-1])
                    continue

                pathway_id = int(cols[0])
                name = cols[1]
                if "pathway" not in name:
                    name += ' pathway'
                if name in self.pathway2id:
                    # print(name, pathway_id)
                    continue
                self.pathway2id[name] = pathway_id

                if 3 <= len(cols) and '' != cols[2]:
                    self.pathway_id2keggid[pathway_id] = cols[2]

        extended = list()
        for pname in self.pathway2id:
            if "pathway" not in pname:
                extended.append(pname + ' pathway')
            else:
                extended.append(pname)
        self.regex = re.compile('|'.join(extended))

        print('# of pathway regex', len(extended))

    def tag(self, text):
        pathways = list()
        for idx, m in enumerate(self.regex.finditer(text)):
            pathway_id = self.pathway2id[m.group()]
            m_span = m.span()

            bern_pathway_id = 'BERN:{}'.format(pathway_id)
            id_fin = \
                self.pathway_id2keggid[pathway_id] + '\t' + bern_pathway_id \
                if pathway_id in self.pathway_id2keggid else bern_pathway_id

            pathways.append({
                "start": m_span[0],
                "id": id_fin,
                "end": m_span[1]
            })
        return pathways


if __name__ == '__main__':
    pf = PathwayFinder('../normalization/resources/dictionary/best_dict_Pathway.txt')
    print(pf.tag('Transferrin receptor-involved HIF-1 signaling pathway in cervical cancer. Cervical cancer is one of the most prevalent gynecologic malignancies and has remained an intractable cancer over the past decades. We analyzed the aberrant expression patterns of cervical cancer using RNA-Seq data from The Cancer Genome Atlas (TCGA). A total of 3352 differently expressed genes (DEGs) were identified in 306 cervical cancer samples compared with 3 control samples, with 1401 genes upregulated and 1951 downregulated. Under Kaplan-Meier analysis, 76 out of these DEGs with a significantly different survival rate between patients with high and low expression groups were picked out and uploaded to perform Kyoto Encyclopedia of Genes and Genomes (KEGG) pathway enrichment, which identified a transferrin receptor (TFRC)-involved HIF-1 signaling pathway (p < 0.05). Clinical data analysis showed that high TFRC expression in cervical cancers was associated with incrementally advanced stage, tumor status, and lymph nodes (all p-values <0.05), while multivariate analysis revealed that TFRC remained an independent prognostic variable for poor overall survival. In conclusion, our data indicated that the TFRC-involved HIF-1 signaling pathway may play a crucial role in cervical cancer.'))
