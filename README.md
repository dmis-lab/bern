# BERN
BERN is a BioBERT-based multi-type NER tool that also supports normalization of extracted entities. This repository contains the official implementation of BERN. You can use BERN at https://bern.korea.ac.kr, or host your own server by following the description below. Please refer to our [paper (Kim et al., IEEE Access 2019)](https://doi.org/10.1109/ACCESS.2019.2920708) for more details. This project is done by [DMIS Laboratory](https://dmis.korea.ac.kr) at Korea University.

**Fixed our gene normalizer to respond to issues between 2020-03-12 and 2020-03-13**
1. Download gene_normalizer_19.jar at [this URL](https://drive.google.com/open?id=1ZTKJyRLBeqG2ioTtUqvmW0C_H6PmHZGl) and place (overwrite) the file under normalization/resources/normalizers/gene directory.  
2. Stop normalizers by running stop_normalizers.sh  
3. Start the normalizers by running load_dicts.sh  

**Done - Server down due to air conditioning problems in our server room 2019-10-10 - 2019-10-11 7:55 AM (UTC-0)**  

**Fixed our disease normalizer 2019-08-19, 2019-08-10 and 2019-08-02 issues**
1. Download disease_normalizer_19.jar at [this URL](https://drive.google.com/open?id=1YbAanyQJ24PPBOu0NO8a1aCxWLdlQhk-) and place the file under normalization/resources/normalizers/disease directory.   
2. Stop normalizers by running stop_normalizers.sh and restart the normalizers by running load_dicts.sh 

**Done - Server check 2019-07-18 8:20 AM - 1:30 PM (UTC-0)**

![BERN](https://github.com/dmis-lab/bern/blob/master/bern_overview.jpg?raw=true)
<p align="center">Overview of BERN.</p>

The description below gives instructions on hosting your own BERN. Please refer to https://bern.korea.ac.kr for the RESTful Web service of BERN.

## Requirements
* Environment
    * Python >= 3.6 
    * CUDA 9 or higher
* Main components
    * [BioBERT NER models (Lee et al., 2019)](https://arxiv.org/abs/1901.08746)
    * [tmTool APIs (Wei et al., 2016)](https://www.ncbi.nlm.nih.gov/research/bionlp/APIs/) 
    * [GNormPlus (Wei et al., 2015)](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/gnormplus/)
    * [tmVar 2.0 (Wei et al., 2018)](https://www.ncbi.nlm.nih.gov/research/bionlp/Tools/tmvar/)
    * [TensorFlow 1.13.1](https://github.com/tensorflow/tensorflow/releases/tag/v1.13.1)

Note that you will need at least 66 GB of free disk space and 32 GB or more RAM.

##  Installation
* Clone this repo
```
cd
git clone https://github.com/dmis-lab/bern.git
```

* Install python packages
```
pip3 install -r requirements.txt --user
```

* Install GNormPlus & run GNormPlusServer.jar
    * FYI: Download Google Drive files with WGET: https://gist.github.com/iamtekeste/3cdfd0366ebfd2c0d805#gistcomment-2316906
```
cd ~/bern
wget https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/GNormPlus/GNormPlusJava.zip
unzip GNormPlusJava.zip

cd GNormPlusJava
wget -O ./crfpp-0.58.tar.gz https://drive.google.com/uc?id=0B4y35FiV1wh7QVR6VXJ5dWExSTQ
tar xvfz crfpp-0.58.tar.gz
cp -rf CRF++-0.58/* CRF
cd CRF
sh ./configure
make
sudo make install

cd ..
chmod 764 Ab3P
# chmod 764 CRF/crf_test

# Set FocusSpecies to 9606 (Human)
sed -i 's/= All/= 9606/g' setup.txt; echo "FocusSpecies: from All to 9606 (Human)"
sh Installation.sh

rm -r CRF++-0.58
rm crfpp-0.58.tar.gz

# Download GNormPlusServer.jar
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1g-JlhqeDIlZX5YFk8Y27_M8BXUXcQRSX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1g-JlhqeDIlZX5YFk8Y27_M8BXUXcQRSX" -O GNormPlusServer.jar && rm -rf /tmp/cookies.txt

# Start GNormPlusServer
nohup java -Xmx16G -Xms16G -jar GNormPlusServer.jar 18895 >> ~/bern/logs/nohup_gnormplus.out 2>&1 &
```


* Install tmVar2 & run tmVar2Server.jar
```
cd ~/bern
wget ftp://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/tmVar2/tmVarJava.zip
unzip tmVarJava.zip

cd tmVarJava
wget -O ./crfpp-0.58.tar.gz https://drive.google.com/uc?id=0B4y35FiV1wh7QVR6VXJ5dWExSTQ
tar xvfz crfpp-0.58.tar.gz
cp -rf CRF++-0.58/* CRF
cd CRF
sh ./configure
make
sudo make install

cd ..
chmod 764 CRF/crf_test

sh Installation.sh

rm -r CRF++-0.58
rm crfpp-0.58.tar.gz

# Download tmVar2Server.jar
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kQYzLHLFLsU9qKpRRGjXkIYmaYK6bPJm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kQYzLHLFLsU9qKpRRGjXkIYmaYK6bPJm" -O tmVar2Server.jar && rm -rf /tmp/cookies.txt

# Download dependencies
wget https://repo1.maven.org/maven2/org/xerial/sqlite-jdbc/3.20.0/sqlite-jdbc-3.20.0.jar
wget https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.5.2/stanford-corenlp-3.5.2.jar

# Start tmVar2Server
nohup java -Xmx8G -Xms8G -jar tmVar2Server.jar 18896 >> ~/bern/logs/nohup_tmvar.out 2>&1 &
```
 
* Download normalization resources and pre-trained BioBERT NER models
```
cd ~/bern/scripts
sh download_norm.sh
sh download_biobert_ner_models.sh
```

* Run named entity normalizers
```
cd ..
sh load_dicts.sh
```

* Run BERN server
```
# Check your GPU number(s)
echo $CUDA_VISIBLE_DEVICES

# Set your GPU number(s)
export CUDA_VISIBLE_DEVICES=0

# Run BERN
# Please check gnormplus_home directory and tmvar2_home directory.
nohup python3 -u server.py --port 8888 --gnormplus_home ~/bern/GNormPlusJava --gnormplus_port 18895 --tmvar2_home ~/bern/tmVarJava --tmvar2_port 18896 >> logs/nohup_BERN.out 2>&1 &

# Print logs
tail -F logs/nohup_BERN.out
```

* Usage
    * PMID(s) (HTTP GET)
        * http://\<YOUR_SERVER_ADDRESS>:8888/?pmid=\<a PMID or comma seperate PMIDs>&format=\<json or pubtator>
        * Example: http://\<YOUR_SERVER_ADDRESS>:8888/?pmid=30429607&format=json&indent=true
        * Example: http://\<YOUR_SERVER_ADDRESS>:8888/?pmid=30429607&format=pubtator
        * Example: http://\<YOUR_SERVER_ADDRESS>:8888/?pmid=30429607,29446767&format=json&indent=true
    * Raw text (HTTP POST)
        * POST Address: http://\<YOUR_SERVER_ADDRESS>:8888
        * Set key, value of a body as follows:          
        ```
        import requests
        import json
        body_data = {"param": json.dumps({"text": "CLAPO syndrome: identification of somatic activating PIK3CA mutations and delineation of the natural history and phenotype. PURPOSE: CLAPO syndrome is a rare vascular disorder characterized by capillary malformation of the lower lip, lymphatic malformation predominant on the face and neck, asymmetry, and partial/generalized overgrowth. Here we tested the hypothesis that, although the genetic cause is not known, the tissue distribution of the clinical manifestations in CLAPO seems to follow a pattern of somatic mosaicism. METHODS: We clinically evaluated a cohort of 13 patients with CLAPO and screened 20 DNA blood/tissue samples from 9 patients using high-throughput, deep sequencing. RESULTS: We identified five activating mutations in the PIK3CA gene in affected tissues from 6 of the 9 patients studied; one of the variants (NM_006218.2:c.248T>C; p.Phe83Ser) has not been previously described in developmental disorders. CONCLUSION: We describe for the first time the presence of somatic activating PIK3CA mutations in patients with CLAPO. We also report an update of the phenotype and natural history of the syndrome."})}
        response = requests.post('http://<YOUR_SERVER_ADDRESS>:8888', data=body_data)
        result_dict = response.json()
        print(result_dict)
        ```

## Result
<details>
    <summary>See a result example in JSON (PMID:29446767) </summary>
<pre>
[
    {
        "denotations": [
            {
                "id": [
                    "MESH:C567763",
                    "BERN:262813101"
                ],
                "obj": "disease",
                "span": {
                    "begin": 0,
                    "end": 13
                }
            },
            {
                "id": [
                    "MIM:171834",
                    "HGNC:8975",
                    "Ensembl:ENSG00000121879",
                    "BERN:324295302"
                ],
                "obj": "gene",
                "span": {
                    "begin": 53,
                    "end": 58
                }
            },
            {
                "id": [
                    "MESH:C567763",
                    "BERN:262813101"
                ],
                "obj": "disease",
                "span": {
                    "begin": 133,
                    "end": 146
                }
            },
            {
                "id": [
                    "MESH:D014652",
                    "BERN:256572101"
                ],
                "obj": "disease",
                "span": {
                    "begin": 158,
                    "end": 174
                }
            },
            {
                "id": [
                    "MESH:C567763",
                    "BERN:262813101"
                ],
                "obj": "disease",
                "span": {
                    "begin": 193,
                    "end": 231
                }
            },
            {
                "id": [
                    "MESH:C567763",
                    "BERN:262813101"
                ],
                "obj": "disease",
                "span": {
                    "begin": 234,
                    "end": 288
                }
            },
            {
                "id": [
                    "MESH:C567763",
                    "BERN:262813101"
                ],
                "obj": "disease",
                "span": {
                    "begin": 589,
                    "end": 593
                }
            },
            {
                "id": [
                    "MIM:171834",
                    "HGNC:8975",
                    "Ensembl:ENSG00000121879",
                    "BERN:324295302"
                ],
                "obj": "gene",
                "span": {
                    "begin": 748,
                    "end": 758
                }
            },
            {
                "id": [
                    "CUI-less"
                ],
                "mutationType": "ProteinMutation",
                "normalizedName": "p.F83S;CorrespondingGene:5290",
                "obj": "mutation",
                "span": {
                    "begin": 857,
                    "end": 866
                }
            },
            {
                "id": [
                    "BERN:257523801"
                ],
                "obj": "disease",
                "span": {
                    "begin": 906,
                    "end": 928
                }
            },
            {
                "id": [
                    "CUI-less"
                ],
                "obj": "gene",
                "span": {
                    "begin": 1009,
                    "end": 1024
                }
            },
            {
                "id": [
                    "MESH:C567763",
                    "BERN:262813101"
                ],
                "obj": "disease",
                "span": {
                    "begin": 1043,
                    "end": 1047
                }
            }
        ],
        "elapsed_time": {
            "ner": 0.611,
            "normalization": 0.218,
            "tmtool": 1.281,
            "total": 2.111
        },
        "project": "BERN",
        "sourcedb": "PubMed",
        "sourceid": "29446767",
        "text": "CLAPO syndrome: identification of somatic activating PIK3CA mutations and delineation of the natural history and phenotype. PURPOSE: CLAPO syndrome is a rare vascular disorder characterized by capillary malformation of the lower lip, lymphatic malformation predominant on the face and neck, asymmetry, and partial/generalized overgrowth. Here we tested the hypothesis that, although the genetic cause is not known, the tissue distribution of the clinical manifestations in CLAPO seems to follow a pattern of somatic mosaicism. METHODS: We clinically evaluated a cohort of 13 patients with CLAPO and screened 20 DNA blood/tissue samples from 9 patients using high-throughput, deep sequencing. RESULTS: We identified five activating mutations in the PIK3CA gene in affected tissues from 6 of the 9 patients studied; one of the variants (NM_006218.2:c.248T>C; p.Phe83Ser) has not been previously described in developmental disorders. CONCLUSION: We describe for the first time the presence of somatic activating PIK3CA mutations in patients with CLAPO. We also report an update of the phenotype and natural history of the syndrome.",
        "timestamp": "Thu Jul 04 06:15:27 +0000 2019"
    }
]
</pre>
</details>


## Restart
```
# Start GNormPlusServer
cd ~/bern/GNormPlusJava
nohup java -Xmx16G -Xms16G -jar GNormPlusServer.jar 18895 >> ~/bern/logs/nohup_gnormplus.out 2>&1 &

# Start tmVar2Server
cd ~/bern/tmVarJava
nohup java -Xmx8G -Xms8G -jar tmVar2Server.jar 18896 >> ~/bern/logs/nohup_tmvar.out 2>&1 &

# Start normalizers
cd ~/bern/
sh load_dicts.sh

# Check your GPU number(s)
echo $CUDA_VISIBLE_DEVICES

# Set your GPU number(s)
export CUDA_VISIBLE_DEVICES=0

# Run BERN
nohup python3 -u server.py --port 8888 --gnormplus_home ~/bern/GNormPlusJava --gnormplus_port 18895 --tmvar2_home ~/bern/tmVarJava --tmvar2_port 18896 >> logs/nohup_BERN.out 2>&1 &

# Print logs
tail -F logs/nohup_BERN.out
```


## Troubleshooting
* Trouble: It takes a long time to get results. 
    * Solution: Make sure TensorFlow is using a GPU.
    For more details, visit https://stackoverflow.com/questions/42326748/tensorflow-on-gpu-no-known-devices-despite-cudas-devicequery-returning-a-pas/48079860#48079860

## Monitoring
* List processes (every 5s)
```
watch -n 5 "ps auxww | egrep 'python|java|node' | grep -v grep"
```

* Periodic HTTPS GET checker
    * Permission setting
    ```
    chmod +x scripts/bern_checker.sh
    ```

    * crontab (every 30 min)
    ```
    crontab -e
    */30 * * * * /home/<YOUR_ACCOUNT>/bern/scripts/bern_checker.sh >> /home/<YOUR_ACCOUNT>/bern/logs/bern_checker.out 2>&1
    ```

## Bug report
Add a new issue to https://github.com/dmis-lab/bern/issues

## Contact
donghyeon@korea.ac.kr

## Citation
* Please cite the following two papers if you use BERN on your work.
```
@article{kim2019neural,
  title={A Neural Named Entity Recognition and Multi-Type Normalization Tool for Biomedical Text Mining},
  author={Kim, Donghyeon and Lee, Jinhyuk and So, Chan Ho and Jeon, Hwisang and Jeong, Minbyul and Choi, Yonghwa and Yoon, Wonjin and Sung, Mujeen and and Kang, Jaewoo},
  journal={IEEE Access},
  volume={7},
  pages={73729--73740},
  year={2019},
  publisher={IEEE}
}

@article{10.1093/bioinformatics/btz682,
    author = {Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
    title = "{BioBERT: a pre-trained biomedical language representation model for biomedical text mining}",
    journal = {Bioinformatics},
    year = {2019},
    month = {09},
    issn = {1367-4803},
    doi = {10.1093/bioinformatics/btz682},
    url = {https://doi.org/10.1093/bioinformatics/btz682},
}
```
