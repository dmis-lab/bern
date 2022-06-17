FROM python:3.7.3-stretch

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
RUN pip install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y wget && apt-get install coreutils && apt-get install unzip  && apt-get install -y default-jre

# Install GNormPlus
RUN wget https://www.ncbi.nlm.nih.gov/CBBresearch/Lu/Demo/tmTools/download/GNormPlus/GNormPlusJava.zip \
&& unzip GNormPlusJava.zip && rm GNormPlusJava.zip
RUN wget -O /usr/src/app/GNormPlusJava/crfpp-0.58.tar.gz https://drive.google.com/uc?id=0B4y35FiV1wh7QVR6VXJ5dWExSTQ
RUN tar xvfz /usr/src/app/GNormPlusJava/crfpp-0.58.tar.gz
RUN cp -rf /usr/src/app/CRF++-0.58/* /usr/src/app/GNormPlusJava/CRF
RUN cd /usr/src/app/GNormPlusJava/CRF/ && sh ./configure
RUN cd /usr/src/app/GNormPlusJava/CRF/ && make 
RUN cd /usr/src/app/GNormPlusJava/CRF/ && make install
RUN chmod 764 /usr/src/app/GNormPlusJava/Ab3P
RUN cd /usr/src/app/GNormPlusJava/ && sed -i 's/= All/= 9606/g' setup.txt; echo "FocusSpecies: from All to 9606 (Human)"
RUN cd /usr/src/app/GNormPlusJava/ && sh Installation.sh
RUN cd /usr/src/app/GNormPlusJava/ && wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1g-JlhqeDIlZX5YFk8Y27_M8BXUXcQRSX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1g-JlhqeDIlZX5YFk8Y27_M8BXUXcQRSX" -O GNormPlusServer.jar && rm -rf /tmp/cookies.txt

# Install tmVar2
RUN cd /usr/src/app/ && wget ftp://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/tmVar2/tmVarJava.zip \
&& unzip tmVarJava.zip  && rm tmVarJava.zip
RUN cp -rf /usr/src/app/CRF++-0.58/* /usr/src/app/tmVarJava/CRF && rm -R /usr/src/app/CRF++-0.58 && rm /usr/src/app/GNormPlusJava/crfpp-0.58.tar.gz
RUN cd /usr/src/app/tmVarJava/CRF/ && sh ./configure
RUN cd /usr/src/app/tmVarJava/CRF/ && make 
RUN cd /usr/src/app/tmVarJava/CRF/ && make install
RUN chmod 764 /usr/src/app/tmVarJava/CRF/crf_test
RUN cd /usr/src/app/tmVarJava/ && sh Installation.sh
RUN cd /usr/src/app/tmVarJava/ && wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kQYzLHLFLsU9qKpRRGjXkIYmaYK6bPJm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kQYzLHLFLsU9qKpRRGjXkIYmaYK6bPJm" -O tmVar2Server.jar && rm -rf /tmp/cookies.txt
RUN cd /usr/src/app/tmVarJava/ && wget https://repo1.maven.org/maven2/org/xerial/sqlite-jdbc/3.20.0/sqlite-jdbc-3.20.0.jar
RUN cd /usr/src/app/tmVarJava/ && wget https://repo1.maven.org/maven2/edu/stanford/nlp/stanford-corenlp/3.5.2/stanford-corenlp-3.5.2.jar

# Copy Project into App
COPY . /usr/src/app
RUN mkdir /usr/src/app/logs
RUN touch /usr/src/app/logs/nohup_BERN.out \
&& touch /usr/src/app/logs/nohup_disease_normalize.out \
&& touch /usr/src/app/logs/nohup_drug_normalize.out \
&& touch /usr/src/app/logs/nohup_gene_normalize.out \
&& touch /usr/src/app/logs/nohup_gnormplus.out \
&& touch /usr/src/app/logs/nohup_mutation_normalize.out \
&& touch /usr/src/app/logs/nohup_species_normalize.out \
&& touch /usr/src/app/logs/nohup_tmvar.out

# Download normalization resources and pre-trained BioBERT NER models
RUN cd /usr/src/app/scripts && sh download_norm.sh \
&& cd /usr/src/app/scripts && sh download_biobert_ner_models.sh

EXPOSE 8888

# Execute
CMD (cd /usr/src/app/GNormPlusJava/ && nohup java -Xmx16G -Xms16G -jar /usr/src/app/GNormPlusJava/GNormPlusServer.jar 18895 >> /usr/src/app/logs/nohup_gnormplus.out 2>&1 &) \
&& (cd /usr/src/app/tmVarJava/ && nohup java -Xmx8G -Xms8G -jar /usr/src/app/tmVarJava/tmVar2Server.jar 18896 >> /usr/src/app/logs/nohup_tmvar.out 2>&1 &) \
&& (cd /usr/src/app/ && sh load_dicts.sh) \
&& echo $CUDA_VISIBLE_DEVICES && export CUDA_VISIBLE_DEVICES=0 \
&& (nohup python3 -u /usr/src/app/server.py --port 8888 --gnormplus_home /usr/src/app/GNormPlusJava --gnormplus_port 18895 --tmvar2_home /usr/src/app/tmVarJava --tmvar2_port 18896 >> /usr/src/app/logs/nohup_BERN.out 2>&1 &) \
&& tail -f logs/nohup_BERN.out