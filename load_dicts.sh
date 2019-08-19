#!/bin/bash

nohup python3 normalizers/chemical_normalizer.py >> logs/nohup_drug.out 2>&1 &
nohup python3 normalizers/species_normalizer.py >> logs/nohup_species.out 2>&1 &
nohup python3 normalizers/mutation_normalizer.py >> logs/nohup_mutation.out 2>&1 &

# Disease (working dir: normalization/)
cd normalization
nohup java -Xmx16G -jar resources/normalizers/disease/disease_normalizer_190819.jar >> ../logs/nohup_disease.out 2>&1 &

# Gene (working dir: normalization/resources/normalizers/gene/, port:18888)
cd resources/normalizers/gene/
nohup java -Xmx20G -jar GNormPlus_180921.jar >> ../../../../logs/nohup_gene.out 2>&1 &

ps auxww | egrep 'python|java' | grep -v grep