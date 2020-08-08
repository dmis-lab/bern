#!/bin/bash

nohup python3 normalizers/chemical_normalizer.py >> logs/nohup_drug_normalize.out 2>&1 &
nohup python3 normalizers/species_normalizer.py >> logs/nohup_species_normalize.out 2>&1 &
nohup python3 normalizers/mutation_normalizer.py >> logs/nohup_mutation_normalize.out 2>&1 &

# Disease (working dir: normalization/)
cd normalization
nohup java -Xmx16G -jar resources/normalizers/disease/disease_normalizer_19.jar >> ../logs/nohup_disease_normalize.out 2>&1 &

# Gene (working dir: normalization/resources/normalizers/gene/, port:18888)
cd resources/normalizers/gene/
nohup java -Xmx20G -jar gnormplus-normalization_19.jar >> ../../../../logs/nohup_gene_normalize.out 2>&1 &

ps auxww | egrep 'python|java' | grep -v grep