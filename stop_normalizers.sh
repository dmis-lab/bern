#!/usr/bin/env bash

pid=`ps auxww | grep GNormPlus_180921.jar | grep -v grep | awk '{print $2}' | sort -r`
if [ "$pid" != "" ]; then
  kill -9 "$pid"
  echo "Stopped GNormPlus_180921.jar"
else
  echo "No GNormPlus_180921.jar found to stop."
fi

pid=`ps auxww | grep species_normalizer.py | grep -v grep | awk '{print $2}' | sort -r`
if [ "$pid" != "" ]; then
  kill -9 "$pid"
  echo "Stopped species_normalizer.py"
else
  echo "No species_normalizer.py found to stop."
fi

pid=`ps auxww | grep chemical_normalizer.py | grep -v grep | awk '{print $2}' | sort -r`
if [ "$pid" != "" ]; then
  kill -9 "$pid"
  echo "Stopped chemical_normalizer.py"
else
  echo "No chemical_normalizer.py found to stop."
fi

pid=`ps auxww | grep mutation_normalizer.py | grep -v grep | awk '{print $2}' | sort -r`
if [ "$pid" != "" ]; then
  kill -9 "$pid"
  echo "Stopped mutation_normalizer.py"
else
  echo "No mutation_normalizer.py found to stop."
fi

pid=`ps auxww | grep disease_normalizer_19 | grep -v grep | awk '{print $2}' | sort -r`
if [ "$pid" != "" ]; then
  kill -9 "$pid"
  echo "Stopped disease_normalizer"
else
  echo "No disease_normalizer found to stop."
fi


rm -rf normalization/resources/inputs/gene/*
rm -rf normalization/resources/inputs/chemical/*
rm -rf normalization/resources/inputs/disease/*
rm -rf normalization/resources/inputs/species/*
rm -rf normalization/resources/inputs/mutation/*

rm -rf normalization/resources/outputs/gene/*
rm -rf normalization/resources/outputs/chemical/*
rm -rf normalization/resources/outputs/disease/*
rm -rf normalization/resources/outputs/species/*
rm -rf normalization/resources/outputs/mutation/*

rm -rf normalization/resources/normalizers/gene/tmp/*

# sh reset.sh