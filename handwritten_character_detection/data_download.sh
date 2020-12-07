#!/bin/bash

if [ ! -r "archive.zip" ]
then
  echo "!! Cannot download the data directly !!"
  echo "   Please go to https://www.kaggle.com/sachinpatel21/az-handwritten-alphabets-in-csv-format"
  echo "   Download the zipped archive (archive.zip) and place it here."
else
  echo "Extracting archive.zip"
  unzip archive.zip
  rm -rf "A_Z\ Handwritten\ Data/"
  mv A_Z\ Handwritten\ Data.csv data.csv
fi