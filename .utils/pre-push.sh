#! /bin/bash


# # black 
# black .
# black sessions/*.ipynb
# black sessions/*/*.ipynb
black sessions/*/*/*.ipynb
black sessions/*/*/*.py


# clean notebooks output
# jupyter nbconvert --clear-output --inplace sessions/*.ipynb
# jupyter nbconvert --clear-output --inplace sessions/*/*.ipynb
jupyter nbconvert --clear-output --inplace sessions/*/*/*.ipynb

# # build .html files 
# jupyter nbconvert --to html sessions /*.ipynb
# jupyter nbconvert --to html sessions /*/*.ipynb
jupyter nbconvert --to html sessions/*/*/*.ipynb

# # build .pdf files 
# jupyter nbconvert --to pdf sessions/*.ipynb
# jupyter nbconvert --to pdf sessions/*/*.ipynb
# jupyter nbconvert --to pdf sessions/*/*/*.ipynb
