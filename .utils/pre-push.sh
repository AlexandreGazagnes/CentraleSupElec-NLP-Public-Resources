#! /bin/bash


# black 
black .
black */*.ipynb
black */*/*.ipynb
black */*/*/*.ipynb

# clean notebooks output
jupyter nbconvert --clear-output --inplace */*.ipynb
jupyter nbconvert --clear-output --inplace */*/*.ipynb
jupyter nbconvert --clear-output --inplace */*/*/*.ipynb

# build .html files 
jupyter nbconvert --to html */*.ipynb
jupyter nbconvert --to html */*/*.ipynb
jupyter nbconvert --to html */*/*/*.ipynb

# build .pdf files 
jupyter nbconvert --to pdf */*.ipynb
jupyter nbconvert --to pdf */*/*.ipynb
jupyter nbconvert --to pdf */*/*/*.ipynb
