#!/bin/bash


###############################################################################
# COLLECT DATA
###############################################################################


# First download the darmstadt data from the following url (https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2448) and put the zip file in this directory.


###############################################################################
# PROCESS DATA
###############################################################################

# Process darmstadt data
unzip DarmstadtServiceReviewCorpus.zip
cd DarmstadtServiceReviewCorpus
unzip universities.zip
grep -rl "&" universities/basedata | xargs sed -i 's/&/and/g'
cd ..
python process_darmstadt.py
rm -rf DarmstadtServiceReviewCorpus
