#!/bin/sh

# install requirements
pip install -r requirements.txt


mkdir "plates"
i=0
for url in $(cat car-plates-urls.txt); do
  wget -P plates $url
done

# [optional]
# create new car plate recognition project
# create topology.txt file, see this link: https://github.com/efthymis-mcl/NN-Tool to edit file.


#nntool create_project cpr
#cd cpr
#nano topology.txt
#nntool train --topology topology.txt --dataset mnist-chars74k --epochs 60 --batchsize 32

