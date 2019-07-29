#!/usr/bin/env bash

python process_data.py -dataset rsc

python process_data.py -dataset tb

python process_data.py -dataset yelp
    $@

