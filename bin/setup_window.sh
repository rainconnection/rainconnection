#!/bin/bash

set -e
root=$(cd $(dirname $(readlink $0 || echo $0))/..;/bin/pwd)

cd ${root}

if [[ ! -d ${root}/venv ]]; then
    echo "python virtual env creating"

    python3 -m venv ${root}/venv

    source ${root}/venv/Scripts/activatex

    pip install --upgrade pip \
    pip install -r ${root}/requirements.txt

fi