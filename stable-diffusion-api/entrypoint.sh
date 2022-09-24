#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh
conda activate ldm
gunicorn --bind 0.0.0.0:5000 --pythonpath api -c api/gunicorn.conf.py wsgi:app