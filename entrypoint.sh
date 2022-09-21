#!/bin/bash

. /opt/conda/etc/profile.d/conda.sh
conda activate ldm
gunicorn --bind 0.0.0.0:5000 --pythonpath scripts -c scripts/gunicorn.conf.py wsgi:app