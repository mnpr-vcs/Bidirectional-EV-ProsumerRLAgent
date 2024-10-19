#!/usr/bin/env bash
# ---------------------------------------------------
#  Script to reproduce results
# ---------------------------------------------------
# steps:
# 1. activate the venv
# 2. run zip to onnx conversion
# 3. run server
# ---------------------------------------------------

# change to server directory
cd ../src/server/
# 1. activate the venv
conda activate expy
# 2. run python main with args
printf ">>---------------------- ONNX Conversion -------------------\n\n"
python zip_to_onnx.py
printf ">>---------------------- Running Server -------------------\n\n"
# 3. run server
# python server.py
docker-compose up -d
printf ">>--- Server Started: http://127.0.0.1:5000 ---\n\n"
printf ">> run cd ../src/server/ && docker-compose down to stop server\n"
printf ">>---------------------- End of Script -------------------\n\n"
