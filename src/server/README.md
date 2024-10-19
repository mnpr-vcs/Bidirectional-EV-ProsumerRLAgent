**setup**

- `pip install onnxruntime-gpu --extra-index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/onnxruntime-cuda-12/pypi/simple/`
- `pip install flask`

**convert checkpoints to onnx format**

- `python zip_to_onnx.py`


**start/stop server**

- `python server.py`

*with docker*
- `docker-compose up -d` and `docker-compose down`
