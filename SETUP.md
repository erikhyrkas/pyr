# Setup

### PyTorch with cuda support

You need a compiler in your path for torch compile to work. On Windows, that's cl.exe, which means visual studio.

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

### Bitnet's transformers (https://github.com/shumingma/transformers)

pip3 install git+https://github.com/shumingma/transformers.git

### Additional installs

pip3 install -r requirements.txt
