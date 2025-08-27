# Deep Learning Starter
A collection of various deep learning model architectures using TensorFlow, Keras, PyTorch and Lightning AI

## Install Python 3.12
https://www.python.org/downloads/

Linux
```
sudo apt install python3 python3-pip python3-venv -y
```

## Environment setup
1. Install required packages in a Python virtual environment.

Windows
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linux
```
python3 -m venv .venv
source .venv/bin/activate
pip3 install -r requirements.txt
```

2. Install Jupyter Notebook extention in Visual Studio Code.

3. Set the VS Code Runtime as the virtual environment and Run the scripts.

## GPU Support Check
```
python scripts\gpu_support_check.py
```