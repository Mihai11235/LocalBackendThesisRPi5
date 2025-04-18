python model/semantic_segmentation.py --model model/model_quant_edgetpu.tflite --input model/image.jpg



python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
python -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"





sudo apt install libcap-dev -y
pip install git+https://github.com/raspberrypi/picamera2.git



    sudo apt update && sudo apt upgrade
    sudo apt install libcap-dev libatlas-base-dev ffmpeg libopenjp2-7
    sudo apt install libcamera-dev
    sudo apt install libkms++-dev libfmt-dev libdrm-dev

source .venv/bin/activate

pip install --upgrade pip
pip install wheel
pip install rpi-libcamera rpi-kms picamera2pip install --upgrade pip
pip install wheel
pip install rpi-libcamera rpi-kms picamera2