# LocalBackend


Activate virtual environment:
```angular2html
source .venv/bin/activate
```
Run tests:
```angular2html
python -m unittest discover
```

Run app:
```angular2html
python app.py
```



Setup for Raspberry Pi 5

Install python 3.9:

```angular2html
sudo apt update
sudo apt install -y build-essential zlib1g-dev libncurses5-dev libgdbm-dev \
  libnss3-dev libssl-dev libreadline-dev libffi-dev libsqlite3-dev wget libbz2-dev

cd ~
wget https://www.python.org/ftp/python/3.9.16/Python-3.9.16.tgz
tar -xf Python-3.9.16.tgz
cd Python-3.9.16
./configure --enable-optimizations
make -j4  # or -jN, where N = number of CPU cores
sudo make altinstall
```

Install edgetpu library:

```angular2html
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update

sudo apt-get install libedgetpu1-std

```





Install pycoral:

```angular2html
python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install --extra-index-url https://google-coral.github.io/py-repo/ pycoral~=2.0
python -c "from pycoral.utils.edgetpu import list_edge_tpus; print(list_edge_tpus())"
```




Install picamera2:
  <!-- https://forums.raspberrypi.com/viewtopic.php?t=361758 -->
  
```angular2html
sudo apt install libcap-dev -y
pip install git+https://github.com/raspberrypi/picamera2.git



    sudo apt update && sudo apt upgrade
    sudo apt install libcap-dev libatlas-base-dev ffmpeg libopenjp2-7
    sudo apt install libcamera-dev
    sudo apt install libkms++-dev libfmt-dev libdrm-dev

source .venv/bin/activate

pip install --upgrade pip
pip install wheel
pip install rpi-libcamera rpi-kms picamera2
```


[//]: # (from picamera2 import Picamera2)

[//]: # ()
[//]: # (picam2 = Picamera2&#40;&#41;)

[//]: # ()
[//]: # (picam2.start_and_capture_file&#40;"test.jpg", show_preview=False&#41;)
