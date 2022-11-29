pip3 install virtualenv
virtualenv .
source bin/activate
pip install -r requirements.txt
apt-get install ffmpeg libsm6 libxext6  -y
# export LD_LIBRARY_PATH=/kaggle/working/msg-gan-v1/lib/python3.7/site-packages/nvidia/cublas/lib/:$LD_LIBRARY_PATH