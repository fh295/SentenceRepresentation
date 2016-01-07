
#export LD_LIBRARY_PATH=/home/fh295/torch/install/lib:/usr/lib:/usr/local/cuda/lib64


THEANO_FLAGS='cuda.root=/usr/local/cuda,device=gpu,floatX=float32'  python train_book.py



