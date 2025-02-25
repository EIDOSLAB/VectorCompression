# Env
```
conda env create -f env.yml
```
```
conda activate vect_compress
```
# train_vq.py
```
python train_vq.py --save-dir test-vq --vq-alpha 2.5 --codebook-size 512
```
# train_compai.py
```
python train_compai.py --save-dir test-compai --lambda 1e-2 
```


## edit model
check on models/autoencoder.py and edit the function: `get_encoder_decoder`
