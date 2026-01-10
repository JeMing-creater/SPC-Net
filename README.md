## requirements
WSI requires the CLAM library for slicing. The following describes how to install it.
```
# CLAM
git clone https://github.com/mahmoodlab/CLAM.git
cd CLAM

conda env create -f env.yml

# VFM
export HF_ENDPOINT="https://hf-mirror.com"
mkdir -p src/models
python ./requirements/download_vfm.py

```
After installing the CLAM environment, install other packages in this environment.
```
conda activate clam_latest
pip install -r ./requirements/requirements.txt

```

SAM
user need to download weight vit_b ```sam vit b 01ec64.pth``` from this link: 
```
https://github.com/facebookresearch/segment-anything?tab=readme-ov-file
```
and then set it into `src/models/`


And use Huggingface, user need to make self-token like:
```
hf_xxxxxxxxxxxxxxxxxxx
```
and set it into ```config.sr.token```


