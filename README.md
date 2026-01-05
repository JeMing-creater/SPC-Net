## requirements
WSI requires the CLAM library for slicing. The following describes how to install it.
```
# VFM
export HF_ENDPOINT="https://hf-mirror.com"
mkdir -p src/model
python ./requirements/download_vfm.py


# CLAM
git clone https://github.com/mahmoodlab/CLAM.git
cd CLAM

conda env create -f env.yml
```
After installing the CLAM environment, install other packages in this environment.
```
conda activate clam_latest
pip install -r ./requirements/requirements.txt

```

And use Huggingface, user need to make self-token like:
```
hf_xxxxxxxxxxxxxxxxxxx
```
and set it into ```config.sr.token```


