# Direct-Inversion with P2P
## environment
```
conda env create --file myconda.yaml
```
## how to modify image location
In the main.py, you can change the value
```
original_prompt = "photo of bird"
editing_prompt = "photo of frog"
image_path = "./img/bird.jpg"
editing_instruction = "Make the bird to frog"
blended_word = ["bird", "frog"]
```
## how to run?
```
bash run.sh
```
# Acknowledgement
This code is modified on the basis of [DirectInversion](https://github.com/cure-lab/PnPInversion)