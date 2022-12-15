# Smarthome Sensor Error Detection
## Environment

Use [miniconda](https://docs.conda.io/en/latest/miniconda.html) to set the environment.
```bash
conda create -y --no-default-packages -n nlp-pr python=3.9.15
conda activate nlp-pr
conda env update --name nlp-pr --file environment.yaml --prune
```
You might need to change directory name from ```environment.yaml``` file.

## Dataset
1. Download CASAS data labaled as Aruba: [link](http://casas.wsu.edu/datasets/aruba.zip).
2. Rename the directory as "aruba_smart_home_dataset".
3. Put it in the root of the directory.

For the first time, run this.
```bash
python main.py preprocess
```
After this, change ```configurations.py``` file as you want and run following command.
```bash
python main.py detector
```

Your results will be saved at ```./test_results``` directory.