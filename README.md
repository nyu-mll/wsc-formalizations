This repository is the code release for our paper "Precise Task Formalization Matters in Winograd Schema Evaluations" at EMNLP 2020.

# Setup

0. Clone the reprository
   `git clone https://github.com/HaokunLiu/wsc-trick.git`
1. Create an environment from environment.yml
   `conda env create -f environment.yml`
2. Activate the environment
   `conda activate wsc`
3. Download some spacy stuff
   `python -m spacy download en_core_web_lg`
4. WSC data can be downloaded [here](https://super.gluebenchmark.com/tasks), Winogrande data can be downloaded [here](https://mosaic.allenai.org/projects/winogrande)
5. (optional) Follow [instructions](https://github.com/NVIDIA/apex) to install apex.

# Structure

`src` includes the source code, `exp_control` includes the script for running experiments, `analysis` includes the script for analyzing results

# Reproducing results

To finetuning a single model use `main.py` in `src`, to run hyperparameter searching of one formalization use `hyper_parameter_tuning.py` in `exp_control`

# Recommanded citation

Stay tuned for update.
