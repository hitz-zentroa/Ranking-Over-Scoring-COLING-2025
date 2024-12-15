# Ranking Over Scoring: Reliable and Robust Automated Evaluation of LLM-Generated Medical Explanatory Arguments

<p align="center">
<a href="https://coling2025.org/">
      <img alt="Web Page" src="https://img.shields.io/badge/COLING-Visit%20Here-red">
</a>
<a href="https://github.com/hitz-zentroa/cn-eval/blob/main/LICENSE">
        <img alt="GitHub license" src="https://img.shields.io/github/license/hitz-zentroa/cn-eval">
</a>
</p>



Welcome to the official repository for **"Ranking Over Scoring: Towards Reliable and Robust Automated Evaluation of LLM-Generated Medical Explanatory Arguments"**, presented at COLING 2025. This repository aims to provide the scientific community with access to our models, proxy tasks, and tools developed during our research.


## Repository Structure

- **/data**
  - /Casimedicos_With_LLM_Argumentation_Mixture: This folder includes three training and development files, each containing a different mix of medical arguments generated by the LLMs used in our study: GPT-4, OpenBioLLM, and Llama 3.
  - /tests: This directory contains the JSONL files we used to test and evaluate our different evaluators.
  - /training: This directory includes example files for training and development, in case you want to train a new model.

- **/results**: This directory is designated for creating and storing results in JSONL format after training a new evaluator and performing inference with that evaluator.

- **/src**
  - /Inference: This folder contains all the necessary scripts to replicate the results of our paper in the MMCQA proxy task (referenced in Table 10) using the evaluator trained with medical arguments generated by Large Language Models (LLMs). Additionally, it includes scripts for evaluating new medical arguments with the models developed in our research.
  - /Training: This folder contains all the necessary scripts to train and evaluate a new model.
  - /Utils: This folder includes the script we used to balance the MMCQA datasets.

- `requirements.txt`: Lists the Python packages needed to run the project.

## Data format
  
## Instructions for Use
To utilize the software provided in this repository, it is recommended to create a new virtual environment and install the necessary Python packages:
```bash
pip install -r requirements.txt
```
We also recommend that if you use a dataset not included in this repository, you first balance it using the `generate_syn_dataset.py` script located in the `src/Utils` folder. For this task set first the `SETS_PATH` variable to the path where your original dataset files are located and set `new_file_path` variable to the path where you want to save your balanced files.


### Replicating our results
In order to replicate the results presented in Table 10 of our paper, where we utilize the evaluator trained with different medical arguments generated by three Large Language Models (LLMs), download the pre-trained models from this link and organize your project directory by placing the downloaded `models` directory alongside the existing `data`, `results`, and `src` folders.

Next, navigate to the `src/Inference` directory and execute the following command:

```bash
python3 test_Results_Replication.py
```

### Testing new medical arguments on MMCQA proxy task

If you wish to assess the quality of new medical arguments, place your JSONL file containing the new medical arguments into the `tests` directory and execute the following command from the `src/Inference` directory:

```bash
python3 test_New_Argumentation.py --test PathToTheTestFile.jsonl
```
Please ensure that you replace `PathToTheTestFile.jsonl` with the actual path to the test file you intend to evaluate. 

### Training a new evaluator





## Citation

If you use any of the resources provided in this repository for research purposes, please cite the following paper:

```bibtex
@misc{delaiglesia2024rankingscoringreliablerobust,
      title={Ranking Over Scoring: Towards Reliable and Robust Automated Evaluation of LLM-Generated Medical Explanatory Arguments}, 
      author={Iker De la Iglesia and Iakes Goenaga and Johanna Ramirez-Romero and Jose Maria Villa-Gonzalez and Josu Goikoetxea and Ander Barrena},
      year={2024},
      eprint={2409.20565},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.20565}, 
}

```