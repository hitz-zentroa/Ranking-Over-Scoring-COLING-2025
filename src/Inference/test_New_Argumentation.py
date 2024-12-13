import random
from args import Arguments
from model import MCQAModel
from dataset import CasiMedicosDatasetBalanced
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch, os
import pandas as pd
import json
from tqdm import tqdm
import time, argparse, datetime

os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ["WANDB_MODE"] = "disabled"
WB_PROJECT = "Results New Argumentation"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(CURRENT_DIR, "..", "..", "data", "tests")
DATASET_FOLDER = os.path.abspath(DATASET_FOLDER)
MODELS_FOLDER = os.path.join(CURRENT_DIR, "..", "..", "models", "checkpoints")
MODELS_FOLDER = os.path.abspath(MODELS_FOLDER)
BASE_MODEL_PATH = "HiTZ/EriBERTa-base"


def evaluate_model(checkpoint_path, test, val, args):
    
    model = MCQAModel.load_from_checkpoint(checkpoint_path, model_name_or_path=BASE_MODEL_PATH)
    model = model.to("cuda")
    model = model.eval()
    
    model.prepare_dataset(train_dataset=None, val_dataset=None, test_dataset=test)
    
    wb = WandbLogger(project=WB_PROJECT, name="Results New", version="1")
    csv_log = CSVLogger(MODELS_FOLDER, name="Results New", version="1")

    trainer = Trainer(accelerator='gpu', strategy="ddp" if not isinstance(args.gpu, list) else None, logger=[wb, csv_log])
    test_results = trainer.test(model, dataloaders=model.test_dataloader())

    return test_results

def evaluate_models(args_New, model_paths):

    test_datasets = {
        "New_Argument": CasiMedicosDatasetBalanced(args_New.test_csv, args_New.use_context)

    }

    results_by_test = {name: [] for name in test_datasets.keys()}

    for checkpoint_path in model_paths:
        print(f"Evaluating model: {checkpoint_path}")


        for test_name, test_dataset in test_datasets.items():
            test_results = evaluate_model(checkpoint_path, test_dataset, None, args_New)
            results_by_test[test_name].append(test_results)


    for test_name, result_list in results_by_test.items():

        if len(result_list) > 0 and len(result_list[0]) > 0:

            keys = result_list[0][0].keys()
            avg_test_results = {k: sum(r[0][k] for r in result_list) / len(result_list) for k in keys}
            
            print(f"Average accuracy for {test_name}: {avg_test_results}")
        else:
            print(f"No results for {test_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Results New Argumentation", help="name of the model")
    parser.add_argument("--use_context", default=False, action='store_true', help="mention this flag to use_context")
    parser.add_argument("--test", required=True, help="Path to the test file")
    cmd_args = parser.parse_args()
    
    model = cmd_args.model

    args_New = Arguments(train_csv=os.path.join(DATASET_FOLDER, ""),
                     dev_csv=os.path.join(DATASET_FOLDER, ""),
                     test_csv=os.path.join(DATASET_FOLDER, cmd_args.test),
                     use_context=cmd_args.use_context
                     )
    model_paths = [
        f'{MODELS_FOLDER}/LLM_Mixture_1-run_1.ckpt',
        f'{MODELS_FOLDER}/LLM_Mixture_1-run_2.ckpt',
        f'{MODELS_FOLDER}/LLM_Mixture_1-run_3.ckpt',
        f'{MODELS_FOLDER}/LLM_Mixture_2-run_1.ckpt',
        f'{MODELS_FOLDER}/LLM_Mixture_2-run_2.ckpt',
        f'{MODELS_FOLDER}/LLM_Mixture_2-run_3.ckpt',
        f'{MODELS_FOLDER}/LLM_Mixture_3-run_1.ckpt',
        f'{MODELS_FOLDER}/LLM_Mixture_3-run_2.ckpt',
        f'{MODELS_FOLDER}/LLM_Mixture_3-run_3.ckpt'

    ]
    
    evaluate_models(args_New, model_paths)
