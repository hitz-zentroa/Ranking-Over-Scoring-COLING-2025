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
WB_PROJECT = "Results Replication"
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
    
    wb = WandbLogger(project=WB_PROJECT, name="Results Replication", version="1")
    csv_log = CSVLogger(MODELS_FOLDER, name="Results Replication", version="1")

    trainer = Trainer(accelerator='gpu', strategy="ddp" if not isinstance(args.gpu, list) else None, logger=[wb, csv_log])
    test_results = trainer.test(model, dataloaders=model.test_dataloader())
    #val_results = trainer.validate(model, dataloaders=model.val_dataloader())

    return test_results

def evaluate_models(args_Baseline, args_Expert, args_Noise, args_Label_Only, args_IR_Passages, args_GPT4, args_OpenBioLLM, args_Llama3, model_paths):

    test_datasets = {
        "No_Argument": CasiMedicosDatasetBalanced(args_Baseline.test_csv, args_Baseline.use_context),
        "Expert": CasiMedicosDatasetBalanced(args_Expert.test_csv, args_Expert.use_context),
        "Noise": CasiMedicosDatasetBalanced(args_Noise.test_csv, args_Noise.use_context),
        "Label_Only": CasiMedicosDatasetBalanced(args_Label_Only.test_csv, args_Label_Only.use_context),
        "IR_Passages": CasiMedicosDatasetBalanced(args_IR_Passages.test_csv, args_IR_Passages.use_context),
        "GPT4": CasiMedicosDatasetBalanced(args_GPT4.test_csv, args_GPT4.use_context),
        "OpenBioLLM": CasiMedicosDatasetBalanced(args_OpenBioLLM.test_csv, args_OpenBioLLM.use_context),
        "Llama3": CasiMedicosDatasetBalanced(args_Llama3.test_csv, args_Llama3.use_context)
    }

    results_by_test = {name: [] for name in test_datasets.keys()}

    for checkpoint_path in model_paths:
        print(f"Evaluating model: {checkpoint_path}")


        for test_name, test_dataset in test_datasets.items():
            test_results = evaluate_model(checkpoint_path, test_dataset, None, args_Baseline)
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
    parser.add_argument("--model", default="Results_Replication", help="name of the model")
    parser.add_argument("--use_context", default=False, action='store_true', help="mention this flag to use_context")
    cmd_args = parser.parse_args()
    
    model = cmd_args.model

    args_No_Argument = Arguments(train_csv=os.path.join(DATASET_FOLDER, ""),
                     dev_csv=os.path.join(DATASET_FOLDER, ""),
                     test_csv=os.path.join(DATASET_FOLDER, "en.test_casimedicos_No_Argument.jsonl"),
                     use_context=cmd_args.use_context
                     )
    args_Expert = Arguments(train_csv=os.path.join(DATASET_FOLDER, ""),
                     dev_csv=os.path.join(DATASET_FOLDER, ""),
                     test_csv=os.path.join(DATASET_FOLDER, "en.test_casimedicos_Expert.jsonl"),
                     use_context=cmd_args.use_context
                     )
    args_Noise = Arguments(train_csv=os.path.join(DATASET_FOLDER, ""),
                     dev_csv=os.path.join(DATASET_FOLDER, ""),
                     test_csv=os.path.join(DATASET_FOLDER, "en.test_casimedicos_Noise.jsonl"),
                     use_context=cmd_args.use_context
                     )
    args_Label_Only = Arguments(train_csv=os.path.join(DATASET_FOLDER, ""),
                     dev_csv=os.path.join(DATASET_FOLDER, ""),
                     test_csv=os.path.join(DATASET_FOLDER, "en.test_casimedicos_Label_Only.jsonl"),
                     use_context=cmd_args.use_context
                     )
    args_IR_Passages = Arguments(train_csv=os.path.join(DATASET_FOLDER, ""),
                     dev_csv=os.path.join(DATASET_FOLDER, ""),
                     test_csv=os.path.join(DATASET_FOLDER, "en.test_casimedicos_IR_Passages.jsonl"),
                     use_context=cmd_args.use_context
                     )
    args_GPT4 = Arguments(train_csv=os.path.join(DATASET_FOLDER, ""),
                     dev_csv=os.path.join(DATASET_FOLDER, ""),
                     test_csv=os.path.join(DATASET_FOLDER, "en.test_casimedicos_GPT4.jsonl"),
                     use_context=cmd_args.use_context
                     )
    args_OpenBioLLM = Arguments(train_csv=os.path.join(DATASET_FOLDER, ""),
                     dev_csv=os.path.join(DATASET_FOLDER, ""),
                     test_csv=os.path.join(DATASET_FOLDER, "en.test_casimedicos_OpenBioLLM.jsonl"),
                     use_context=cmd_args.use_context
                     )
    args_Llama3 = Arguments(train_csv=os.path.join(DATASET_FOLDER, ""),
                     dev_csv=os.path.join(DATASET_FOLDER, ""),
                     test_csv=os.path.join(DATASET_FOLDER, "en.test_casimedicos_Llama3.jsonl"),
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
    
    evaluate_models(args_No_Argument, args_Expert, args_Noise, args_Label_Only, args_IR_Passages, args_GPT4, args_OpenBioLLM, args_Llama3, model_paths)
