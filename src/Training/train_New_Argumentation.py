import random
from args import Arguments
from model import MCQAModel
from dataset import CasiMedicosDatasetBalanced
from pytorch_lightning.loggers import CSVLogger
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch, os
import pandas as pd
import json
from tqdm import tqdm
import time, argparse, datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_FOLDER = os.path.join(CURRENT_DIR, "..", "..", "data", "training")
DATASET_FOLDER = os.path.abspath(DATASET_FOLDER)
TEST_FOLDER = os.path.join(CURRENT_DIR, "..", "..", "data", "tests")
TEST_FOLDER = os.path.abspath(TEST_FOLDER)
MODELS_FOLDER = os.path.join(CURRENT_DIR, "..", "..", "models", "new_models")
MODELS_FOLDER = os.path.abspath(MODELS_FOLDER)
RESULTS_FOLDER = os.path.join(CURRENT_DIR, "..", "..", "results")
RESULTS_FOLDER = os.path.abspath(RESULTS_FOLDER)
PRETRAINED_MODEL = "HiTZ/EriBERTa-base"
NUM_RUNS = 3
MODEL_NAME_PREFIX = "Train_New_Argumentation"

def train(gpu, args, experiment_name, models_folder, version, seed):
    pl.seed_everything(seed)

    torch.cuda.init()
    print("Cuda is available? " + str(torch.cuda.is_available()))
    print("Available devices? " + str(torch.cuda.device_count()))

    EXPERIMENT_FOLDER = os.path.join(models_folder, experiment_name)
    os.makedirs(EXPERIMENT_FOLDER, exist_ok=True)
    experiment_string = experiment_name + '-{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}'

    csv_log = CSVLogger(models_folder, name=experiment_name, version=version)

    train_dataset = CasiMedicosDatasetBalanced(args.train_csv, args.use_context)
    val_dataset = CasiMedicosDatasetBalanced(args.dev_csv, args.use_context)

    early_stopping_callback = pl.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.00, patience=args.early_stopping_patience, verbose=True, mode='max')

    cp_callback = pl.callbacks.ModelCheckpoint(monitor='val_acc', dirpath=os.path.join(EXPERIMENT_FOLDER, experiment_string), save_top_k=1, save_weights_only=False, mode='max')

    mcqaModel = MCQAModel(model_name_or_path=args.pretrained_model_name, args=args.__dict__)

    mcqaModel.prepare_dataset(train_dataset=train_dataset, test_dataset=None, val_dataset=val_dataset)

    trainer = Trainer(
        accelerator='gpu',
        strategy="ddp" if not isinstance(gpu, list) else None,
        callbacks=[early_stopping_callback, cp_callback],
        logger=csv_log, 
        max_epochs=args.num_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches
    )

    trainer.fit(mcqaModel)
    print(f"Training completed")

    checkpoints_dir = os.path.join(EXPERIMENT_FOLDER, experiment_string)
    print(f"Checkpoints dir: {checkpoints_dir}")
    ckpt = [f for f in os.listdir(checkpoints_dir) if f.endswith('.ckpt')]
    print(f"Checkpoints list {ckpt}")

    inference_model = MCQAModel.load_from_checkpoint(os.path.join(checkpoints_dir, ckpt[0]))
    inference_model = inference_model.to("cuda")
    inference_model = inference_model.eval()

    return inference_model, trainer

def run_inference_and_save_results(inference_model, trainer, args, test_csv, experiment_name, suffix):
    test_dataset = CasiMedicosDatasetBalanced(test_csv, args.use_context)

    dev_dataset = None  
    inference_model.prepare_dataset(train_dataset=None, val_dataset=dev_dataset, test_dataset=test_dataset)

    test_results = trainer.test(ckpt_path=None, dataloaders=inference_model.test_dataloader())

    experiment_folder = os.path.join(MODELS_FOLDER, experiment_name)

    test_set = []
    with open(test_csv, 'r') as file:
        test_set = [json.loads(row) for row in file]
    predictions = run_inference(inference_model, inference_model.test_dataloader(), args)
    for instance, prediction in zip(test_set, predictions):
        instance['prediction'] = prediction.item() + 1
    with open(os.path.join(experiment_folder, f'test_predictions_{suffix}.jsonl'), 'w') as file:
        for instance in test_set:
            file.write(json.dumps(instance) + '\n')
    print(f"Test predictions written to {os.path.join(experiment_folder, f'test_predictions_{suffix}.jsonl')}")

    return test_results[0]

def run_inference(model, dataloader, args):
    predictions = []
    for idx, (inputs, labels) in tqdm(enumerate(dataloader)):
        for key in inputs.keys():
            inputs[key] = inputs[key].to(args.device)
        with torch.no_grad():
            outputs = model(**inputs)
        prediction_idxs = torch.argmax(outputs, axis=1).cpu().detach().numpy()
        predictions.extend(list(prediction_idxs))
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HiTZ/EriBERTa-base", help="Path to the pretrained model")
    parser.add_argument("--use_context", default=False, action='store_true', help="Mention this flag to use context")
    parser.add_argument("--train", required=True, help="Path to the training file")
    parser.add_argument("--dev", required=True, help="Path to the development (dev) file")
    parser.add_argument("--test", required=True, help="Path to the test file")
    cmd_args = parser.parse_args()

    model = cmd_args.model
    print(f"Training started for model - {model} variant - {DATASET_FOLDER} use_context - {str(cmd_args.use_context)}")

    args = Arguments(train_csv=os.path.join(DATASET_FOLDER, cmd_args.train),
                     dev_csv=os.path.join(DATASET_FOLDER, cmd_args.dev),
                     test_csv=os.path.join(DATASET_FOLDER, cmd_args.test),
                     pretrained_model_name=PRETRAINED_MODEL,
                     use_context=cmd_args.use_context
                     )

    seeds = [random.randint(0, 100000) for _ in range(NUM_RUNS)]
    all_test_results = {
        'No_Argument': [],
        'Expert': [],
        'Noise': [],
        'Label_Only': [],
        'IR_Passages': [],
        'GPT4': [],
        'OpenBioLLM': [],
        'Llama3': []
    }

    for i, seed in enumerate(seeds):
        current_datetime = datetime.datetime.now()
        formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M")
        exp_name = f"{MODEL_NAME_PREFIX}_run_{i+1}___data{os.path.basename(args.train_csv)}___seqlen{str(args.max_len)}___execTime{str(formatted_datetime)}".replace("/", "_")

        inference_model, trainer = train(gpu=args.gpu, args=args, experiment_name=exp_name, models_folder=MODELS_FOLDER, version=exp_name, seed=seed)

        for group in ['No_Argument', 'Expert', 'Noise', 'Label_Only', 'IR_Passages', 'GPT4', 'OpenBioLLM', 'Llama3']:
            test_csv = os.path.join(TEST_FOLDER, f'en.test_casimedicos_{group}.jsonl')
            test_results = run_inference_and_save_results(inference_model, trainer, args, test_csv, exp_name, group)
            all_test_results[group].append(test_results)

    test_avg_results = {
        group: {
            "mean": pd.DataFrame(results).mean().to_dict(),
            "std_dev": pd.DataFrame(results).std().to_dict(),
            "individual_results": results
        } 
        for group, results in all_test_results.items()
    }

    results = {
        "test_results": test_avg_results,
        "seeds_used": seeds
    }

    print(f"Test Average Results: {test_avg_results}")
    print(f"Seeds used: {seeds}")

    output_file = f"{MODEL_NAME_PREFIX}_results.json"
    output_path = os.path.join(RESULTS_FOLDER, output_file)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")

