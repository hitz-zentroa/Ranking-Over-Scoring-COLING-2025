import pytorch_lightning as pl
from torch.utils.data import SequentialSampler, RandomSampler
from torch import nn
import numpy as np
import math
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import RandomSampler
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModel
import functools
import wandb

'''
Adaptado de: https://github.com/paulaonta/medmcqa/blob/main/model_5ans.py

Adaptaciones:

Explanation of Changes:x

Removed TrainResult and EvalResult: These classes are no longer used in PyTorch Lightning 1.9.0 or later.
Directly Logging Metrics: Instead of using deprecated methods like result.log, you can now use self.log within your training_step and test_step methods.
Logging on Epoch: The on_epoch=True argument ensures that the metrics are logged only at the end of each epoch.

Additional Notes:

If you need to log additional data like logits or labels during inference (test step), you can add them to self.log as well.
This rewritten code maintains the core functionality of the original code while adhering to the updated PyTorch Lightning API.
'''


class MCQAModel(pl.LightningModule):
    def __init__(self,
                 model_name_or_path,
                 args):

        super().__init__()
        self.init_encoder_model(model_name_or_path)
        self.args = args
        self.batch_size = self.args['batch_size']
        self.dropout = nn.Dropout(self.args['hidden_dropout_prob'])
        self.linear = nn.Linear(in_features=self.args['hidden_size'], out_features=1)
        self.ce_loss = nn.CrossEntropyLoss()
        self.save_hyperparameters()

    def init_encoder_model(self, model_name_or_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

    def prepare_dataset(self, train_dataset, val_dataset, test_dataset=None):
        """
        helper to set the train and val dataset. Doing it during class initialization
        causes issues while loading checkpoint as the dataset class needs to be
        present for the weights to be loaded.
        """
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        if test_dataset != None:
            self.test_dataset = test_dataset
        else:
            self.test_dataset = val_dataset

    def forward(self, input_ids, attention_mask):  # token_type_ids deleted from parameters because eriberta's tokenizer does not provide them
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.linear(pooled_output)
        reshaped_logits = logits.view(-1, self.args['num_choices'])  # Convierte a la forma (batch_size,num_choices x combinations_of_correct_option); es decir, un tensor de 2 dimensiones de batch_size filas y num_choices columnas
        return reshaped_logits

    def training_step(self, batch, batch_idx):
        inputs, labels = batch
        for key in inputs:
            inputs[key] = inputs[key].to(self.args["device"])
        logits = self(**inputs)
        loss = self.ce_loss(logits, labels)
        # Log metrics directly using self.log
        self.log('train_loss', loss, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        for key in inputs:
            inputs[key] = inputs[key].to(self.args["device"])
        logits = self(**inputs)
        loss = self.ce_loss(logits, labels)

        self.log('test_loss', loss, on_epoch=True)
        return {'val_loss': loss, 'logits': logits, 'labels': labels}  # Previously: return result (EvalResult). Now there is no need to return a result object

    def test_epoch_end(self, outputs):
        # 'outputs' is a list of whatever you returned in `test_step`. Each element of the list corresponds to one batch of samples from each step in an epoch.
        # Calcular el promedio del loss
        avg_loss = sum([x['val_loss'].cpu() for x in outputs]) / len(outputs)

        # Calcular la predicción haciendo argmax de los logits
        predictions = [torch.argmax(x['logits'], axis=-1) for x in outputs]

        # Calcular cuantos aciertos han habido (accuracy)
        correct_predictions = 0
        for index, x in enumerate(outputs):
            correct_predictions += torch.sum(predictions[index] == x['labels'])
        each_prediction_size = predictions[0].size()[0]
        predictions_length = len(predictions)
        total_number_of_predictions = each_prediction_size * predictions_length
        accuracy = correct_predictions.cpu().detach().numpy() / total_number_of_predictions

        # Get the confusion matrix
        labels = [x['labels'].cpu().detach().item() for x in outputs]
        predictions = [x.cpu().detach().item() for x in predictions]
        confusion_matrix = wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=predictions, class_names=['A', 'B', 'C', 'D', 'E'])

        self.log_dict({"test_loss": avg_loss, "test_acc": accuracy}, prog_bar=True, on_epoch=True)
        self.log('avg_test_loss', avg_loss)
        self.log('avg_test_acc', accuracy)
        wandbLogger = self.loggers[0]
        wandbLogger.log_metrics({"conf_matrix_test": confusion_matrix})
        return avg_loss  # Previously: return result (EvalResult). Now there is no need to return a result object

    def validation_step(self, batch, batch_idx):
        inputs, labels = batch
        # Move to self.args['device']
        for key in inputs:
            inputs[key] = inputs[key].to(self.args['device'])
        logits = self(**inputs)  # calls forward
        loss = self.ce_loss(logits, labels)
        self.log('val_loss', loss, on_epoch=True)
        return {'val_loss': loss, 'logits': logits, 'labels': labels}  # Previously: return result (EvalResult). Now there is no need to return a result object.

    def validation_epoch_end(self, outputs):
        # 'outputs' is a list of whatever you returned in `validation_step`. Each element of the list corresponds to one batch of samples from each step in an epoch.

        # Calcular el promedio del loss
        avg_loss = sum([x['val_loss'].cpu() for x in outputs]) / len(outputs)

        # Calcular la predicción haciendo argmax de los logits
        predictions = [torch.argmax(x['logits'], axis=-1) for x in outputs]

        # Calcular cuantos aciertos han habido (accuracy)
        correct_predictions = 0
        for index, x in enumerate(outputs):
            correct_predictions += torch.sum(predictions[index] == x['labels'])
        each_prediction_size = predictions[0].size()[0]
        predictions_length = len(predictions)
        total_number_of_predictions = each_prediction_size * predictions_length
        accuracy = correct_predictions.cpu().detach().numpy() / total_number_of_predictions

        # Get the confusion matrix
        labels = [x['labels'].cpu().detach().item() for x in outputs]
        predictions = [x.cpu().detach().item() for x in predictions]
        confusion_matrix = wandb.plot.confusion_matrix(probs=None, y_true=labels, preds=predictions, class_names=['A', 'B', 'C', 'D', 'E'])

        # Logging
        self.log_dict({"val_loss": avg_loss, "val_acc": accuracy}, prog_bar=True, on_epoch=True)
        self.log('avg_val_loss', avg_loss)
        self.log('avg_val_acc', accuracy)
        wandbLogger = self.loggers[0]
        wandbLogger.log_metrics({"conf_matrix": confusion_matrix})
        return avg_loss  # Previously: return result (EvalResult). Now there is no need to return a result object

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.args['learning_rate'], eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=100,
            num_training_steps=(self.args['num_epochs'] + 1) * math.ceil(len(self.train_dataset) / self.args['batch_size']),
        )
        return [optimizer], [scheduler]

    def process_batch(self, batch, tokenizer, max_len=32):
        '''
        For instance in the batch, generates all possible combinations of question and options.

        Instance: (question, [options], label)
        - Combination 1: <s>question</s>opa</s>
        - Combination 2: <s>question</s>opb</s>
        - Combination 3: <s>question</s>opc</s>
        - Combination 4: <s>question</s>opd</s>
        - Combination 5: <s>question</s>ope</s>
        '''
        expanded_batch = []
        labels = []
        context = None
        for data_tuple in batch:
            if len(data_tuple) == 4:
                context, question, options, label = data_tuple
            else:
                question, options, label = data_tuple
            question_option_pairs = [
                question +
                tokenizer.sep_token +
                option if type(option) != float and type(option) != np.float64 else question + "" for option in options
            ]

            labels.append(label)

            if context:
                contexts = [context] * len(options)
                expanded_batch.extend(zip(contexts, question_option_pairs))
            else:
                expanded_batch.extend(question_option_pairs)
        tokenized_batch = tokenizer(expanded_batch, truncation=True, padding="max_length", max_length=max_len, return_tensors="pt")

        return tokenized_batch, torch.tensor(labels)

    ## ↓ DataLoaders ↓ ##

    def train_dataloader(self):
        train_sampler = RandomSampler(self.train_dataset)
        model_collate_fn = functools.partial(
            self.process_batch,
            tokenizer=self.tokenizer,
            max_len=self.args['max_len']
        )
        train_dataloader = DataLoader(self.train_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=8,
                                      sampler=train_sampler,
                                      collate_fn=model_collate_fn)
        return train_dataloader

    def val_dataloader(self):
        eval_sampler = SequentialSampler(self.val_dataset)
        model_collate_fn = functools.partial(
            self.process_batch,
            tokenizer=self.tokenizer,
            max_len=self.args['max_len']
        )
        val_dataloader = DataLoader(self.val_dataset,
                                    batch_size=self.batch_size,
                                    num_workers=8,
                                    sampler=eval_sampler,
                                    collate_fn=model_collate_fn)
        return val_dataloader

    def test_dataloader(self):
        eval_sampler = SequentialSampler(self.test_dataset)
        model_collate_fn = functools.partial(
            self.process_batch,
            tokenizer=self.tokenizer,
            max_len=self.args['max_len']
        )
        test_dataloader = DataLoader(self.test_dataset,
                                     batch_size=self.batch_size,
                                     num_workers=8,
                                     sampler=eval_sampler,
                                     collate_fn=model_collate_fn)
        return test_dataloader
