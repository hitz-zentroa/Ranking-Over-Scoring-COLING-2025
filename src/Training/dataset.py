from torch.utils.data import Dataset
import pandas as pd
import json


class MCQADataset5(Dataset):
    '''
    Author: MedMCQA team
    Description: Load CSV as Dataset for data with 5 possible answers
    '''

    def __init__(self,
                 csv_path,
                 use_context=True):
        #     self.dataset = dataset['train'] if training == True else dataset['test']
        self.dataset = pd.read_csv(csv_path)
        self.use_context = use_context

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return_tuple = tuple()
        if self.use_context:
            context = self.dataset.loc[idx, 'exp']
            return_tuple += (context,)
        question = self.dataset.loc[idx, 'question']
        options = self.dataset.loc[idx, ['opa', 'opb', 'opc', 'opd', 'ope']].values
        label = self.dataset.loc[idx, 'cop']
        return_tuple += (question, options, label)
        return return_tuple


class MCQADataset4(Dataset):
    '''
    Author: MedMCQA team
    Description: Load CSV as Dataset for data with 4 possible answers
    '''

    def __init__(self,
                 csv_path,
                 use_context=True):
        #     self.dataset = dataset['train'] if training == True else dataset['test']
        self.dataset = pd.read_csv(csv_path)
        self.use_context = use_context

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return_tuple = tuple()
        if self.use_context:
            context = self.dataset.loc[idx, 'exp']
            return_tuple += (context,)
        question = self.dataset.loc[idx, 'question']
        options = self.dataset.loc[idx, ['opa', 'opb', 'opc', 'opd']].values
        label = self.dataset.loc[idx, 'cop']
        return_tuple += (question, options, label)
        return return_tuple


class CasiMedicosDataset(Dataset):
    '''
    Author: Aingeru
    '''

    def __init__(self,
                 jsonl_path,
                 use_context=False):
        with open(jsonl_path, 'r') as file:
            self.dataset = [json.loads(row) for row in file]
        self.use_context = use_context  # Aunque no lo vamos a utilizar

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get instance of the dataset at the given index
        instance = self.dataset[index]

        return_tuple = tuple()
        question = self.dataset[index]['full_question']
        options = (
            instance['options']["1"],
            instance['options']["2"],
            instance['options']["3"],
            instance['options']["4"],
            instance['options']["5"] if isinstance(instance['options']["5"], str) else ""
        )
        label = instance['correct_option'] - 1 # -1 because it has to be [0, 1, 2, 3, 4] and not [1, 2, 3, 4, 5]
        return_tuple += (question, options, label)
        return return_tuple


class CasiMedicosDatasetBalanced(Dataset):
    '''
    Author: Aingeru
    '''

    def __init__(self,
                 jsonl_path,
                 use_context=False):
        with open(jsonl_path, 'r') as file:
            self.dataset = [json.loads(row) for row in file]
        self.use_context = use_context  # Aunque no lo vamos a utilizar

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Get instance of the dataset at the given index
        instance = self.dataset[index]

        return_tuple = tuple()
        question = self.dataset[index]['question']
        options = (
            instance['options']["1"] if isinstance(instance['options']["1"], str) else "",
            instance['options']["2"] if isinstance(instance['options']["2"], str) else "",
            instance['options']["3"] if isinstance(instance['options']["3"], str) else "",
            instance['options']["4"] if isinstance(instance['options']["4"], str) else "",
            instance['options']["5"] if isinstance(instance['options']["5"], str) else ""
        )
        label = instance['correct_option'] - 1  # -1 because it has to be [0, 1, 2, 3, 4] and not [1, 2, 3, 4, 5]
        return_tuple += (question, options, label)
        return return_tuple
