import json
from dataset import CasiMedicosDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import os

SETS_PATH = [
    '/content/drive/MyDrive/Prompting/LLama3-70B/Argumentazio_Emaitzak_Neutralized/Ondo_Eginda/en.train_casimedicos_Llama3-70_Neutralized.jsonl',
    '/content/drive/MyDrive/Prompting/LLama3-70B/Argumentazio_Emaitzak_Neutralized/Ondo_Eginda/en.test_casimedicos_Llama3-70_Neutralized.jsonl',
    '/content/drive/MyDrive/Prompting/LLama3-70B/Argumentazio_Emaitzak_Neutralized/Ondo_Eginda/en.dev_casimedicos_Llama3-70_Neutralized.jsonl'
]

for set_path in SETS_PATH:
    set = CasiMedicosDataset(
        jsonl_path=set_path,
        use_context=False
    )

    new_file_path = f'/content/drive/MyDrive/IKER/dataset/Orekatuak/{os.path.basename(set_path)}'

    with open(new_file_path, 'w') as file:
        dataloader = DataLoader(set, batch_size=1, shuffle=False)

        for instance in dataloader:
            question = instance[0][0]
            correct_option_id = instance[2].item()
            correct_option = instance[1][correct_option_id][0]
            incorrect_options = [instance[1][index][0] for index, option in enumerate(instance[1]) if index != correct_option_id]

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'question': question,
                'options': {
                    "1": correct_option,
                    "2": incorrect_options[0],
                    "3": incorrect_options[1],
                    "4": incorrect_options[2],
                    "5": incorrect_options[3]
                },
                'correct_option': 1
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'question': question,
                'options': {
                    "1": incorrect_options[0],
                    "2": correct_option,
                    "3": incorrect_options[1],
                    "4": incorrect_options[2],
                    "5": incorrect_options[3]
                },
                'correct_option': 2
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'question': question,
                'options': {
                    "1": incorrect_options[0],
                    "2": incorrect_options[1],
                    "3": correct_option,
                    "4": incorrect_options[2],
                    "5": incorrect_options[3]
                },
                'correct_option': 3
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'question': question,
                'options': {
                    "1": incorrect_options[0],
                    "2": incorrect_options[1],
                    "3": incorrect_options[2],
                    "4": correct_option,
                    "5": incorrect_options[3]
                },
                'correct_option': 4
            }) + '\n')

            np.random.shuffle(incorrect_options)
            file.write(json.dumps({
                'question': question,
                'options': {
                    "1": incorrect_options[0],
                    "2": incorrect_options[1],
                    "3": incorrect_options[2],
                    "4": incorrect_options[3],
                    "5": correct_option
                },
                'correct_option': 5
            }) + '\n')

    print(f"File {new_file_path} created.")
