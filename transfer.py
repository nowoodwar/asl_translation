import json

with open('ASL_Trainer.ipynb', "r", encoding='utf-8') as file:
    reader = json.load(file)["cells"]
    script = open("train.py", "w")
    
    for cell in reader:
        for line in cell["source"]:
            script.write(line)