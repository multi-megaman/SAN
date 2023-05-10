import inference2
import torch
import os
from datetime import datetime

actualDate = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
actualDevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpointsPath = "./checkpoints"
checkpointsFolder = [f.path for f in os.scandir(os.path.abspath(checkpointsPath)) if f.is_dir()]   #ex: ["C:/checkpoints/model_1","C:/checkpoints/model_2"]
checkpointsFile = [os.path.join(x,(os.path.basename(x)+".pth"))  for x in checkpointsFolder]       #ex: ["C:/checkpoints/model_1/model_1.pth","C:/checkpoints/model_2/model_2.pth"]
checkpointsConfig = [os.path.join(x,"config.yaml")  for x in checkpointsFolder]                    #ex: ["C:/checkpoints/model_1/config.yaml","C:/checkpoints/model_2/config.yaml"]
actualModelConfig = ""

# print(checkpointsFolder)
# print(checkpointsFile)
for x in range(len(checkpointsFolder)):
    exp_rate, pred_time_mean, experiment = inference2.Make_inference(checkpointFolder=checkpointsFolder[x],
                              configPath=checkpointsConfig[x],
                              checkpointPath=checkpointsFile[x], 
                              device=actualDevice,
                              date=actualDate)

