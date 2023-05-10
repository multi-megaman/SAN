import inference2
import torch
import os
from datetime import datetime
import matplotlib.pyplot as plt

actualDate = datetime.now().strftime("%d-%m-%Y %H-%M-%S")
actualDevice = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpointsPath = "./checkpoints"
checkpointsFolder = [f.path for f in os.scandir(os.path.abspath(checkpointsPath)) if f.is_dir()]   #ex: ["C:/checkpoints/model_1","C:/checkpoints/model_2"]
checkpointsName = [os.path.basename(x) for x in checkpointsFolder]                                 #ex: ["model_1","model_2"]
checkpointsFile = [os.path.join(x,(os.path.basename(x)+".pth"))  for x in checkpointsFolder]       #ex: ["C:/checkpoints/model_1/model_1.pth","C:/checkpoints/model_2/model_2.pth"]
checkpointsConfig = [os.path.join(x,"config.yaml")  for x in checkpointsFolder]                    #ex: ["C:/checkpoints/model_1/config.yaml","C:/checkpoints/model_2/config.yaml"]
actualModelConfig = ""

inferencesInfos = {}

# print(checkpointsFolder)
# print(checkpointsFile)
for x in range(len(checkpointsFolder)):
    exp_rate, pred_time_mean, experiment =  inference2.Make_inference(checkpointFolder=checkpointsFolder[x],
                                            configPath=checkpointsConfig[x],
                                            checkpointPath=checkpointsFile[x], 
                                            device=actualDevice,
                                            date=actualDate)
    
    if experiment in inferencesInfos:
        inferencesInfos[experiment]["exp_rate"].append(exp_rate)
        inferencesInfos[experiment]["time_mean"].append(pred_time_mean)
        inferencesInfos[experiment]["model_name"].append(checkpointsName[x])
    else:
        inferencesInfos[experiment] = {"exp_rate":[exp_rate],
                                       "time_mean": [pred_time_mean],
                                       "model_name": [checkpointsName[x]]}

fig, ax = plt.subplots(figsize=(12,8))
plt.xlabel("exp_rate", size=12)
plt.ylabel("inference_time_mean", size=12)
plt.title("Inferences", size=15)

for n,experiment in enumerate(inferencesInfos):
    plt.plot(inferencesInfos[experiment]["exp_rate"],inferencesInfos[experiment]["time_mean"],'o')
    for i, modelName in enumerate(inferencesInfos[experiment]["model_name"]):
        ax.annotate(str(experiment) + " " + str(modelName), (inferencesInfos[experiment]["exp_rate"][i], inferencesInfos[experiment]["time_mean"][i]))

plt.show()


print(str(inferencesInfos))

