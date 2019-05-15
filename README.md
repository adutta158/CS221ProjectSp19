# CS221 Project Spring '19 
#### Classification <and Generation> of Hand-Drawn Doodles

## Overleaf links
- Project Proposal: https://www.overleaf.com/4471386335bswnnjtryszb
- Progress Report: https://www.overleaf.com/6295179215tkcmswtdvrdq
- Final Report: <TBA>

## Dataset
- Download data for the classes listed below from [here] (https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1) and place in the data folder.
- We are using the following classes for the project:
  1. Apple  
  2. Candle
  3. Door

## Creating environment
From the src folder, run "conda env create -f environment.yml"

Next run "conda activate cs221Project"

## Structure
### BaseModel class
Base class for classification models. Contains 2 methods:
* train
* predict

The different models that will be implemented will use this as the parent class.

### util.py
Contains utility functions including the function to load the data.

### experiment.py
Used to run experiments on different models to pick the best models and hyperparamters.

#test change
## Authors 
* **Sarah Robinson**
* **Aparajita Dutta**
