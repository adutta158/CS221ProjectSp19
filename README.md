# CS221 Project Spring '19 
#### Classification <and Generation> of Hand-Drawn Doodles

## Overleaf links
- [Project Proposal](https://www.overleaf.com/4471386335bswnnjtryszb)
- [Progress Report](https://www.overleaf.com/2186518721bfkwcrpyjcnk)
- [Final Report](https://www.overleaf.com/2837455734bbxwgzvpvmty)

## Dataset
- Download data for the classes listed below from [here](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1) and place in the data folder.
- We are using the following classes for the project: Apple, Banana, Bicycle, Birthday Cake, Butterfly, Candle, Computer, Door, Drums, Firetruck, Hat, Horse, Ice Cream, Leaf, Panda, Peanut, Pencil, Rainbow, Smiley Face, Snowman, Soccer Ball and Umbrella

## Creating environment
Install anaconda prompt from  https://www.anaconda.com/distribution/

From the src folder, run "conda env create -f environment.yml"

Next run "conda activate cs221Project"

## Structure
### BaseModel class
Base class for classification models. Contains 2 methods:
* train
* predict

The different models (*_model.py) that are implemented use this as the parent class.

### util.py
Contains utility functions including the function to load the data.

### experiment.py
Used to run experiments on different models to pick the best models and hyperparamters.

### demo_app.py
Demonstration app that let's you draw one of the 22 classes and tries to identify it. Run using "python demo_app.py".

## Authors 
* **Sarah Robinson**
* **Aparajita Dutta**
