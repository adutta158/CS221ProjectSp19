# CS221 Project Spring '19 
#### Classification <and Generation> of Hand-Drawn Doodles

## Overleaf links
- [Project Proposal](https://www.overleaf.com/4471386335bswnnjtryszb)
- [Progress Report](https://www.overleaf.com/2186518721bfkwcrpyjcnk)
- [Final Report](https://www.overleaf.com/2837455734bbxwgzvpvmty)

## Dataset
- Download data for the classes listed below from [here](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pli=1) and place in the data folder.
- We are using the following classes for the project:
1. Apple
2. Banana
3. Bicycle
4. Birthday Cake
5. Butterfly
6. Candle
7. Computer
8. Door
9. Drums
10. Firetruck
11. Hat
12. Horse
13. Ice Cream
14. Leaf
15. Panda
16. Peanut
17. Pencil
18. Rainbow
19. Smiley Face
20. Snowman
21. Soccer Ball
22. Umbrella

## Creating environment
Install anaconda prompt from  https://www.anaconda.com/distribution/

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

## Authors 
* **Sarah Robinson**
* **Aparajita Dutta**
