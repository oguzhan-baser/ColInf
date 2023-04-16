# The Advanced Computer Vision Project Repository
## How to build the environment
conda env create -f environment.yml
## How to build datasets corresponding to different hardness levels referring to robot's operational region
run main.py  
(the generated environments are already given in hardEnvs folder)
## How to train models for different environments with different final nominal accuracies
comment out the part in the main.py and rerun (will take time to generate the same models so these models are given in models_original folder to be used to generate the results in the paper)
## How to regenerate the results
After building the environment and selecting it, run the following jupyter notebooks:  
1- generateEnvToy.ipynb to see the results for differentially private noise in MNIST image and histograms  
2- optimization.ipynb to see the boxplot results shown in the paper  
3- generalImp.ipynb to see the concave utility results shown in the paper  
