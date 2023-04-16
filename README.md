The Advanced Computer Vision Project Repository
## How to build the environment
conda env create -f environment.yml
## How to build datasets corresponding to different hardness levels referring to robot's operational region
run main.py 
## How to regenerate the results
After building the environment and selecting it, run the following jupyter notebooks:  
1- generateEnvToy.ipynb to see the results for differentially private noise in MNIST image and histograms  
2- optimization.ipynb to see the boxplot results shown in the paper  
3- generalImp.ipynb to see the concave utility results shown in the paper  
