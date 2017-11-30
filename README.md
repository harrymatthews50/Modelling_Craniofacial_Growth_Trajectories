##Overview
This package contains supporting code and data for the manuscript:

Matthews H., Penington A.,Hardiman R., Fan Y., Clement J., Kilpatrick N., Claes P. Modelling 3D craniofacial growth trajectories for population comparison and classification: Illustrated using sex-differences", submitted to Scientific Reports.

If re-using the methodology, data or code from this paper, please cite it appropriately.


###Available Data
The paper analyses 3D photographs of children which cannot be redistributed. However we provide an abstract coding of the shape of each head in the form of principal component projections, in 'Participant Data\PC_projections.xlsx'. These code the variability in the sample of shapes and much of the anlaysis can be performed using only these.

They were created using the following code in python:

```python
import numpy as np

#mean centre
Ymean = np.mean(Y,axis=0)
Y0 = Y - Ymean

# get eigenvectors (v)
u, s, v = np.linalg.svd(Y0)

# project to principal components
PC_projections = np.dot(v,Y0.T).T # transposed so that rows correspond to observations
```

where Y is n(observations) by p(point co-ordinates) matrix after alignment by generalised Procrustes analysis described in supplementary TextS1, section 1.2 of the manuscript. 

This recoding has no effect on the statistics used in the analysis (PLS-Regression, Procrustes Distance), although can affect results of some statistics.


This coding is completely reversible if the eigenvectors (v) are available:

```python
Y = np.dot(PC_projections,v) + Ymeam
```

In order to protect participant anonymity we do not provide the eigenvectors as this would allow reconstruction of the images.

In addition to these shape codings we provide participant age and sex in 'Participant_Metadata.xlsx' and their head size in "Mean_Distance_To_Centroid.xlsx".

We also provide all data generated from the analysis (except figures as they take up a lot of space, but these can be generated by running the 'Create_figures.py' script) in 'Generated Data'. This includes expected heads (as mesh '.obj' files) and growth vectors (as a matrix written to comma delimited '.txt' files) and classification scores (as an excel spreadsheet) .


###Available code
We have provided the complete code for the analysis. This uses some custom classes and functions, which are defined in 'Modules\ShapeStats'. It also requires some external Python packages that will need to be installed (see 'Installation' below).
The scripts for running the analysis are in: 'Main_Analysis_Scripts'. These are lettered (A, B, C, D and E) indicating the order they should be run in. This order is important since later scripts use output from the earlier ones that is saved to disk.
They can all be run in sequence by executing the 'run_all.py' file (see 'Running Scripts' below)

Some operations (e.g. creatiing expected heads) require the shape coding, described above, to be reversed. For the reasons above we do not provide the necessary data to do this. The points in the scripts where this is required are commented out as below, so that the code can still be seen but will not be executed when you run the scripts. The required output of these code blocks is always provided so that the anlysis can be run from start to finish. 


```python

# Operation requires back-projection    
#==============================================================================
# code goes here 
#==============================================================================

``` 


##Installation

First you need to clone or download this repository from GitHub. 

###Install Anaconda
If you do not have it already you will need to install a distribution of Python. I suggest Anaconda (and the instructions below assume this) since it gives you easy control over package and Python version. I have run this code using python 2.7 and the package versions below. It is quite possible it will work with other versions I just haven't tested it. 

You can install Anaconda or the 'light' version Miniconda by following the instructions here:

<https://conda.io/docs/user-guide/install/index.html>



###Create Python Environment

You then need to create a 'Python environment'. This is a self-contained environment running a particular Python version, with particular packages installed. These are handy because you can create many of them without affecting the others which is useful if you have other code that maybe requires a different version of Python or of some package. 

<https://realpython.com/blog/python/python-virtual-environments-a-primer/>


####Option 1: From Disc

We have created a copy of the required environment in the 'python_environment.yml' file. After installing anaconda you should be able to import this environment by running:

```
conda env create -n newEnvironment -f /path/python_environment.yml 
```

in the terminal and following the prompts


####Option 2: Manually 
If you have issues with the above try creating the environment manually

```
conda env create -n newEnvironment
```

activate the environment

```
activate newEnvironment
```

install the required packages by running the following commands one by one and following the prompts after each

```
conda install -c anaconda scipy=0.18
conda install -c anaconda pandas=0.19
conda install -c conda-forge scikit-learn=0.19
conda install -c anaconda mayavi=4.5
conda install -c conda-forge matplotlib=1.5 
conda install xlrd=1.0, openpyxl=2.4

```

##Running scripts

1. First, towards the top of each script in the 'Main_Analysis_Scripts' folder is the line

```python
modulepath = 'C:\Users\harry.matthews\Documents\Projects\Modelling_3D_Craniofacial_Growth_Curves_Supp_Material\Modules' #TODO Rememeber to update location on your machine
```

you need to change this path, in each script, to the location of the 'Modules' folder (from this repository) on your computer. This is necessary so that Python can find our custom classes and functions. If you ever get an 'import error' to do with 'ShapeStats' it is likely that this path is incorrect. The scripts load all data relative to that folder so if you have moved the 'Participant Data' or 'Generated Data'  folders you will also need to change the lines in the script that define the part_datapath and the gen_datapath variables in the scrpts accordingly


2. Activate the correct Python environment
```
activate newEnvironment
```
3. Run all scrpts from the terminal

```
python \path\run_all.py
```