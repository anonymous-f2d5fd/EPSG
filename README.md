EPSG: Event Participation Sequence Generation
---------------
This is implementation of our work EPSG.

requirements
------------
The script has been tested running under Python 3.7.6, with the following packages installed (along with their dependencies):

- `numpy==1.16.4`
- `torch==1.5.1`
- `pandas==1.0.1`
- `networkx==2.4`
- `torch-geometric==1.6.1`
- `torch-cluster==1.5.6`
- `torch-scatter==2.0.5`
- `torch-sparse==0.6.6`

Overview
--------------
Here we provide the implementation of RAA,RIA,EPSGF,its variants and datasets.

The folder is organised as follows:
- `data/` contains:
    - `meetup_ca` contains:
        * `data.pt`  a saved pytorch geometric dataset object contains meetup_ca 
    - `meetup_sg` contains:
        * `data.pt`  a saved pytorch geometric dataset object contains meetup_sg
- `model.py` contains EPSG and its variants.
- `util.py` contains useful functions for evaluation metrics, dataset splitting, training ande testing model.
- `function.py` contains functions for training all methods.
- `method.py` contains heuristic algorithms RAA,RIA and baseline Random.
- `train.py` is used to training Random,RIA,RAA,EPSGF on the dataset.

How to run
---------------
`python train.py`