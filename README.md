# feature_selection

Does feature selection methods improve the predictive performance and computation time of a range of well-known machine learning models? This is the answer that this project aimed to answer.<br>We conducted a large scale experiment through OpenML on a subset of the OpenML-CC18 dataset, a well-curated classification dataset. We find that feature selection can greatly decrease training and prediction time, but does little in terms of predictive performance gain. In fact, it may instead hurt performance slightly.

## To run the code

First need to create a virtual environment and install required libraries

$ python3 -m venv venv

$ . venv/bin/activate

$ pip install -r req.txt

Next go to the code directory

cd feature_selection/main

Now you are ready to run the code, if you want to run it in background it is recommended to use deatached process
$ nohup python main.py > log.file &

## Results

If Json 'res' folder is in same directory as results.ipynb:
All cells can be run in order to plot results from the report
