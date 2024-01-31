# How to run:
It is recommended to work on a virtual environment such as venv. python 3.7-3.11 is supported.
First, install pytorch manually as it's not going to work from requirements.txt. 
copy the appropriate pytorch installation command from [here](https://pytorch.org/get-started/locally/)

Install from requirements.txt:
```
pip install -r requirements.txt
```

Run one of the examples:
example_pose_with_segmentation.py
example_pose_with_depth.py

First time you run it, it may take more time as it will download SAM pretrained weights.

