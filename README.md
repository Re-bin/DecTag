# DecTag: The Deep Deconfounded Tag Recommender System

 The codes are associated with the following paper:

 >**Deep Deconfounded Content-based UGC Tag Recommendation with Causal Intervention,**  
 >Yaochen Zhu\*,  Xubin Ren\*,  Jing Yi  and  Zhenzhong Chen

## Environment

The codes are written in Python 3.7.12. with the following dependencies.

- numpy == 1.21.2
- pytorch == 1.8.0 (GPU version)
- cudatoolkit == 11.1.1
- scipy == 1.7.3

##  YT-8M-Causal dataset

The original YouTube-8M dataset can be accessed [here](https://research.google.com/youtube8m/download.html).

For preparation, create a data_split folder and unzip the sub-dataset into the folder.

## Examples to run the codes

  - **Train the deconfounded tag recommender on confounded datasets**: 

    ```python train_DecTag_{NFM, LightGCN}.py --dataset YT8M-Causal-{PH, AB} --split [1-5] --gpu [0-7]```   

    The trained model will be saved in the folder **./check\_point/YT8M-Causal-{PH, AB}/{NFM, LightGCN}/split_[1-5].**

    (Please create the folder first)

  - **Evaluate the model and save the testing results**:

    ```python test_DecTag_{NFM, LightGCN}.py --dataset YT8M-Causal-{PH, AB} --split [1-5] --gpu [0-7]```

    The results will be saved in the folder **./results/YT8M-Causal-{PH, AB}/{NFM, LightGCN}/split_[1-5].**

    (Please create the folder first)

 **For advanced usage of arguments, run the code with --help argument.**

**Thanks for your interest in our work**
