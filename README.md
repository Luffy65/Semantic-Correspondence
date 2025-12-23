# Semantic-Correspondence

Project for the AML course.

## Colab wandb

```url
https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb
```

## TODO

[ ] Revise the feature extraction and the PCK calculation part in task 1.
    [ ] Why is SAM so bad? Should we give it the prompts?
    [ ] Evaluate SAM 3 as well (project extension), ensuring it works correctly.
[ ] (for both task 1 and task 2) Clone the repo of dinov2 (and download the checkpoints) instead of using the huggingface implementation.
[ ] (optional but good) remove redundancy in the two notebooks, by doing common operations only once. Examples of common operations between the two notebooks are: cloning repositories, instantiating models, downloading data, defining paths.
[V] in train.ipynb, the plot of history is showing batch: xxx | loss: xxxx | keypoints: xx| byt it should be PCK instead of keypoints
[V] Fix tensor dimension problem for SAM in train.ipynb

### Doubts

* what is that train.py? can we remove it?
  * train.py was replaced bu train.ipynb
