# Semantic-Correspondence

Project for the AML course.

## Colab wandb

```url
https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb
```

## TODO in ORDER

- The evaluation of sam requires 3h, fix that (lower image res)
- Remove normalization on imagened statistics before feeding input to SAM (?)
- Better visualization of the results (example: comparing the ground truth keypoint of an image with the predicted keypoint). We should plot at every finetune; at the end of each epoch we update the plot. The plot informs us on how the finetune is going / has gone.
- Evaluate the finetuned models using window softargmax
- Implement task 4: evaluate on a new dataset (spwillow), both the untrained and the trained models
- (optional): add sam3 for task4

## Reporting results

Results will be reported per keypoint and per image, following the definition in DIFT [1]. Here is the definition:

Some works use the total number of correctly-predicted points in the whole dataset (or each category split) divided by the total number of predicted points as the final PCK, while some works first calculate a PCK value for each image and then average it across the dataset (or each category split). We denote the first metric as PCK per point and the second as PCK per image.

In the baselines, our "overall" PCK is "per-keypoint" (total correct / total keypoints), and also we have the "per-image" (PCK computed for each image pair, then report mean/std/min/max).

## Explaining results

SAM performs way worse than the dino models. This is because SAM features are optimized for segmentation boundaries, not semantic similarity. They encode "what's an object edge" rather than "what's semantically similar." Also, SAM 1 was designed to be prompted with visual inputs (points, bounding boxes, or masks), but in this baseline we're not passing any prompts. SAM features aren't well-suited for our task.
