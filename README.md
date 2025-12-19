# Semantic-Correspondence

Project for the AML course

## Colab wandb

```url
https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb
```

## How to write comments : guideline

### Extension on VS-code

```bash
    Better comments
```

### Guideline

```C
//* highlight important thing
//? something to ask ? 
//! urgent thing
//TODO something to do 
```

## TODO

[ ] Revise the feature extraction and the PCK calculation part in task 1.
    [ ] Why is SAM so bad? Should we give it the prompts?
    [ ] Evaluate SAM 3 as well (project extension)
[ ] Clone the repo of dinov2 (and download the checkpoints) instead of using the huggingface implementation.
[ ] (optional but good) remove redundancy in the two notebooks by doing common operations only once.

## TASK 2 - LIGHT FINETUNING
  
### Source

train.ipynb
  
### How to run

- run the last cells at the bottom of the page with preset configurations. REMEMBER to set debug=False in the main_alt() method if you don't want to debug.
- DEBUG MODE : set debug=True in the main_alt() method.
  
### Problems

- tensor dimension problem for SAM

### Doubts

* what is that train.py? can we remove it?
* Why do we have 2 main functions?