# Semantic-Correspondence

Project for the AML course

## Colab wandb

```url
https://colab.research.google.com/github/wandb/examples/blob/master/colabs/intro/Intro_to_Weights_%26_Biases.ipynb
```

## How to write comments : guideline

### Extension on VS-code

```
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

[ ] Revise the feature extraction and the PCK calculation part.
    [ ] Why is SAM so bad? Should we give it the prompts?
    [ ] Evaluate SAM 3 as well (project extension)

## TASK 2 - LIGHT FINETUNING 
  
### Source: 
    train.ipynb
  
### How to run
- compile and run the main_alt() --> the alternative version of main with parameters passed through arguments instead of bash 
  
### Problems
- it works inly with dinov3 (weird!)
- if u try with sam it says that sam does not have any "block" as layer --> not true
- probably it run only the first epoch and not continue to the second and so on
- ALWAYS COMPILE AND RUN THE SpairDataset class in the file before running the main