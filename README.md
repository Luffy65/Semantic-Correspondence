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
- run the last cells at the bottom of the page with pre setted configurations. REMEMBER to set debug=False in the main_alt() method.
- DEBUG MODE : set debug=True in the main_alt() method
  
### Problems
- tensor dimension problem for SAM