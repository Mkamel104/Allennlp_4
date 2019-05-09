# Classifying Names with a Character-Level RNN using Allennlp
We will be building and training a basic character-level RNN to classify words. A character-level RNN reads words as a series of characters - outputting a prediction and “hidden state” at each step, feeding its previous hidden state into each next step. We take the final prediction to be the output, i.e. which class the word belongs to.

# Run this model:
1. To run this project you need to install allennlp using this link:  [Allennlp](https://github.com/allenai/allennlp). It's perferable that you clone the allennlp project then run : 

          pip install allennlp
          
2. You can install allennlp using: 
    
         conda install allennlp -c pytorch -c allennlp -c conda-forge
         
3. Using Allennlp train option you can run the project as below command:

         allennlp train config.json -s ./output --include-package mylib
                    
# Results: 
The results should looks like below after running the above command using allen nlp options.

2019-04-15 22:54:22,144 - INFO - allennlp.common.util - Metrics: {<p>
  "training_duration": "00:00:01", <p> 
  "training_start_epoch": 0,<p>
  "training_epochs": 37,<p>
  "epoch": 37,<p>
  "training_accuracy": 0.9444444444444444,<p>
  "training_loss": 0.47344669699668884,<p>
  "validation_accuracy": 0.05555555555555555,<p>
  "validation_loss": 6.152934551239014,<p>
  "best_epoch": 8,<p>
  "best_validation_accuracy": 0.05555555555555555,<p>
  "best_validation_loss": 4.326207160949707<p>
}<p>
