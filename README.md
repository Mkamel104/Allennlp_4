# Classifying Names with a Character-Level RNN using Allennlp
We will be building and training a basic character-level RNN to classify words. A character-level RNN reads words as a series of characters - outputting a prediction and “hidden state” at each step, feeding its previous hidden state into each next step. We take the final prediction to be the output, i.e. which class the word belongs to.

# Run this model:
1- To run this project you need to install allennlp using this link:  https://github.com/allenai/allennlp. 

2. You need to run: 
                    allennlp train name_model.json -s ./output --include-package mylib
                    
# Results: 
The results should looks like below after running the above command using allen nlp options.

2019-04-15 22:54:22,144 - INFO - allennlp.common.util - Metrics: {
  "training_duration": "00:00:01",
  "training_start_epoch": 0,
  "training_epochs": 37,
  "epoch": 37,
  "training_accuracy": 0.9444444444444444,
  "training_loss": 0.47344669699668884,
  "validation_accuracy": 0.05555555555555555,
  "validation_loss": 6.152934551239014,
  "best_epoch": 8,
  "best_validation_accuracy": 0.05555555555555555,
  "best_validation_loss": 4.326207160949707
}
