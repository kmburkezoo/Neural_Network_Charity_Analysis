# Neural_Network_Charity_Analysis

## Overview
This analysis uses TensorFlow's Keras sequential model to perform deep machine learning on a dataset of over 34,000 charity funding applications. The neural network model considers a number of features from within the dataset and uses them to predict whether a particular charity will be considered to have used its grant effectively. Theoretically, this model could then be used to evaluate new applications and determine whether funds should be granted.

## Results
### Preprocessing
* The target variable is IS_SUCCESSFUL, which indicates whether an organization is considered to have used its funds effectively
* Feature variables include:
    * APPLICATION_TYPE—Alphabet Soup application type
    * AFFILIATION—Affiliated sector of industry
    * CLASSIFICATION—Government organization classification
    * USE_CASE—Use case for funding
    * ORGANIZATION—Organization type
    * INCOME_AMT—Income classification
    * ASK_AMT—Funding amount requested
    * Additionally, two columns were considered features in the first attempt at training a model but removed during optimization since their values were so heavily skewed
        * STATUS—Active status
        * SPECIAL_CONSIDERATIONS—Special consideration for application
* Two columns that provided identification information were also dropped:
    * EIN
    * NAME
### Compiling, Training, and Evaluation
| |Hidden Layers|Neurons|Hidden Layer Activation Functions|Output Layer Activation Function|Loss|Accuracy|
|---|---|---|---|---|---|---|
|Initial model|2|80, 30|relu, relu|sigmoid|0.65|0.63|
|Optimization 1|2|120, 40|relu, relu|sigmoid|0.62|0.64|
|Optimization 2|3|80, 30, 20|relu, relu, relu|sigmoid|0.82|0.43|
|Optimization 3|3|100, 40, 20|relu, sigmoid, sigmoid|sigmoid|0.69|0.53|

In my attempts to optimize this model, I:
* Removed two features with heavily skewed value distribution (Optimizations 1, 2, 3)
* Increased the number of neurons (optimization 1)
* Returned to the original number of neruons but added an additional hidden layer (optimization 2)
* Increased the number of neurons in the first two layers and changed the activation function for hidden layers 2 and 3 from relu to sigmoid

None of the optimization attempts met the desired threshold of 75% accuracy, likely because I was conservative in the scale of changes I made to the model, rather than throwing everything I could at it all at once. My first attempt at optimization, by simply adding more neurons to the existing layers, was most successful, but even it made only a minute improvement. Adding additional layers with similar numbers of neurons as the initial model made it less accurate than the initial model, as did changing some of the activation functions from relu to sigmoid.

## Summary
 While a neural network is a reasonable choice for such a complicated data set (over 40 features, after encoding), the results obtained from this model are disappointing. Additional attempts at optimization would be required to determine which activation functions are most effective; this, in addition to increasing the number of neurons and layers, would likely be more effective at optimizing performance than any of these changes individually.

Alternately, other models could be considered. Since the desired task is binary classification, either a Random Forest or Support Vector Machine model should be explored to see if they are more effective than the neural network created here.