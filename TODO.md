bugs
----
 * bug in nfold_CV where all of input data is used to trained the model
 * add label vector compatability with Float32 for regression
 * increase the performance of current adaboost implementation

tests and validation
--------------------
 * add benchmarks for other functions
 * add more tests

features
--------
 * boost : logit boost
 * boost : gradient boost
 * boost : SAMME.R
 * boost : adaboost for regression


 * trees : add feature importance
 * trees : min_weights_leaf + min_weights_split
 * trees : least absolute deviation
 * trees : friedman's improvement score
 * trees : sample-weight support
 * trees : user input purity criterion for classification

 * prune : post-pruning with validation data

optimizations
-------------
 * "compact" treeclassifier-esque leaves 
 * vectorize : change `nc[:] .= 0` to loops

api and cleanliness
-------------------
 * use range [0,1] for purity_thresh in new implementations of `prune_tree` (currently commented out)
 * standardize variable names to snake case
