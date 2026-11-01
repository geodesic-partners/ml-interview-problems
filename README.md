# Machine learning coding interview problems, from scratch in NumPy

40 problems of the kind set in the implementation round of ML engineering, research and quant
interviews: write the softmax, the backward pass, the attention layer or the AUC yourself, without a library.

Each file holds the problem statement, a starter function and the test table. Write the function, then run the
file: `python problems/stable_softmax.py`. NumPy is the only dependency.

The lessons, hints, worked solutions and an in-browser editor are at
https://geodesicpartners.co.uk/machine-learning-coding?utm_source=github&utm_medium=repo

| Topic | Problem | Level | Solution |
|---|---|---|---|
| Numerical stability | [Softmax that never overflows](problems/stable_softmax.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-stable-softmax?utm_source=github&utm_medium=repo) |
| Numerical stability | [Cross-entropy loss and its gradient from logits](problems/cross_entropy_from_logits.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-cross-entropy-from-logits?utm_source=github&utm_medium=repo) |
| Numerical stability | [Binary cross-entropy from logits](problems/bce_with_logits.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-bce-with-logits?utm_source=github&utm_medium=repo) |
| Linear models | [Ridge regression in closed form](problems/ridge_regression.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-ridge-regression?utm_source=github&utm_medium=repo) |
| Linear models | [Linear regression by gradient descent](problems/linear_regression_gradient_descent.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-linear-regression-gradient-descent?utm_source=github&utm_medium=repo) |
| Linear models | [Logistic regression with an L2 penalty](problems/logistic_regression.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-logistic-regression?utm_source=github&utm_medium=repo) |
| Neural networks | [Forward pass of a two-layer network](problems/mlp_forward_pass.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-mlp-forward-pass?utm_source=github&utm_medium=repo) |
| Neural networks | [Backpropagation through a two-layer network](problems/mlp_backpropagation.py) | Hard | [lesson and solution](https://geodesicpartners.co.uk/ml-mlp-backpropagation?utm_source=github&utm_medium=repo) |
| Neural networks | [The Adam optimiser, with bias correction](problems/adam_optimizer.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-adam-optimizer?utm_source=github&utm_medium=repo) |
| Training a network | [Inverted dropout, forward and backward](problems/inverted_dropout.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-inverted-dropout?utm_source=github&utm_medium=repo) |
| Training a network | [Gradient clipping by global norm](problems/gradient_clipping.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-gradient-clipping?utm_source=github&utm_medium=repo) |
| Training a network | [Linear warm-up with cosine decay](problems/learning_rate_schedule.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-learning-rate-schedule?utm_source=github&utm_medium=repo) |
| Deep learning layers | [Layer normalisation](problems/layer_normalization.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-layer-normalization?utm_source=github&utm_medium=repo) |
| Deep learning layers | [Batch normalisation in training mode](problems/batch_normalization.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-batch-normalization?utm_source=github&utm_medium=repo) |
| Deep learning layers | [Scaled dot-product attention with a causal mask](problems/scaled_dot_product_attention.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-scaled-dot-product-attention?utm_source=github&utm_medium=repo) |
| Deep learning layers | [2-D convolution with stride and padding](problems/conv2d.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-conv2d?utm_source=github&utm_medium=repo) |
| Sequence models | [One step of an LSTM cell](problems/lstm_cell.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-lstm-cell?utm_source=github&utm_medium=repo) |
| Sequence models | [One step of a GRU cell, as PyTorch computes it](problems/gru_cell.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-gru-cell?utm_source=github&utm_medium=repo) |
| Sequence models | [Beam search decoding](problems/beam_search.py) | Hard | [lesson and solution](https://geodesicpartners.co.uk/ml-beam-search?utm_source=github&utm_medium=repo) |
| Embeddings and retrieval | [Top-k search by cosine similarity](problems/cosine_top_k.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-cosine-top-k?utm_source=github&utm_medium=repo) |
| Embeddings and retrieval | [The InfoNCE contrastive loss](problems/info_nce_loss.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-info-nce-loss?utm_source=github&utm_medium=repo) |
| Embeddings and retrieval | [Skip-gram with negative sampling: loss and gradients](problems/negative_sampling.py) | Hard | [lesson and solution](https://geodesicpartners.co.uk/ml-negative-sampling?utm_source=github&utm_medium=repo) |
| Classic algorithms | [k-nearest neighbours classifier](problems/k_nearest_neighbors.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-k-nearest-neighbors?utm_source=github&utm_medium=repo) |
| Classic algorithms | [k-means clustering (Lloyd's algorithm)](problems/k_means_clustering.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-k-means-clustering?utm_source=github&utm_medium=repo) |
| Classic algorithms | [Principal component analysis](problems/pca.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-pca?utm_source=github&utm_medium=repo) |
| Classic algorithms | [Best decision-tree split by Gini impurity](problems/decision_tree_split.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-decision-tree-split?utm_source=github&utm_medium=repo) |
| Probabilistic models | [Gaussian naive Bayes classifier](problems/gaussian_naive_bayes.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-gaussian-naive-bayes?utm_source=github&utm_medium=repo) |
| Probabilistic models | [One EM step for a Gaussian mixture](problems/gmm_em_step.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-gmm-em-step?utm_source=github&utm_medium=repo) |
| Probabilistic models | [The Viterbi algorithm for a hidden Markov model](problems/viterbi_algorithm.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-viterbi-algorithm?utm_source=github&utm_medium=repo) |
| Probabilistic models | [Stationary distribution of a Markov chain](problems/markov_stationary_distribution.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-markov-stationary-distribution?utm_source=github&utm_medium=repo) |
| Statistics in code | [Running mean and variance in one pass](problems/running_variance.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-running-variance?utm_source=github&utm_medium=repo) |
| Statistics in code | [ROC AUC from scores, with ties](problems/roc_auc.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-roc-auc?utm_source=github&utm_medium=repo) |
| Statistics in code | [Spearman rank correlation (the information coefficient)](problems/spearman_rank_correlation.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-spearman-rank-correlation?utm_source=github&utm_medium=repo) |
| Statistical inference | [The autocorrelation function of a series](problems/autocorrelation_function.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-autocorrelation-function?utm_source=github&utm_medium=repo) |
| Statistical inference | [Regression coefficients with standard errors and t-statistics](problems/ols_standard_errors.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-ols-standard-errors?utm_source=github&utm_medium=repo) |
| Statistical inference | [An exact permutation test for a difference in means](problems/permutation_test.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-permutation-test?utm_source=github&utm_medium=repo) |
| Statistical inference | [A one-dimensional Kalman filter](problems/kalman_filter_1d.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-kalman-filter-1d?utm_source=github&utm_medium=repo) |
| Ranking and evaluation metrics | [NDCG at k](problems/ndcg_at_k.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-ndcg-at-k?utm_source=github&utm_medium=repo) |
| Ranking and evaluation metrics | [Expected calibration error](problems/expected_calibration_error.py) | Medium | [lesson and solution](https://geodesicpartners.co.uk/ml-expected-calibration-error?utm_source=github&utm_medium=repo) |
| Ranking and evaluation metrics | [Macro-averaged F1 across classes](problems/macro_f1.py) | Easy | [lesson and solution](https://geodesicpartners.co.uk/ml-macro-f1?utm_source=github&utm_medium=repo) |

The problems and tests are original. Every reference solution is checked against an independent implementation
(exact arithmetic, brute force or finite differences) on these tests and on random inputs.
