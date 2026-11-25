# Copyright 2026 Google LLC
use strict;
use warnings;
use PDL;
use PDL::ML::Metrics qw(
    accuracy
    precision_recall_f1
    confusion_matrix
    roc_auc
    mse
    rmse
    mae
    r2_score
    cosine_similarity
);

# 1. Classification evaluation
my $y_true  = long([1, 0, 1, 1, 0, 1, 0, 0]);
my $y_pred  = long([1, 0, 0, 1, 1, 1, 0, 0]);
my $y_score = double([0.92, 0.15, 0.42, 0.88, 0.61, 0.79, 0.10, 0.25]);

my $acc = accuracy($y_true, $y_pred);
my ($prec, $rec, $f1) = precision_recall_f1($y_true, $y_pred, 1);
my $cm  = confusion_matrix($y_true, $y_pred, 2);
my $auc = roc_auc($y_true, $y_score);

printf "Accuracy: %.3f | Precision: %.3f | Recall: %.3f | F1: %.3f | ROC-AUC: %.3f\n",
    $acc, $prec, $rec, $f1, $auc;
print 'Confusion matrix (pred x true):', "\n", $cm, "\n";

# 2. Regression evaluation
my $target = double([3.0, -0.5, 2.0, 7.0]);
my $pred   = double([2.5,  0.0, 2.0, 8.0]);

printf "MSE: %.4f | RMSE: %.4f | MAE: %.4f | R2: %.4f\n",
    mse($target, $pred),
    rmse($target, $pred),
    mae($target, $pred),
    r2_score($target, $pred);

# 3. Embedding / vector similarity
my $sim = cosine_similarity(double([1, 2, 3]), double([2, 4, 6]));
printf "Cosine similarity: %.4f\n", $sim;
