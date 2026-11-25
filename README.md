# PDL::ML::Metrics

Classification, regression, ranking, and vector-similarity evaluation metrics operating directly on [PDL](https://pdl.perl.org/) ndarrays.

## Installation

```bash
cpanm .
```

Or via ExtUtils::MakeMaker:

```bash
perl Makefile.PL
make
make test
make install
```

## Quickstart

```perl
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

my $y_true  = long([1, 0, 1, 1, 0, 1, 0, 0]);
my $y_pred  = long([1, 0, 0, 1, 1, 1, 0, 0]);
my $y_score = double([0.92, 0.15, 0.42, 0.88, 0.61, 0.79, 0.10, 0.25]);

my $acc = accuracy($y_true, $y_pred);
my ($prec, $rec, $f1) = precision_recall_f1($y_true, $y_pred, 1);
my $cm  = confusion_matrix($y_true, $y_pred, 2);
my $auc = roc_auc($y_true, $y_score);

printf "Accuracy=%.3f F1=%.3f ROC-AUC=%.3f\n", $acc, $f1, $auc;
```

## Documentation & Examples

- Runnable tutorial: `perl examples/quickstart.pl` (see [`examples/quickstart.pl`](examples/quickstart.pl))
- Full POD documentation: `perldoc PDL::ML::Metrics` (or [`lib/PDL/ML/Metrics.pm`](lib/PDL/ML/Metrics.pm))

## License

Copyright 2026 Google LLC. Licensed under the the same terms as Perl itself.
