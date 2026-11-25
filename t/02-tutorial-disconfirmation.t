use strict;
use warnings;
use Test::More;
use PDL;
use PDL::ML::Metrics qw(
    accuracy
    precision_recall_f1
    confusion_matrix
    mse
    rmse
    mae
    r2_score
);

subtest 'multi-class protein localization evaluation tutorial' => sub {
    my $y_true = pdl(long, [0, 0, 1, 1, 2, 2]);
    my $y_pred = pdl(long, [0, 1, 1, 1, 2, 0]);

    my $acc = accuracy($y_true, $y_pred);
    ok(abs($acc - 4/6) < 1e-6, 'accuracy matches 4/6');

    my $cm = confusion_matrix($y_true, $y_pred, 3);
    is($cm->at(0, 0), 1, 'true class 0 predicted 0');
    is($cm->at(1, 0), 1, 'true class 0 predicted 1');
    is($cm->at(1, 1), 2, 'true class 1 predicted 1');
    is($cm->at(2, 2), 1, 'true class 2 predicted 2');

    my ($p, $r, $f1) = precision_recall_f1($y_true, $y_pred, 1);
    ok(abs($p - 2/3) < 1e-6, 'class 1 precision = 2/3');
    ok(abs($r - 1.0) < 1e-6, 'class 1 recall = 1.0');
    ok(abs($f1 - 0.8) < 1e-6, 'class 1 F1 = 0.8');
};

subtest 'disconfirmation: worst-case all-wrong predictions' => sub {
    my $y_true = pdl(long, [0, 0, 1, 1]);
    my $y_pred = pdl(long, [1, 1, 0, 0]);
    is(accuracy($y_true, $y_pred), 0, 'all-wrong accuracy is 0');

    my $reg_true = pdl(double, [1, 2, 3, 4]);
    my $reg_pred = pdl(double, [4, 3, 2, 1]);
    ok(mse($reg_true, $reg_pred) > 0, 'non-zero MSE on inverted predictions');
    ok(rmse($reg_true, $reg_pred) > 0, 'non-zero RMSE on inverted predictions');
    ok(mae($reg_true, $reg_pred) > 0, 'non-zero MAE on inverted predictions');
    ok(r2_score($reg_true, $reg_pred) < 0, 'negative R2 on worse-than-mean predictions');
};

done_testing();
