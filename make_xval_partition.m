function [part] = make_xval_partition(n, n_folds)
% MAKE_XVAL_PARTITION - Randomly generate cross validation partition.
%
% Usage:
%
%  PART = MAKE_XVAL_PARTITION(N, N_FOLDS)
%
% Randomly generates a partitioning for N datapoints into N_FOLDS equally
% sized folds (or as close to equal as possible). PART is a 1 X N vector,
% where PART(i) is a number in (1...N_FOLDS) indicating the fold assignment
% of the i'th data point.

% YOUR CODE GOES HERE


d =  ceil(n/n_folds);
shuffled = randperm(n);
[~, y]=  meshgrid(1: d, 1:n_folds);
y =  reshape(y, 1, numel(y));
y = y(1:n);
part= zeros(1, n);
part(shuffled) = y;




