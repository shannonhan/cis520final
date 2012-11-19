function ig = multi_ig(X, Y)
% DT_CHOOSE_FEATURE_MULTI - Selects feature with maximum multi-class IG.
%
% Usage:
% 
%   [FIDX FVAL MAX_IG] = dt_choose_feature(X, Z, XRANGE, COLIDX)
%
% Given N x D data X and N x K indicator labels Z, where X(:,j) can take on values in XRANGE{j}, chooses
% the split X(:,FIDX) <= VAL to maximize information gain MAX_IG. I.e., FIDX is
% the index (chosen from COLIDX) of the feature to split on with value
% FVAL. MAX_IG is the corresponding information gain of the feature split.
%
% Note: The relationship between Y and Z is that Y(i) = find(Z(i,:)).
% Z is the categorical representation of Y: Z(i,:) is a vector of all zeros
% except for a one in the Y(i)'th column.
% 
% Hint: It is easier to compute entropy, etc. when using Z instead of Y.
%
% SEE ALSO
%    DT_TRAIN_MULTI
%s
% YOUR CODE GOES HERE

%%% Compute the entropy of Y

% could vectorize using repmat
%X = [1 2 3 ; 3 2 1]';
%Z = eye(3);
%colidx = [1 , 2];
%Xrange = {};
%Xrange{1} = [1,2,3];
%Xrange{2} = [1,2,3];

K = max(Y);
Z = bsxfun(@eq, Y, 1:K);
[N, K]= size(Z);
y_k = sum(Z);
Y = y_k / sum(y_k);
H = multi_entropy(Y');
Xrange={};
for i = 1:size(X, 2)
    Xrange{i} = unique(X(:,i));
end

colidx=1:size(X, 2);


%%%
ig = zeros(numel(Xrange), 1);
split_vals = zeros(numel(Xrange), 1);

t = CTimeleft(numel(colidx));
fprintf('Evaluating features on %d examples: ', N);
for i = colidx
    t.timeleft();
    
    % Check for constant values.
    if numel(Xrange{i}) == 1
        ig(i) = 0;
        split_vals(i) = 0;
        continue;
    end
    
    % Compute up to 10 possible splits of the feature.
    r = linspace(double(Xrange{i}(1)), double(Xrange{i}(end)), double(min(10, numel(Xrange{i}))));
    split_f = bsxfun(@le, X(:,i), r(1:end-1));
    
    % Compute conditional entropy of all possible splits.
    % Compute p(x) for all splits
    px = mean(split_f);
    % Compute 
    %y_eq = bsxfun(@eq, y, 1:K);   y_eq is the same as Z here!
    p_given_x = zeros(K, length(r)-1);
    p_given_notx = zeros(K, length(r)-1);

    for j = 1:K,
        y_given_x = bsxfun(@and, Z(:,j), split_f);
        y_given_notx = bsxfun(@and, Z(:,j), ~split_f);
        % Compute p(Yk|X) and p(Yk|~X)
        p_given_x(j,:)= sum(y_given_x)./sum(split_f);
        p_given_notx(j,:)= sum(y_given_notx)./sum(~split_f);
    end
    cond_H = px.*multi_entropy(p_given_x) + ...
        (1-px).*multi_entropy(p_given_notx);    
    %%%
    
    % Choose split with best IG, and record the value split on.
    [ig(i) best_split] = max(H-cond_H);
    split_vals(i) = r(best_split);
end