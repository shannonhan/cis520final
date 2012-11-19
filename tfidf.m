function Y = tfidf( X )
% tf seems not necessary since all song are similar in length.

tf = bsxfun(@rdivide,X,sum(X,2));

N= size(X,1);

csum = sum(X>0,1);
idf = log(N./csum);    
idf(isinf(idf)) = 0;


Y = bsxfun(@times, tf, idf);


end


