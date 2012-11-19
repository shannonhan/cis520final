function Y = tfidf( X )
% tf seems not necessary since all song are similar in length.

tf = bsxfun(@rdivide,X,sum(X,2));

N= size(X,1);
    
idf = log(N./sum(X>0,1));    


Y = bsxfun(@times, tf, idf);


end


