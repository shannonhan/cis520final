function [ X,tuples ] = stem(X,vocab)
    gen =0;
    if gen,
        stemmed = {};
        for i =1:numel(vocab),
            stemmed{i} = porterStemmer(vocab{i});
        end

        tuples = {};
        p=1;
        for i =1:numel(vocab),
            word = stemmed{i};
            booleanIndex = strcmp(word, stemmed);
            inds = find(booleanIndex);

            if(numel(inds)>1 && inds(1)==i),
                tuples{p}=inds;
                p=p+1;
            end   
        end
    
    else
        load 'tuples.mat';

        for t = 1:numel(tuples),
            tuple = tuples{t};
            base = X(:,tuple(1));
            for tt = 2:numel(tuple),
                base = base + X(:,tuple(tt));
                X(:,tuple(tt)) = 0;      
            end

        end
    end
    
    
    
    
end