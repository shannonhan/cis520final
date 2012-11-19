clear all
close all
%%load all files
params = {[0.0003 6],[0.00025 6],[0.0003 5]};
results={};


for parai = 1:numel(params),
    load ../data/music_dataset.mat
    [Xt_lyrics] = make_lyrics_sparse(train, vocab);
    [Xq_lyrics] = make_lyrics_sparse(quiz, vocab);


    Yt = zeros(numel(train), 1);
    for i=1:numel(train)
        Yt(i) = genre_class(train(i).genre);
    end

    Xt_audio = make_audio(train);
    Xq_audio = make_audio(quiz);

    param = params{parai};

    %%pre-processing, generate X,Y

    cols = feature_selection_lyrics(Xt_lyrics, Yt, Xq_lyrics,vocab, param(1), param(2));
    X=Xt_lyrics(:,cols);
    X=tfidf(X);
    size(X)
    Y = Yt;
    %%set parameters
    %store parameters to be choosed here. 

    iter_count=3;
    fold_count=6;
    test1=1;
    test2=0;

    N = size(X,1);
    trainN = round(N/fold_count*(fold_count-1));
    testN = N-trainN;

    if test1
        part = make_xval_partition(N , fold_count);
        n = length(part);
        errors = zeros(1, fold_count);
        infos = {};
        for test_i = 1:fold_count,
            test_X = X(part == test_i,:);
            test_Y = Y(part == test_i,:);

            train_X = X(part ~= test_i,:);
            train_Y = Y(part ~= test_i,:);

            k = @(x,x2) kernel_intersection(x, x2);
            [test_err info]=kernel_libsvm(train_X, train_Y, test_X, test_Y,k);
            infos{test_i}=info;
            errors(test_i) = test_err;    
        end
        error = mean(errors)
        result.error = error;
        result.infos = infos;
        result.errors = errors;
        results{parai} = result;
    end
    
    
end
