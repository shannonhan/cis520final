function ranks = predict_genre(Xt_lyrics, Xq_lyrics, ...
                               Xt_audio, Xq_audio, ...
                               Yt)
% Returns the predicted rankings, given lyric and audio features.
%
% Usage:
%
%   RANKS = PREDICT_GENRE(XT_LYRICS, YT_LYRICS, XQ_LYRICS, ...
%                         XT_AUDIO, YT_AUDIO, XQ_AUDIO);
%
% This is the function that we will use for checkpoint evaluations and the
% final test. It takes a set of lyric and audio features and produces a
% ranking matrix as explained in the project overview. 
%
% This function SHOULD NOT DO ANY TRAINING. This code needs to run in under
% 5 minutes. Therefore, you should train your model BEFORE submission, save
% it in a .mat file, and load it here.

% YOUR CODE GOES HERE
% THIS IS JUST AN EXAMPLE
example =0;
dt =1;
addpath('.\dt');
if(example),
    N = size(Xq_lyrics, 1);
    scores = zeros(N, 5);
    Xt = bsxfun(@rdivide, Xt_lyrics, sum(Xt_lyrics, 2));
    Xq = bsxfun(@rdivide, Xq_lyrics, sum(Xq_lyrics, 2));

    D = Xq*Xt';
    [~, idx] = max(D, [], 2);
    ynn = idx(:, 1);
    yhat = Yt(ynn);

    for i=1:N
        scores(i, yhat(i)) = 1;
    end
    ranks = get_ranks(scores);
end
    nan_count=0;
    load('nb.mat');
    X= Xq_lyrics(:,feature_selected);
    X= full(X);
    X= [X Xq_audio];
    Yhat = nb.predict(X);
    ranks = zeros(size(Yhat,1),10);
    rank = [5 4 2 8 7 6 3 1 9 10];
    for i=1:size(Yhat,1),
        cur = rank;
        if(~isnan(Yhat(i)))
            cur(cur==Yhat(i))=[];
            cur =[Yhat(i) cur];
        else,
            nan_count=nan_count+1;
        end
        ranks(i,:)=cur;
    end
    nan_count
end
