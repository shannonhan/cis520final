clear;
addPath('./libsvm');
load ../data/music_dataset.mat

param = [0.0001 4];


[Xt_lyrics] = make_lyrics_sparse(train, vocab);
[Xq_lyrics] = make_lyrics_sparse(quiz, vocab);


Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

Xt_audio = make_audio(train);
Xq_audio = make_audio(quiz);
train=1;

cols = feature_selection_lyrics(Xt_lyrics, Yt, Xq_lyrics,vocab, param(1), param(2));
X=Xt_lyrics(:,cols);
X=tfidf(X);
size(X)
Y = Yt;

Xq = Xq_lyrics(:, cols);
Xq = tfidf(Xq);
Yq = zeros(size(Xq, 1),1);


k = @(x,x2) kernel_intersection(x, x2);

Kxq = k(X, Xq);



[test_err info]=kernel_libsvm(X, Y, X, Y,k,0);


save('svm.mat', 'cols', 'Kxq', 'info');