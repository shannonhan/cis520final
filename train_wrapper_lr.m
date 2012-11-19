clear;
load ../data/music_dataset.mat

[Xt_lyrics] = make_lyrics_sparse(train, vocab);
%[Xq_lyrics] = make_lyrics_sparse(quiz, vocab);


Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

Xt_audio = make_audio(train);
%Xq_audio = make_audio(quiz);
features =(sum(Xt_lyrics)>=150);
X= Xt_lyrics(:,features);
X= [X Xt_audio];
size(X)
1
B=mnrfit(X,Yt);
save('lr.mat','B','features');
2
pHatNom = mnrval(B,X);
3
mean(pHatNom~=Yt);

