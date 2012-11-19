clear;
load ../data/music_dataset.mat

[Xt_lyrics] = make_lyrics_sparse(train, vocab);
[Xq_lyrics] = make_lyrics_sparse(quiz, vocab);


Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

Xt_audio = make_audio(train);
%Xq_audio = make_audio(quiz);
train=1;

feature_selected=(sum(Xq_lyrics)>=10);
%size(feature_selected)
X= Xt_lyrics(:,feature_selected);
dist_array=repmat({'mvmn'},1,size(X,2)+30);
for ii=(size(X,2)+1): (size(X,2)+30),
    dist_array{ii}='normal';
end
X=full(X);
%X=tfidf(X);

X= [X Xt_audio];
size(X)
Xt =Xq_lyrics(:,feature_selected);
train_error=[];
train_loss=[];
run =1;
if run,
    nb=NaiveBayes.fit(X, Yt,'Distribution',dist_array);
    save('nb.mat','nb','feature_selected');
    if(train)
        Yhat = nb.predict(X);
    else
        Yhat = nb.predict(Xt);
    end
    
    
    ranks = zeros(size(Yhat,1),10);
    rank = [5 4 2 8 7 6 3 1 9 10];
    for i=1:size(Yhat,1),
        cur = rank;
        if(~isnan(Yhat(i)))
            cur(cur==Yhat(i))=[];
            cur =[Yhat(i) cur];
            if(train),
                index = find(cur==Yt(i));
                train_loss(i)=1-1/index;
                train_error(i)=Yt(i)~=Yhat(i);               
            end
        end
        ranks(i,:)=cur;    
    end
    mean(train_loss)
    mean(train_error)
end