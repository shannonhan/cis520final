clear;
load ../data/music_dataset.mat
addPath('dt');
[Xt_lyrics] = make_lyrics_sparse(train, vocab);
%[Xq_lyrics] = make_lyrics_sparse(quiz, vocab);


Yt = zeros(numel(train), 1);
for i=1:numel(train)
    Yt(i) = genre_class(train(i).genre);
end

%Xt_audio = make_audio(train);
%Xq_audio = make_audio(quiz);
nodes ={};

X= Xt_lyrics(:,(sum(Xt_lyrics)>=250));

test=1;
if test,                                                                                                                  
    train_e = zeros(1, 10);
    train_l = zeros(1, 10);
    for depth = 4:8,
        node =  dt_train_multi(Xt_lyrics, Yt, depth);
        nodes{i-3}=node;
        loss = zeros(size(Yt));
        error = zeros(size(Yt));
        for i = 1:size(Yt),
            line = Xt_lyrics(i, :);
      
            p =  dt_value(node, line); % p is a row vector
            if(size(p,2)~=10)
                continue
            end
            
            [~, rank] = sort(p,2);
            rindex = find(rank==Yt(i));
            loss(i) =  1-1/rindex;
            [~, maxi] = max(p);
            error(i) =  maxi~= Yt(i);     
            
        end
        train_l(depth)= mean(loss)
        train_e(depth) = mean(error)
    end
end