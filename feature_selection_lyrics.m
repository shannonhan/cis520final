function [cols] = feature_selection_lyrics(Xt, Yt, Xq,vocab, ig_threshold, tf_threshold)
    Xall = [Xt;Xq];
    [Xall ~] = stem(Xall,vocab);
    
    load 'igs.mat';
    
    ig_filtered = igs' > ig_threshold;
    
    tf_filtered = sum(Xall) > tf_threshold;
    
    mean(ig_filtered==tf_filtered)
    
    merge_filtered = ig_filtered.* tf_filtered;
    
    cols = ~~merge_filtered;
end

    
