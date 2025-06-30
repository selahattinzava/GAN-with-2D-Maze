function [X, Y, Z, W] = preprocessMiniBatch(Map, Path, Point, Noise)
    Map   = cellfun(@(x) x{1}, Map,   'UniformOutput', false);
    Path  = cellfun(@(x) x{1}, Path,  'UniformOutput', false);
    Point = cellfun(@(x) x{1}, Point, 'UniformOutput', false);
    Noise = cellfun(@(x) x{1}, Noise, 'UniformOutput', false);
    
    X = cat(4, Map{:});
    Y = cat(4, Path{:});
    Z = cat(4, Point{:});
    W = cat(4, Noise{:});
    
    X = rescale(im2single(X), -1, 1);
    Y = rescale(im2single(Y), -1, 1);
    Z = rescale(im2single(Z), -1, 1);
    W = rescale(im2single(W), -1, 1);
    
    X = dlarray(X, "SSCB");
    Y = dlarray(Y, "SSCB");
    Z = dlarray(Z, "SSCB");
    W = dlarray(W, "SSCB");
    
    if canUseGPU
        X = gpuArray(X);
        Y = gpuArray(Y);
        Z = gpuArray(Z);
        W = gpuArray(W);
    end
end