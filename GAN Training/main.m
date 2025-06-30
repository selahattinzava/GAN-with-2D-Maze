%% Training
clc; reset(gpuDevice)
numEpochs = 10;
miniBatchSize = 8;
ds = load("dataset.mat");
gnet  = GanNetworkBuilder.buildGenerator();
dnet1 = GanNetworkBuilder.buildDiscriminator1();
dnet2 = GanNetworkBuilder.buildDiscriminator2();              
[gnet, dnet1, dnet2] = trainGan(ds, gnet, dnet1, dnet2, numEpochs, miniBatchSize);

%% Create Image & Plot With Trained Network
load("GNet.mat");
n=3;
figure
for i=1:n
    map = im2single(ds.Map{i*11000});
    point = im2single(ds.Point{i*11000});
    noise = im2single(ds.Noise{i*18000});
    
    subplot(2, n, i ) %2*i - 1
    imshow(map)
    map = rescale(map, -1, 1);
    point = rescale(point, -1, 1);
    noise = rescale(noise, -1, 1);
    map = dlarray(map, "SSCB");
    point = dlarray(point, "SSCB");
    noise = dlarray(noise, "SSCB");
    if canUseGPU
        map = gpuArray(map);
        point = gpuArray(point);
        noise = gpuArray(noise);
    end  
    generatedPath = predict(gnet, map, point, noise);
    generatedImg = extractdata(generatedPath);
    generatedImg = rescale(generatedImg, 0, 1);  

    subplot(2, n, i+n ) %2*i
    imshow(generatedImg(:,:,:,1))  
    title("Generated Path Output")
end
