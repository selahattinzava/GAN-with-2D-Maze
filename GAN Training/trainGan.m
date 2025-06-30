function [gnet, dnet1, dnet2] = trainGan(ds, gnet, dnet1, dnet2, maxEpochs, miniBatchSize)
    
    learnRateG = 0.001; 
    learnRateD = 0.0000001; 
    gradientDecayFactor = 0.5;
    squaredGradientDecayFactor = 0.99;
    
    trailingAvgG = [];
    trailingAvgSqG = [];
    trailingAvgD1 = [];
    trailingAvgSqD1 = [];
    trailingAvgD2 = [];
    trailingAvgSqD2 = [];

    dsMap   = arrayDatastore(ds.Map,   'IterationDimension', 1);
    dsPath  = arrayDatastore(ds.Path,  'IterationDimension', 1);
    dsPoint = arrayDatastore(ds.Point, 'IterationDimension', 1);
    dsNoise = arrayDatastore(ds.Noise, 'IterationDimension', 1);
    cds = combine(dsMap, dsPath, dsPoint, dsNoise);
    mbq = minibatchqueue(cds, 4, ...
        'MiniBatchSize', miniBatchSize, ...
        'MiniBatchFcn', @preprocessMiniBatch, ...
        'MiniBatchFormat', {'SSCB', 'SSCB', 'SSCB', 'SSCB'}, ...
        'OutputCast', 'single', ...
        'PartialMiniBatch', 'discard', ...
        'OutputEnvironment','gpu', ...
        'PreprocessingEnvironment','parallel'); 
    
    gnet = initialize(gnet);
    dnet1 = initialize(dnet1);
    dnet2 = initialize(dnet2);
    
    monitor = trainingProgressMonitor( ...
        Metrics=["GeneratorLoss","DiscriminatorLoss","GeneratorScore","DiscriminatorScore"], ...
        Info=["GeneratorLRate","DiscriminatorLRate","Epoch","Iteration"], ...
        XLabel="Iteration");
    groupSubPlot(monitor,"Loss",["GeneratorLoss","DiscriminatorLoss"]);
    groupSubPlot(monitor, Score=["GeneratorScore","DiscriminatorScore"]);
    
    epoch = 0;
    iteration = 0;
    numIterations = floor(height(ds) / miniBatchSize);
    
    while epoch < maxEpochs && ~monitor.Stop
        epoch = epoch + 1;
        fprintf('Epoch %d/%d\n', epoch, maxEpochs);
    
        shuffle(mbq);
    
        while hasdata(mbq) && ~monitor.Stop
            iteration = iteration + 1;
            % Next Batch
            [mapBatch, pathBatch, pointBatch, noiseBatch] = next(mbq);
            % Loss Function
            [lossG, lossD, gradientsG, gradientsD1, gradientsD2, stateG, ...
             scoreG, scoreD1, scoreD2] = ...
             dlfeval(@modelLoss, gnet, dnet1, dnet2, mapBatch, pointBatch, pathBatch, noiseBatch);
            % Update GNet after every 2 iterations
            gnet.State = stateG;
            if mod(iteration, 2) == 0
                [gnet, trailingAvgG, trailingAvgSqG] = adamupdate( ...
                    gnet, gradientsG, trailingAvgG, trailingAvgSqG, iteration, ...
                    learnRateG, gradientDecayFactor, squaredGradientDecayFactor);
            end
    
            [dnet1, trailingAvgD1, trailingAvgSqD1] = adamupdate( ...
                dnet1, gradientsD1, trailingAvgD1, trailingAvgSqD1, iteration, ...
                learnRateD, gradientDecayFactor, squaredGradientDecayFactor);
    
            [dnet2, trailingAvgD2, trailingAvgSqD2] = adamupdate( ...
                dnet2, gradientsD2, trailingAvgD2, trailingAvgSqD2, iteration, ...
                learnRateD, gradientDecayFactor, squaredGradientDecayFactor);
            % Logs
            recordMetrics(monitor, iteration, ...
                GeneratorLoss=lossG, ...
                DiscriminatorLoss=lossD, ...
                GeneratorScore=gather(extractdata(scoreG)), ...
                DiscriminatorScore=gather(extractdata((scoreD1 + scoreD2) / 2)));
    
            updateInfo(monitor, ...
                GeneratorLRate= learnRateG, ...
                DiscriminatorLRate= learnRateD, ...
                Epoch=string(epoch) + " of " + string(maxEpochs), ...
                Iteration=iteration);

            monitor.Progress = 100 * iteration / (numIterations * maxEpochs);
    
            fprintf('Iter %d | G Score: %.4f | D Score: %.4f\n', ...
            iteration, extractdata(scoreG), extractdata((scoreD1 + scoreD2)/2));
        end
    end
end
