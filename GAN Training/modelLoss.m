function [lossG, lossD, gradientsG, gradientsD1, gradientsD2, stateG, scoreG, scoreD1, scoreD2] = ...
    modelLoss(netG, netD1, netD2, map, point, path, noise)
    epsilon = 0.00001; % for avoiding log(0) syntax
    
    % Dynamic weight coeffient hyperparameter
    k = 3;
    
    % L1 loss weight
    lambda = 100;
    
    % Discriminitor output for real samples
    probRealD1 = forward(netD1, map, path);     
    probRealD2 = forward(netD2, point, path);  
    
    % Generator output
    [generatedPath, stateG] = forward(netG, map, point, noise);
    
    % Discriminator output for generated samples
    probFakeD1 = forward(netD1, map, generatedPath);   
    probFakeD2 = forward(netD2, point, generatedPath);
    
    % Sigmoid Cross-Entropy Loss 
    sigmoidCrossEntropy = @(pred, label) ...
        -mean(label .* log(pred + epsilon) + (1 - label) .* log(1 - pred + epsilon), "all");
    
    % Discriminator Losses 
    lossD1 = sigmoidCrossEntropy(probFakeD1, 0) + sigmoidCrossEntropy(probRealD1, 1);
    lossD2 = sigmoidCrossEntropy(probFakeD2, 0) + sigmoidCrossEntropy(probRealD2, 1);
    lossD = lossD1 + lossD2;
    
    % Dinamik β coefficients 
    beta1 = extractdata((k * lossD1) / (lossD2 + k * lossD1 + epsilon));
    beta2 = extractdata(lossD2 / (lossD2 + k * lossD1 + epsilon));
    
    % Generator Loss 
    lossAdvG = beta1 * sigmoidCrossEntropy(probFakeD1, 1) + ...
               beta2 * sigmoidCrossEntropy(probFakeD2, 1);
    
    l1Loss = mean(abs(path - generatedPath), "all");
    
    lossG = lossAdvG + lambda * l1Loss;
    
    % Scores
    scoreG  = (mean(probFakeD1, "all") + mean(probFakeD2, "all")) / 2;
    scoreD1 = (mean(probRealD1, "all") + mean(1 - probFakeD1, "all")) / 2;
    scoreD2 = (mean(probRealD2, "all") + mean(1 - probFakeD2, "all")) / 2;
    
    % Gradients
    gradientsG  = dlgradient(lossG,  netG.Learnables, RetainData=true);
    gradientsD1 = dlgradient(lossD1, netD1.Learnables);
    gradientsD2 = dlgradient(lossD2, netD2.Learnables);

end