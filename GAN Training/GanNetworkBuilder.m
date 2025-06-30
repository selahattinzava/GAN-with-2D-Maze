classdef GanNetworkBuilder
    methods(Static)
        
        function gnet = buildGenerator()
            gnet = dlnetwork;

            % Point input
            pointInputg = [imageInputLayer([64 64 3])
                           convolution2dLayer([4 4],16,"Padding", "same")
                           batchNormalizationLayer
                           reluLayer('Name','relupointInputg')];
            gnet = addLayers(gnet, pointInputg);

            % Map input
            mapInputg = [imageInputLayer([64 64 3])
                         convolution2dLayer([4 4],16,"Padding", "same")
                         batchNormalizationLayer
                         reluLayer('Name','relumapInputg')];
            gnet = addLayers(gnet, mapInputg);

            % Noise input
            noiseInputg = [imageInputLayer([64 64 1])
                           convolution2dLayer([4 4],32,"Padding", "same")
                           batchNormalizationLayer
                           reluLayer('Name','relunoiseInputg')];
            gnet = addLayers(gnet, noiseInputg);

            % Concatenation layers
            concat_1 = concatenationLayer(3,2,'Name','ConcateMap-Point');
            concat_2 = concatenationLayer(3,2,'Name','ConcateNoise-MapPoint');
            gnet = addLayers(gnet, concat_1);
            gnet = addLayers(gnet, concat_2);

            gnet = connectLayers(gnet, 'relumapInputg', 'ConcateMap-Point/in1');
            gnet = connectLayers(gnet, 'relupointInputg', 'ConcateMap-Point/in2');
            gnet = connectLayers(gnet, 'relunoiseInputg', 'ConcateNoise-MapPoint/in1');
            gnet = connectLayers(gnet, 'ConcateMap-Point', 'ConcateNoise-MapPoint/in2');

            % Core network
            tempnet = [convolution2dLayer([4 4],128,"Stride", 2 ,"Padding", "same", 'Name','tempnetinput')
                       batchNormalizationLayer
                       reluLayer
                       convolution2dLayer([4 4],256,"Stride", 2 ,"Padding", "same")
                       batchNormalizationLayer
                       reluLayer
                       convolution2dLayer([4 4],512,"Stride", 2 ,"Padding", "same")
                       batchNormalizationLayer
                       dropoutLayer(0.5)
                       reluLayer
                       transposedConv2dLayer([4 4],256,"Stride", 2 ,"Cropping", "same")
                       batchNormalizationLayer
                       reluLayer
                       transposedConv2dLayer([4 4],128,"Stride", 2 ,"Cropping", "same")
                       batchNormalizationLayer
                       reluLayer
                       transposedConv2dLayer([4 4],256,"Stride", 2 ,"Cropping", "same")
                       reluLayer('Name','relutempnetoutput')];
            gnet = addLayers(gnet, tempnet);
            gnet = connectLayers(gnet, 'ConcateNoise-MapPoint','tempnetinput');

            % Concatenation with earlier features
            concat_3 = concatenationLayer(3,2,'Name','Concateb1-b7');
            gnet = addLayers(gnet, concat_3);
            gnet = connectLayers(gnet,'relutempnetoutput','Concateb1-b7/in1');
            gnet = connectLayers(gnet,'ConcateNoise-MapPoint','Concateb1-b7/in2');

            % Output block
            tempnet2 = [convolution2dLayer([4 4],32,"Padding", "same",'Name','tempnet2input')
                        batchNormalizationLayer
                        reluLayer
                        convolution2dLayer([4 4],3,"Padding", "same")
                        tanhLayer];
            gnet = addLayers(gnet, tempnet2);
            gnet = connectLayers(gnet,'Concateb1-b7','tempnet2input');
        end

        function dnet1 = buildDiscriminator1()
            dnet1 = dlnetwork;

            mapInputd = [imageInputLayer([64 64 3])
                         convolution2dLayer([4 4],32,"Stride", 2 ,"Padding", "same")
                         leakyReluLayer(0.2, 'Name','selfmapInputd_output')];

            setimg = [imageInputLayer([64 64 3])
                      convolution2dLayer([4 4],32,"Stride", 2 ,"Padding", "same")
                      leakyReluLayer(0.2, 'Name','selfsetimg_output')];

            concat_4 = concatenationLayer(3,2, 'Name','ConcateMap-Set');

            dnet1 = addLayers(dnet1, mapInputd);
            dnet1 = addLayers(dnet1, setimg);
            dnet1 = addLayers(dnet1, concat_4);

            dnet1 = connectLayers(dnet1, 'selfmapInputd_output', 'ConcateMap-Set/in1');
            dnet1 = connectLayers(dnet1, 'selfsetimg_output', 'ConcateMap-Set/in2');

            tempnet3 = [convolution2dLayer([4 4],128,"Stride", 2 ,"Padding", "same",'Name','tempnet3input')
                        batchNormalizationLayer
                        leakyReluLayer(0.2)
                        convolution2dLayer([4 4],256,"Stride", 2 ,"Padding", "same")
                        leakyReluLayer(0.2)
                        convolution2dLayer([4 4],512,"Stride", 2 ,"Padding", "same")
                        leakyReluLayer(0.2)
                        sigmoidLayer];
            dnet1 = addLayers(dnet1, tempnet3);
            dnet1 = connectLayers(dnet1, 'ConcateMap-Set', 'tempnet3input');
        end

        function dnet2 = buildDiscriminator2()
            dnet2 = dlnetwork;

            pointInputd = [imageInputLayer([64 64 3])
                           convolution2dLayer([4 4],48,"Stride", 2 ,"Padding", "same")
                           leakyReluLayer(0.2, 'Name','pointInputd_output')];

            setimg1 = [imageInputLayer([64 64 3])
                       convolution2dLayer([4 4],16,"Stride", 2 ,"Padding", "same")
                       leakyReluLayer(0.2, 'Name','setimg1_output')];

            concat_5 = concatenationLayer(3,2, 'Name','ConcatePoint-Set');

            dnet2 = addLayers(dnet2, pointInputd);
            dnet2 = addLayers(dnet2, setimg1);
            dnet2 = addLayers(dnet2, concat_5);

            dnet2 = connectLayers(dnet2, 'pointInputd_output', 'ConcatePoint-Set/in1');
            dnet2 = connectLayers(dnet2, 'setimg1_output', 'ConcatePoint-Set/in2');

            tempnet4 = [convolution2dLayer([4 4],128,"Stride", 2 ,"Padding", "same",'Name','tempnet4_input')
                        batchNormalizationLayer
                        leakyReluLayer(0.2)
                        convolution2dLayer([4 4],256,"Stride", 2 ,"Padding", "same")
                        leakyReluLayer(0.2)
                        convolution2dLayer([4 4],512,"Stride", 2 ,"Padding", "same")
                        leakyReluLayer(0.2)
                        sigmoidLayer];
            dnet2 = addLayers(dnet2, tempnet4);
            dnet2 = connectLayers(dnet2, 'ConcatePoint-Set', 'tempnet4_input');
        end
    end
end
