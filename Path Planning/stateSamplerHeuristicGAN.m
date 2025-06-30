classdef stateSamplerHeuristicGAN < nav.StateSampler & ...
        matlabshared.planning.internal.EnforceScalarHandle & ...
        nav.algs.internal.InternalAccess

    properties
        XH              % Promising region points (Nx2)
        Mu = 0.9        % Heuristic sampling ratio
    end

    methods
        function obj = stateSamplerHeuristicGAN(stateSpace, XH, Mu)
            arguments
                stateSpace (1,1) nav.StateSpace
                XH (:,2) double
                Mu (1,1) double {mustBeInRange(Mu,0,1)} = 0.9
            end
            obj@nav.StateSampler(stateSpace);
            obj.XH = XH;
            obj.Mu = Mu;
        end

        function states = sample(obj, numSamples)
            arguments
                obj
                numSamples (1,1) {mustBePositive, mustBeInteger} = 1
            end

            states = zeros(numSamples, obj.StateSpace.NumStateVariables);

            for i = 1:numSamples
                if rand < obj.Mu && ~isempty(obj.XH)
                    idx = randi(size(obj.XH, 1));
                    x = obj.XH(idx, 1);
                    y = obj.XH(idx, 2);
                    theta = rand * 2 * pi - pi;
                    states(i,:) = [x, y, theta];
                else
                    states(i,:) = obj.StateSpace.sampleUniform();
                end
            end
        end

        function copyObj = copy(obj)
            copyObj = stateSamplerHeuristicGAN(obj.StateSpace, obj.XH, obj.Mu);
        end
    end
end
