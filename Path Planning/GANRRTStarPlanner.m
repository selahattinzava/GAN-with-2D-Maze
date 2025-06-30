clc; clear; close all;
load("GNet.mat");
% Random maze creation suitable with trained network
map = mapMaze(5, 1, "MapSize", [10 10], "MapResolution", 2.5);
xlim = map.XWorldLimits;
ylim = map.YWorldLimits;

ss = stateSpaceSE2([xlim; ylim; -pi pi]);
sv = validatorOccupancyMap(ss, Map=map);
sv.ValidationDistance = 0.2;

[startState, goalState] = sampleStartGoal(sv, 1);
start = startState(1:3);
goal = goalState(1:3);

% binary to RGB convertion with proper sizes
xsize = 64; ysize = 64;
occupancy = getOccupancy(map);
scaledOcc = imresize(occupancy, [xsize ysize], 'nearest');
occupied = scaledOcc == 1;

rgbMap = ones(xsize, ysize, 3);
for c = 1:3
    ch = rgbMap(:,:,c);
    ch(occupied) = 0;
    rgbMap(:,:,c) = ch;
end
rgbMap_uint8 = uint8(rgbMap * 255);

% Adding inital points to image
rgbPoint = ones(xsize, ysize, 3);
scaleX = xsize / diff(xlim);
scaleY = ysize / diff(ylim);
xStart = floor((start(1) - xlim(1)) * scaleX + 0.5);
yStart = ysize - floor((start(2) - ylim(1)) * scaleY + 0.5) + 1;
xGoal = floor((goal(1) - xlim(1)) * scaleX + 0.5);
yGoal = ysize - floor((goal(2) - ylim(1)) * scaleY + 0.5) + 1;

rgbPoint = drawSquareSafe(rgbPoint, occupied, xStart, yStart, [0, 1, 0]);
rgbPoint = drawSquareSafe(rgbPoint, occupied, xGoal, yGoal, [1, 0, 0]);
rgbPoint_uint8 = uint8(rgbPoint * 255);

% Generator Output
dlMap = dlarray(rescale(im2single(rgbMap_uint8), -1, 1), "SSCB");
dlPoint = dlarray(rescale(im2single(rgbPoint_uint8), -1, 1), "SSCB");
dlNoise = dlarray(rescale(randn([64 64 1], 'single'), -1, 1), "SSCB");
if canUseGPU
    dlMap = gpuArray(dlMap);
    dlPoint = gpuArray(dlPoint);
    dlNoise = gpuArray(dlNoise);
end
output = predict(gnet, dlMap, dlPoint, dlNoise);
genImg = gather(extractdata(output));
genImg = (genImg + 1) / 2;

% Mask promising region from created image
blueMask = (genImg(:,:,3) > 0.7) & (genImg(:,:,1) < 0.3) & (genImg(:,:,2) < 0.3);
[pixelY, pixelX] = find(blueMask);

XH_dense = zeros(length(pixelX), 2);
for i = 1:length(pixelX)
    XH_dense(i,1) = xlim(1) + (pixelX(i) - 0.5) / scaleX;
    XH_dense(i,2) = ylim(1) + (ysize - pixelY(i) + 0.5) / scaleY;
end

% Subsampling with kmeans
k = 10; 
if size(XH_dense,1) > k
    [~, XH] = kmeans(XH_dense, k, 'MaxIter', 100, 'Replicates', 3);
else
    XH = XH_dense;
end

% RRT* planner with GAN sampler
sampler = stateSamplerHeuristicGAN(ss, XH, 0.9); % µ = 0.9
planner = plannerRRTStar(ss, sv);
planner.StateSampler = sampler;
planner.MaxConnectionDistance = 1;
planner.MaxIterations = 2000;
planner.ContinueAfterGoalReached = false;

% Planning
fprintf("Planning with GAN + uniform sampling...\n");
tic
[pthObjGAN, solnInfoGAN] = plan(planner, start, goal);
toc

% Plots
figure; imshow(genImg); title('GAN Output (Promising Region)');
figure; show(map); hold on;
plot(solnInfoGAN.TreeData(:,1), solnInfoGAN.TreeData(:,2), '.-k');
plot(start(1), start(2), 'go', 'MarkerSize', 8, 'LineWidth', 2);
plot(goal(1), goal(2), 'ro', 'MarkerSize', 8, 'LineWidth', 2);
plot(pthObjGAN.States(:,1),pthObjGAN.States(:,2),'r-','LineWidth',2);
title('Heuristic RRT* using GAN-based Sampling');
fprintf("Total Heuristics RRT* Path Nodes: %d\n", size(pthObjGAN.States, 1));
fprintf("Heuristics RRT* Tree Node Count: %d\n\n", size(solnInfoGAN.TreeData, 1));

%% uavWaypointFollower
states = pthObjGAN.States;           % [x y theta]
waypoints2D = states(:,1:2);      % [x y]
yawAngles = states(:,3);          % theta (yaw)

waypoints = [waypoints2D, zeros(size(waypoints2D,1),1)];  % z = 0

wpFollowerObj = uavWaypointFollower("UAVType","multirotor","StartFrom","first","Waypoints",waypoints,"YawAngles",yawAngles,"TransitionRadius",0.5,'MinLookaheadDistance',1);

dt = 0.1;
tVec = 0:dt:50;
lookaheadDist = 2.0;

pose = zeros(3, numel(tVec));  % [x; y; yaw]
pose(:,1) = [start(1); start(2); yawAngles(1)];

for k = 2:numel(tVec)
    currentPose = pose(:,k-1);  % [x; y; yaw]

    x = currentPose(1);
    y = currentPose(2);
    yaw = currentPose(3);
    inputPose = [x; y; 0; yaw];  % [x; y; z=0; yaw]

    [lookaheadPoint, ~, desiredYaw, ~, ~, status] = ...
        wpFollowerObj(inputPose, lookaheadDist);

    direction = lookaheadPoint(1:2) - [x; y];
    directionNorm = norm(direction);
    if directionNorm > 0.05
        velocity = 1.5 * direction / directionNorm;  % 1×2 vector
    else
        velocity = [0 0];
    end

    pose(1:2,k) = pose(1:2,k-1) + dt * velocity(:);  % velocity(:) = sütun vektör
    pose(3,k) = desiredYaw;

    if status == 1
        pose(:,k+1:end) = [];
        tVec(k+1:end) = [];
        break;
    end
end

figure; hold on; grid on;
plot(waypoints(:,1), waypoints(:,2), 'ko--', 'DisplayName', 'Waypoints');
plot(pose(1,:), pose(2,:), 'r-', 'LineWidth', 2, 'DisplayName', 'UAV Path');
xlabel('X (m)'); ylabel('Y (m)');
title('2D UAV Waypoint Follower');
axis equal;
legend;
