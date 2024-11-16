function qSolutions_deg = gikmaze_centered_auto_gik_tcp()

% 1. Load and crop maze
full_img = imread('maze3.jpg');
gray_full = rgb2gray(full_img);
bw_mask = imbinarize(gray_full);
props = regionprops(bw_mask, 'BoundingBox', 'Area');
[~, largest_idx] = max([props.Area]);
bbox = round(props(largest_idx).BoundingBox);
cropped = imcrop(full_img, bbox);
cropped = imrotate(cropped, 180);  % Align
img_orig = cropped;

% 2. Backend version for masking red and green
hsv = rgb2hsv(img_orig);
hue = hsv(:,:,1);
sat = hsv(:,:,2);
val = hsv(:,:,3);
greenMask = hue > 0.25 & hue < 0.45 & sat > 0.2 & val > 0.2;
redMask1 = hue < 0.05 & sat > 0.2 & val > 0.2;
redMask2 = hue > 0.9 & sat > 0.2 & val > 0.2;
redMask = redMask1 | redMask2;
colorMask = imdilate(greenMask | redMask, strel('disk', 8));
img_proc = img_orig;
img_proc(repmat(colorMask, [1 1 3])) = 255;

% 3. Preprocess backend image
gray = rgb2gray(img_proc);
bw = imbinarize(gray, 'adaptive');
bw = ~bw;
distMap = bwdist(~bw);
safeMask = distMap > 2;

% 4. Click start and goal
figure, imshow(img_orig); title('Click START (green) and GOAL (red)');
[x, y] = ginput(2);
start = round([x(1), y(1)]);
goal = round([x(2), y(2)]);
close;

[start, snapped1] = snapToSafe(start, safeMask);
[goal, snapped2] = snapToSafe(goal, safeMask);
if snapped1, disp('⚠️ Start snapped'); end
if snapped2, disp('⚠️ Goal snapped'); end

% 5. Pathfinding
[prev, dist, path] = dijkstraPath(safeMask, distMap, start, goal);
if isempty(path), error('❌ No path found!'); end
path = smoothdata(path, 1, 'gaussian', 4);
path_30 = interparc(15, double(path(:,1)), double(path(:,2)), 'linear');

% 6. Transform to robot coords
scale_factor = 14 / size(img_orig, 1);
real_coords = path_30 * scale_factor;
image_pts = [0 14; 0 0; 14 0; 14 14]; % cm
robot_pts = [376.476 325.093;
             376.808 185.534;
             233.237 185.707;
             233.117 332.678]; % mm
tform = fitgeotrans(image_pts, robot_pts, 'projective');
robot_xy = transformPointsForward(tform, real_coords);
z_val = 130;
path_mm = [robot_xy, repmat(z_val, size(robot_xy,1), 1)];

% 7. Extend with entry/exit and Z=211 transitions
first_point = path_mm(1,:); first_dup = first_point; first_dup(3) = 211;
last_point = path_mm(end,:); last_dup = last_point; last_dup(3) = 211;
const_point = [383, 16, 211];
path_mm = [const_point;
           first_dup;
           first_point;
           path_mm(2:end-1,:);
           last_point;
           last_dup;
           const_point];

% ✅ Visualize
figure, imshow(img_orig); hold on;
plot(path(:,1), path(:,2), 'c-', 'LineWidth', 2);
plot(start(1), start(2), 'ro', 'MarkerFaceColor', 'r');
plot(goal(1), goal(2), 'go', 'MarkerFaceColor', 'g');
plot(path_30(:,1), path_30(:,2), 'bo', 'MarkerFaceColor', 'b');
title('✅ Final Path with Entry/Exit');

% 8. GIK solve
robot = importrobot('myCobot28.urdf', 'MeshPath', "meshes", 'DataFormat', 'row');
gik = generalizedInverseKinematics('RigidBodyTree', robot, 'ConstraintInputs', {'position','orientation'});
posTgt = constraintPositionTarget('Link6'); posTgt.Weights = 1;
orientTgt = constraintOrientationTarget('Link6'); orientTgt.TargetOrientation = axang2quat([0 1 0 pi]); orientTgt.Weights = 1;

qSolutions_deg = zeros(size(path_mm, 1), numel(robot.homeConfiguration));
initialGuess = robot.homeConfiguration;

for i = 1:size(path_mm,1)
    posTgt.TargetPosition = path_mm(i,:) / 1000;
    [configSol, solInfo] = gik(initialGuess, posTgt, orientTgt);
    if strcmp(solInfo.Status, 'success')
        qDeg = rad2deg(configSol);
        qDeg(2) = qDeg(2) - 90; qDeg(4) = qDeg(4) - 90;
        qSolutions_deg(i,:) = qDeg;
        initialGuess = configSol;
    else
        warning("GIK failed for point %d", i);
    end
    figure(2); clf;
    show(robot, configSol, 'Visuals', 'on', 'PreservePlot', false);
    title(['Simulated Robot Pose at Point ', num2str(i)]);
    drawnow; pause(0.2);
end

% 9. TCP Send
server_ip = '192.168.1.159';
server_port = 5001;
for i = 1:size(qSolutions_deg, 1)
    angles = qSolutions_deg(i, :);
    msg = sprintf('set_angles(%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,100)', angles);
    fprintf('📤 Sending [%02d]: %s\n', i, msg);
    send_tcp_packet(server_ip, server_port, msg);
    pause(1);
end

disp('✅ Completed full path planning, simulation, and execution.');

end

% ----------------- Helper Functions -----------------

function [pt_new, snapped] = snapToSafe(pt, mask)
    if mask(pt(2), pt(1))
        pt_new = pt; snapped = false;
    else
        [yy, xx] = find(mask);
        [~, idx] = min((xx - pt(1)).^2 + (yy - pt(2)).^2);
        pt_new = [xx(idx), yy(idx)];
        snapped = true;
    end
end

function [prev, dist, path] = dijkstraPath(mask, distMap, start, goal)
    sz = size(mask); dist = inf(sz); visited = false(sz);
    prev = zeros([sz 2], 'int16');
    dirs = [0 1; 1 0; 0 -1; -1 0];
    dist(start(2), start(1)) = 0;
    queue = [start, 0];
    while ~isempty(queue)
        [~, i] = min(queue(:,3));
        current = queue(i,1:2); queue(i,:) = [];
        x = current(1); y = current(2);
        if visited(y,x), continue; end
        visited(y,x) = true;
        if all(current == goal), break; end
        for d = 1:4
            nx = x + dirs(d,1); ny = y + dirs(d,2);
            if nx > 0 && ny > 0 && nx <= sz(2) && ny <= sz(1)
                if mask(ny,nx)
                    cost = dist(y,x) + 1/(1 + distMap(ny,nx));
                    if cost < dist(ny,nx)
                        dist(ny,nx) = cost;
                        prev(ny,nx,:) = [x y];
                        queue(end+1,:) = [nx ny cost];
                    end
                end
            end
        end
    end
    if all(prev(goal(2), goal(1), :) == 0)
        path = [];
    else
        path = goal;
        while ~isequal(path(1,:), start)
            p = squeeze(prev(path(1,2), path(1,1), :))';
            path = [p; path];
        end
    end
end

function send_tcp_packet(ip, port, message)
    try
        t = tcpclient(ip, port, 'Timeout', 3);
        write(t, uint8(message));
        clear t;
    catch err
        fprintf('⚠️ TCP Send Error: %s\n', err.message);
    end
end