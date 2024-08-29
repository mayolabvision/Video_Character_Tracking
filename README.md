# README for Contour Tracking Project

## Overview

This project is designed to track specific characters within video files, extracting detailed contour data that outlines the target character's body in each frame. The contours are stored in a structured table format for further analysis and visualization. The project processes video frames, identifies the target character, and saves the contour data along with relevant metadata into a master table.

## contour_table Structure

The final output of the project is a table (`contour_table`) stored in a `.mat` file called `ALL_CONTOURS.mat`. This table is composed of several columns, each holding specific information about the video, the character, and their contours. Below is an explanation of each column in the `contour_table`:

### Columns

1. **video_name**
   - **Description:** This column contains the name of the video file (without the path and file extension) from which the contour data was extracted.
   - **Type:** Categorical
   
2. **character_name**
   - **Description:** This column contains the name of the target character (e.g., "Principal") whose contours are being tracked and stored.
   - **Type:** Categorical
   
3. **time_alignedVideoStart_sec**
   - **Description:** This column contains the time points in seconds, aligned to the start of the video. Each entry corresponds to a specific frame in the video, providing a temporal context for the contours.
   - **Type:** Numeric (Double)
   
4. **contours_xy**
   - **Description:** This column contains the contour data for the target character in each frame. The contours are represented as a cell array, where each cell contains a matrix of shape (50x2). Each matrix provides the (x, y) coordinates outlining the target character’s body in that frame. The coordinates are normalized such that (0,0) is the center of the video frame. The bounds of the frame extend from [-frame_width/2, frame_width/2] in the x-direction and [-frame_height/2, frame_height/2] in the y-direction. This means the top-right corner of the frame is at (frame_width/2, frame_height/2) and the bottom-left corner is at (-frame_width/2, -frame_height/2). If the values of the contours are NaN, it indicates that the target character was not detected in the video during that frame.
   - **Type:** Cell array of 50x2 matrices

## Explanation of Contours

The "contours" represent the (x, y) coordinates that outline the shape of the target character in each frame of the video. These contours are generated by the detection algorithm, which identifies the edges of the character's body.
For example, if the contour had only 4 points, it would simply draw a box around the character. However, to achieve a more accurate and detailed outline, we use 50 contour points. These points form a more precise boundary around the character, following the actual shape of their body as closely as possible.

To handle cases where the character was not detected in every frame, interpolation was applied between the first and last frames where the character was detected. The interpolation fills in missing contour data to maintain a smooth and continuous outline across frames.



https://github.com/user-attachments/assets/665c1bcc-f8c8-422e-8664-518ca19bc243



https://github.com/user-attachments/assets/07e9e749-c259-494f-b694-0c87253a3960




## Usage

The table stored in `ALL_CONTOURS.mat` can be loaded into MATLAB or other compatible software for analysis. You can access and manipulate the contour data to study the movement, position, and other characteristics of the target character over time.

### Example MATLAB Usage:

#### 1. Load the Table and Filter by `video_name`:
```matlab
% Load the contour table from the .mat file
load('ALL_CONTOURS.mat');  

% Filter the table for a specific video name
selected_video = 'Exp_Left_FourthDoor_High';
filtered_table = contour_table(contour_table.video_name == selected_video, :);
```

#### 2. Make some fake eye position data:
```matlab
% Concatenate all contour data into one matrix to determine xlim and ylim
all_contours = vertcat(contour_table.contours_xy{:});
mins = min(cell2mat(all_contours));
maxs = max(cell2mat(all_contours));

% Generate simulated eye positions
num_frames = height(filtered_table);
eye_positions = cell(num_frames, 1);
for i = 1:num_frames
    % Use the centroid of the contour plus some random noise as the eye position
    contour_data = cell2mat(filtered_table.contours_xy{i});
    if isnan(contour_data(1,1))
        eye_positions{i} = NaN(1, 2);
    else
        contour_centroid = mean(contour_data, 1);
        eye_positions{i} = contour_centroid + randn(1, 2) * 20; % Adding noise
    end
end
```

#### 3. Plot fake eye position and contour boundaries over time:
```matlab
% Loop through the filtered rows and plot the contours with the eye positions
figure;
hold off;
for i = 1:num_frames
    contour_data = cell2mat(filtered_table.contours_xy{i});  % Extract the (50x2) contour matrix
    
    % Check if the first value is NaN and skip if true
    if isnan(contour_data(1,1))
        continue;
    end
    
    % Plot the contour (50x2 points) for the current frame
    plot(contour_data(:,1), contour_data(:,2), '-','linewidth',2,'color','red');  % Plot the (x,y) coordinates
    hold on;
    
    % Determine if the eye position is inside or outside the contour
    is_inside = inpolygon(eye_positions{i}(1), eye_positions{i}(2), contour_data(:,1), contour_data(:,2));
    
    % Plot the eye position: red if inside, black if outside
    if is_inside
        plot(eye_positions{i}(1), eye_positions{i}(2), 'ro', 'MarkerFaceColor', 'red');  % Red if inside
    else
        plot(eye_positions{i}(1), eye_positions{i}(2), 'ko', 'MarkerFaceColor', 'black');  % Black if outside
    end
    
    % Set title and axes limits
    title(sprintf('Frame: %d, Time: %.2f sec', i, filtered_table.time_alignedVideoStart_sec(i)));
    xlim([mins(1)-1, maxs(1)+1]);
    ylim([mins(2)-1 maxs(2)+1]);
    
    % Pause for animation effect
    pause(0.1);
    hold off;
end
hold off;
```
