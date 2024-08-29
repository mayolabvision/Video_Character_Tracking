clear
clc

% Define the directory path
PATH = '/Users/kendranoneman/Projects/mayo/video_image_matcher/tests/characterTracking_finalRepo';

% Get a list of all .mat files in the directory
mat_files = dir(fullfile(PATH, '*.mat'));
% Extract the names of the .mat files
file_names = {mat_files.name};
% Sort the file names alphabetically
[sorted_file_names, sort_index] = sort(file_names);
% Rearrange the mat_files structure array based on the sorted indices
mat_files = mat_files(sort_index);

% Initialize an empty table to concatenate all individual tables
contour_table = table();

% Loop over each .mat file
for k = 1:length(mat_files)
    % Load the .mat file
    load(fullfile(PATH, mat_files(k).name));

    % Ensure video_name and character_name are categorical
    video_name = categorical(reshape(video_name, [], 1));
    character_name = categorical(reshape(character_name, [], 1));
    
    % Reshape time_points to be a column vector and convert to seconds
    time_points = reshape(time_points, [], 1) / 1000;

    % Convert contours to a cell array of 50x2 matrices
    num_frames = size(contours, 1);  % Get the number of frames (usually 361)
    contours_cell = cell(num_frames, 1);
    for i = 1:num_frames
        contours_cell{i} = squeeze(contours(i, :, :));
    end

    % Create the table for the current file
    T = table(video_name, character_name, time_points, contours_cell, ...
              'VariableNames', {'video_name', 'character_name', 'time_alignedVideoStart_sec', 'contours_xy'});

    % Concatenate the current table with the master table
    contour_table = [contour_table; T]; % Use vertical concatenation
end

% Define the path to save the combined contours table
save_path = fullfile(PATH, 'ALL_CONTOURS.mat');

% Save the contour_table to a .mat file
save(save_path, 'contour_table');

fprintf('Contour table saved to %s\n', save_path);