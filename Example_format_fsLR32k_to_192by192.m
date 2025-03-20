addpath('./CIFTI_read_save')
clear;close all;clc;

% Set up the parameters
img_size = 192; % size of geometric-reformatted image
mask_path = './mask'; % path of the geometric reformatting transformation and medial wall mask
output_path = './data'; % path of the output data for VAE inference
data_full_path = './Example_Lynch2024_Priors/Lynch_2024_Nature_Priors.mat' % Priors.FC contains the matrix where each column is a FC profile for the network
namestr = 'Lynch2024_45subj_Prior' % book-keeping: name to give to the file
parcelname = '20NetsParcel' %book-keeping: if my data comes from a parcellation, then note what parcellation it is with number of parcels/networks so I can know how many images to expect
sampleinterval = 1; % 1/N sample rate if N = 1 then no subsample, if N = 10 means downsample by a factor of 10, etc.
randpermutation = false; % whether you want to randomize the order so that a different set of seed vertices are sampled from each of my training subjects
cifti_type = 'parcel_dconn'; % 'dtseries' or 'dconn' or 'parcel_dconn' % if it's 'dconn', I set the self-connectivity on the diagonals to 0 below (will be ignored when calculating the cost function). 
h5file_path = ['./data/',namestr,'_',parcelname,'.h5'];

if ~isfolder(output_path)
    mkdir(output_path);
    disp(['Target directory does not exist and is created: ', output_path]);
else
    disp(['Target directory exists: ', output_path]);
end

%% Load masks and transformation matrix
LeftMask = load(fullfile(mask_path, 'MSE_Mask.mat')).Regular_Grid_Left_Mask;
RightMask = load(fullfile(mask_path, 'MSE_Mask.mat')).Regular_Grid_Right_Mask;

left_transmat = load(fullfile(mask_path, 'Left_fMRI2Grid_192_by_192_NN.mat')).grid_mapping;
right_transmat = load(fullfile(mask_path, 'Right_fMRI2Grid_192_by_192_NN.mat')).grid_mapping;

%% Load data
load(data_full_path)

%% Start converting data to .h5 format
fmri_data1 = Priors.FC;
assert(size(fmri_data1,1)==59412) % check that it has the correct number of cortical vertices (has to be 59412 in fsLR32k), if not, do something before to remove non-cortical vertices
% Format data
[LeftSurfData, RightSurfData] = FormatData(fmri_data1,cifti_type,left_transmat,right_transmat,randpermutation,sampleinterval,img_size);
% Save data to h5
SaveData(LeftSurfData, RightSurfData,h5file_path)

%% Dependency Functions
% Function to format data
function [LeftSurfData, RightSurfData] = FormatData(fmri_data,cifti_type,left_transmat,right_transmat,randpermutation,sampleinterval,img_size)
    if strcmp(cifti_type, 'dconn')
        fmri_data(1:size(fmri_data,1)+1:end) = 0; % Set diagonal to zero      
    end

    if randpermutation
        idx = randperm(size(fmri_data, 2));
        idx = idx(1:sampleinterval:size(fmri_data,2));
    else
        idx = 1:sampleinterval:size(fmri_data,2);
    end

    left_data = fmri_data(1:29696, idx);
    right_data = fmri_data(29697:59412, idx);
    disp(['Loading data the size of the left hemisphere is ', num2str(size(left_data)), '; the size of the right hemisphere is ', num2str(size(right_data))]);

    % Apply transformation and reshape
    LeftSurfData = reshape(left_transmat * left_data, img_size, img_size, 1,[]);
    RightSurfData = reshape(right_transmat * right_data, img_size, img_size,1, []);

    disp(size(LeftSurfData));
    disp(size(RightSurfData));
    disp('here in format data');
end

% Function to save data
function SaveData(LeftSurfData, RightSurfData,file_path)
    disp(size(LeftSurfData));
    disp(['saving to ' file_path]);
    if isfile(file_path)
        disp('Output Data Exists. Append Data To The End');
        fileInfo = h5info(file_path,'/LeftData');
        currentSize = fileInfo.Dataspace.Size(4);
    else
            disp('Output Data Does Not Exist. Creating New Data');
            currentSize = 0;
            h5create(file_path, '/LeftData', [size(LeftSurfData,[1,2,3]),Inf], 'Datatype', 'single','ChunkSize',[size(LeftSurfData,[1,2,3]),1]);
            h5create(file_path, '/RightData', [size(RightSurfData,[1,2,3]),Inf], 'Datatype', 'single','ChunkSize',[size(RightSurfData,[1,2,3]),1]);
    end
    
    N = size(LeftSurfData,4);
    startIndex = [1,1,1,currentSize+1];
    chunkSize = [size(RightSurfData,[1,2,3]),N];
    
    h5write(file_path, '/LeftData', single(LeftSurfData),startIndex,chunkSize);
    h5write(file_path, '/RightData', single(RightSurfData),startIndex,chunkSize);

    disp('here in save_data');
end

