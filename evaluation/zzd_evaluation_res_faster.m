clc;clear all;close all;
addpath ../test
addpath ..
addpath CM_Curve
rank_size = 2000;

%% add necessary paths
query_dir = '/data/uts511/reid/market1501/query/';% query directory
test_dir = '/data/uts511/reid/market1501/bounding_box_test/';% database directory
alpha = 0.5;

%% calculate query features
Hist_query = importdata('../test/resnet_query.mat')';
%Hist_query = [];
align_query = importdata('../test/resnet_query_align.mat')';
Hist_query = cat(1,Hist_query*alpha,align_query*(1-alpha));
nQuery = size(Hist_query, 2);

%% calculate database features
Hist_test = importdata('../test/resnet_gallery.mat')';
align_test = importdata('../test/resnet_gallery_align.mat')';
Hist_test = cat(1,Hist_test*alpha,align_test*(1-alpha));
nTest = size(Hist_test, 2);

%% calculate the ID and camera for database images
test_files = dir([test_dir '*.jpg']);
testID = zeros(length(test_files), 1);
testCAM = zeros(length(test_files), 1);
if ~exist('data/testID.mat')
    for n = 1:length(test_files)
        img_name = test_files(n).name;
        if strcmp(img_name(1), '-') % junk images
            testID(n) = -1;
            testCAM(n) = str2num(img_name(5));
        else
            %img_name
            testID(n) = str2num(img_name(1:4));
            testCAM(n) = str2num(img_name(7));
        end
    end
    save('data/testID.mat', 'testID');
    save('data/testCAM.mat', 'testCAM');
else
    testID = importdata('data/testID.mat');
    testCAM = importdata('data/testCAM.mat');
end

%% calculate the ID and camera for query images
query_files = dir([query_dir '*.jpg']);
queryID = zeros(length(query_files), 1);
queryCAM = zeros(length(query_files), 1);
if ~exist('data/queryID.mat')
    for n = 1:length(query_files)
        img_name = query_files(n).name;
        if strcmp(img_name(1), '-') % junk images
            queryID(n) = -1;
            queryCAM(n) = str2num(img_name(5));
        else
            queryID(n) = str2num(img_name(1:4));
            queryCAM(n) = str2num(img_name(7));
        end
    end
    save('data/queryID.mat', 'queryID');
    save('data/queryCAM.mat', 'queryCAM');
else
    queryID = importdata('data/queryID.mat');
    queryCAM = importdata('data/queryCAM.mat');
end

%% search the database and calcuate re-id accuracy
ap = zeros(nQuery, 1); % average precision
ap_max_rerank  = zeros(nQuery, 1); % average precision with MultiQ_max + re-ranking
ap_pairwise = zeros(nQuery, 6); % pairwise average precision with single query (see Fig. 7 in the paper)

CMC = zeros(nQuery, rank_size);
CMC_max_rerank = zeros(nQuery, rank_size);

r1 = 0; % rank 1 precision with single query
r1_max_rerank = 0; % rank 1 precision with MultiQ_max + re-ranking
r1_pairwise = zeros(nQuery, 6);% pairwise rank 1 precision with single query (see Fig. 7 in the paper)

dist = sqdist(Hist_test, Hist_query); % distance calculate with single query. Note that Euclidean distance is equivalent to cosine distance if vectors are l2-normalized
%dist_cos_max = (2-dist_max)./2; % cosine distance with MultiQ_max, used for re-ranking

knn = 1; % number of expanded queries. knn = 1 yields best result
%queryCam = importdata('data/queryCAM_duke.mat'); % camera ID for each query
%testCam = importdata('data/testCAM_duke.mat'); % camera ID for each database image

for k = 1:nQuery
    % load ground truth for each query (good and junk)
    good_index = intersect(find(testID == queryID(k)), find(testCAM ~= queryCAM(k)))';% images with the same ID but different camera from the query
    junk_index1 = find(testID == -1);% images neither good nor bad in terms of bbox quality
    junk_index2 = intersect(find(testID == queryID(k)), find(testCAM == queryCAM(k))); % images with the same ID and the same camera as the query
    junk_index = [junk_index1; junk_index2]';
    score = dist(:, k);
    %score_avg = dist_avg(:, k);
    %score_max = dist_max(:, k);
    
    % sort database images according Euclidean distance
    [~, index] = sort(score, 'ascend');  % single query
    
    % re-rank  select rank_size=1000 index
    
    index = index(1:rank_size);
    %{
    retr = index;
    for kk = 1:numel(junk_index)
        retr(retr==junk_index(kk))=[];
    end
    duke_index(k,:) = retr(1:10);
    %}
    [ap(k), CMC(k, :)] = compute_AP_rerank(good_index, junk_index, index);% compute AP for single query
    %ap_pairwise(k, :) = compute_AP_multiCam(good_index, junk_index1,junk_index2, index, queryCAM(k), testCAM); % compute pairwise AP for single query
    %fprintf('%d::%f\n',k,ap(k));
    %%%%%%%%%%% calculate pairwise r1 precision %%%%%%%%%%%%%%%%%%%%
    %r1_pairwise(k, :) = compute_r1_multiCam(good_index, junk_index1,junk_index2, index, queryCAM(k), testCAM); % pairwise rank 1 precision with single query
    %%%%%%%%%%%%%% calculate r1 precision %%%%%%%%%%%%%%%%%%%%
end
CMC = mean(CMC);
%% print result
fprintf('alpha: %f single query: mAP = %f, r1 precision = %f\r\n', alpha,mean(ap), CMC(1));
