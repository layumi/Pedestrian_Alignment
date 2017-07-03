function train_id_net_vgg16(varargin)
% -------------------------------------------------------------------------
% Part 4.1: prepare the datagpu_c
% -------------------------------------------------------------------------
addpath('../matconvnet_itchat');
% Load character dataset
imdb = load('./url_data.mat') ;
imdb = imdb.imdb;
%imdb.images.set(1:10000) = 3;
% -------------------------------------------------------------------------
% Part 4.2: initialize a CNN architecture
% -------------------------------------------------------------------------
net = resnet52_market_stn_alignment();
net.conserveMemory = true;
net.meta.normalization.averageImage = reshape([105.6920,99.1345,97.9152],1,1,3);
% -------------------------------------------------------------------------
% Part 4.3: train and evaluate the CNN
% -------------------------------------------------------------------------
opts.train.averageImage = net.meta.normalization.averageImage;
opts.train.batchSize = 32;
opts.train.continue = true ; 
opts.train.gpus = [2,4];
opts.train.prefetch = false ;
opts.train.expDir = './data/resnet52_stn_align_baseline_initial0.8_drop_0.9_1e-5_batch32' ;
opts.train.derOutputs = {'objective', 0,'objective_local',1};
opts.train.gamma = 0.9;
opts.train.momentum = 0.9;
opts.train.nesterovUpdate = true;
%opts.train.start_your_dancing = 3;
%opts.train.constraint = 5;
opts.train.learningRate = [0.1*ones(1,30),0.01*ones(1,10)] ;
opts.train.weightDecay = 0.0005;
opts.train.numEpochs = numel(opts.train.learningRate) ;
[opts, ~] = vl_argparse(opts.train, varargin) ;

% Call training function in MatConvNet
[net,info] = cnn_train_dag2(net, imdb, @getBatch,opts) ;
zzd_email(opts.expDir);
% --------------------------------------------------------------------
function inputs = getBatch(imdb,batch,opts)
% --------------------------------------------------------------------
im_url = imdb.images.data(batch) ; 
im = vl_imreadjpeg(im_url,'Pack','Resize',[224,224],'Flip',...
    'CropLocation','random','CropSize',[0.875,1],...
    'Interpolation', 'bicubic','NumThreads',8);
labels = imdb.images.label(batch) ;
oim = bsxfun(@minus,im{1},opts.averageImage);
inputs = {'data',gpuArray(oim),'label',labels,'label_local',labels};
