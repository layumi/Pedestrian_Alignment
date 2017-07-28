% In this file, we extract the feature from alignment branch
clear;
addpath ..;
netStruct = load('../data/resnet52_stn_align_baseline_initial0.8_drop0.9_1e-5_batch32/net-epoch-40.mat');
%--------add l2 norm
net = dagnn.DagNN.loadobj(netStruct.net);
net.addLayer('lrn_test',dagnn.LRN('param',[4096,0,1,0.5]),{'pool5_local'},{'pool5n_local'},{});
clear netStruct;
net.mode = 'test';
net.move('gpu') ;
net.conserveMemory = true;
im_mean = net.meta(1).normalization.averageImage;
%im_mean = imresize(im_mean,[224,224]);
p = dir('/data/uts511/reid/market1501/bounding_box_test/*jpg');  % change image dir
ff = [];
%%------------------------------

for i = 1:200:numel(p)
    disp(i);
    oim = [];
    str=[];
    for j=1:min(200,numel(p)-i+1)
        str = strcat('/data/uts511/reid/market1501/bounding_box_test/',p(i+j-1).name);
        imt = imresize(imread(str),[224,224]);
        oim = cat(4,oim,imt);
    end
    f = getFeature2(net,oim,im_mean,'data','pool5n_local');
    f = sum(sum(f,1),2);
    f2 = getFeature2(net,fliplr(oim),im_mean,'data','pool5n_local');
    f2 = sum(sum(f2,1),2);
    f = f+f2;
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    s = sqrt(sum(f.^2,2));
    dim = size(f,2);
    s = repmat(s,1,dim);
    f = f./s;
    ff = cat(1,ff,f);
end
save('../test/resnet_gallery_align.mat','ff','-v7.3');
%}

%---------query
p = dir('/data/uts511/reid/market1501/query/*jpg');
ff = [];
for i = 1:200:numel(p)
    disp(i);
    oim = [];
    str=[];
    for j=1:min(200,numel(p)-i+1)
        str = strcat('/data/uts511/reid/market1501/query/',p(i+j-1).name);
        imt = imresize(imread(str),[224,224]);
        oim = cat(4,oim,imt);
    end
    f = getFeature2(net,oim,im_mean,'data','pool5n_local');
    f = sum(sum(f,1),2);
    f2 = getFeature2(net,fliplr(oim),im_mean,'data','pool5n_local');
    f2 = sum(sum(f2,1),2);
    f = f+f2;
    size4 = size(f,4);
    f = reshape(f,[],size4)';
    s = sqrt(sum(f.^2,2));
    dim = size(f,2);
    s = repmat(s,1,dim);
    f = f./s;
    ff = cat(1,ff,f);
end
save('../test/resnet_query_align.mat','ff','-v7.3');
