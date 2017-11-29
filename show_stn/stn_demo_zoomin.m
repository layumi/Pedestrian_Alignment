clear;
addpath ..;
netStruct = load('../data/resnet52_stn_align_baseline_initial0.8_drop0.9_1e-5/net-epoch-40.mat');
net2 = dagnn.DagNN.loadobj(netStruct.net);
p = dir('./*jpg');
net2.mode = 'test' ;
net2.move('gpu') ;
net2.conserveMemory = false;
for j=1:numel(p)
    image_index = j;
    count = 1;
    for i=0:1:15
        str = strcat('./',p(image_index).name);
        o_im = imread(str);
        o_im = o_im(1+2*i:end-2*i,1+i:end-i,:);
        im = imresize(o_im,[224,224]);%,[224,224]);
        %subplot(1,2,1);
        %imshow(o_im);
        im_mean = net2.meta(1).normalization.averageImage;
        oim = bsxfun(@minus,single(im),im_mean);
        net2.layers(351).block.Ho = 128;
        net2.layers(351).block.Wo = 64;
        net2.eval({'data',gpuArray(oim)});
        grid = gather(net2.vars(net2.getVarIndex('grid')).value);
        %subplot(1,2,2);
        im_s = uint8(vl_nnbilinearsampler(single(o_im),grid));
        %imshow(im_s);
        im_stack(:,:,:,count) = cat(2,imresize(o_im,[128,64]),im_s);
        count = count+1;
    end
    write_gif(im_stack,sprintf('%s_zoomin.gif',p(image_index).name(1:end-4)));
end
