function net = resnet52_market_stn_res_fine()

if(~exist('net_align.mat'))
    %------------main identification stream   (Base Branch in paper)
    %netStruct = load('/home/zzd/re_ID_gan_uts5/data/resnet52_2stream_drop0.9_baseline_batch32_gan24000_all/net_single.mat') ;
    netStruct = load('/home/zzd/re_ID_gan_uts5/data/res52_drop0.75_batch16_baseline/net-epoch-25.mat');
    net1 = dagnn.DagNN.loadobj(netStruct.net) ;
    for i = 1:numel(net1.params)
        net1.params(i).learningRate = 0;
        net1.params(i).weightDecay = 0;
    end
    net1.removeLayer('top5err');
    %-----------local stream   （Alignment Branch in paper）
    net2 = resnet52_market(); %imagenet
    %remove former
    for i = 1:35
        net2.removeLayer(net2.layers(1).name);
    end
    %change name
    for i = 1:numel(net2.layers)
        net2.renameLayer(net2.layers(i).name,sprintf('%s_local',net2.layers(i).name));
    end
    for i = 1:numel(net2.params)
        net2.renameParam(net2.params(i).name,sprintf('%s_local',net2.params(i).name));
    end
    for i = 1:numel(net2.vars)
        net2.renameVar(net2.vars(i).name,sprintf('%s_local',net2.vars(i).name));
    end
    %---------localization network （Grid Network in paper）
    net3 = resnet52_market(); %imagenet
    %remove former
    for i = 1:140
        net3.removeLayer(net3.layers(1).name);
    end
    %remove end
    %for i = 1:5
     %   net3.removeLayer(net3.layers(end).name);
    %end
    
    %change name
    for i = 1:numel(net3.layers)
        net3.renameLayer(net3.layers(i).name,sprintf('local_%s',net3.layers(i).name));
    end
    for i = 1:numel(net3.params)
        net3.renameParam(net3.params(i).name,sprintf('local_%s',net3.params(i).name));
    end
    for i = 2:numel(net3.vars)
        net3.renameVar(net3.vars(i).name,sprintf('local_%s',net3.vars(i).name));
    end
    %concat three nets
    net = concat_2net(net1,net2);
    net = concat_2net(net,net3);
    net_struct = net.saveobj();
    save('net_align.mat','net_struct');
else
    load('net_align.mat');
    net = dagnn.DagNN.loadobj(net_struct);
end

% Add extra layer to Grid network
% Predict 6-dim transform parameter
l_out128 = dagnn.Conv('size',[1,1,2048,128],'pad',0,'stride',1,'hasBias',true);
net.addLayer('l_out128', l_out128, {'local_pool5'}, {'local_pool5_128'}, {'lof128','lob128'});

l_out = dagnn.Conv('size',[1,1,128,6],'pad',0,'stride',1,'hasBias',true);
net.addLayer('l_out', l_out, {'local_pool5_128'}, {'aff'}, {'lof','lob'});
aff_grid = dagnn.AffineGridGenerator('Ho',56,'Wo',56);
net.addLayer('aff', aff_grid,{'aff'},{'grid'});
sampler = dagnn.BilinearSampler();
net.addLayer('samp',sampler,{'res2cx','grid'},{'res2c_local'});

%-------add loss 
dropoutBlock = dagnn.DropOut('rate',0.9);
net.addLayer('dropout_local',dropoutBlock,{'pool5_local'},{'pool5_locald'},{});
fc751Block = dagnn.Conv('size',[1 1 2048 751],'hasBias',true,'stride',[1,1],'pad',[0,0,0,0]);
net.addLayer('fc751_local',fc751Block,{'pool5_locald'},{'prediction_local'},{'fc751f_local','fc751b_local'});
net.addLayer('softmaxloss_local',dagnn.Loss('loss','softmaxlog'),{'prediction_local','label_local'},'objective_local');
net.addLayer('top1err_local', dagnn.Loss('loss', 'classerror'), ...
    {'prediction_local','label_local'}, 'top1err_local') ;

net.initParams();

%--------re-inital for local net
f_prev = net.params(net.getParamIndex('lof')).value;
net.params(net.getParamIndex('lof')).value = 0*f_prev;
b_prev = 0*net.params(net.getParamIndex('lob')).value;
b_prev(1) = 0.8; b_prev(4) = 0.8; 
net.params(net.getParamIndex('lob')).value = b_prev;
net.params(net.getParamIndex('lof')).learningRate = 1e-5;
net.params(net.getParamIndex('lob')).learningRate = 1e-5;

%----------test
net.conserveMemory = false;
%net.addLayer('Batch_Center_Loss',dagnn.Batch_Center_Loss(),{'res2c_local'},{'objective_align'},{});
net.eval({'data',single(rand(224,224,3,1)),'label',1,'label_local',1});

