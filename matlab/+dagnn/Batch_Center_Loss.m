classdef Batch_Center_Loss < dagnn.Loss
    %EPE
    methods
        function outputs = forward(obj, inputs, params)
            x = gather(inputs{1})*0.1;
            t_mean = mean(x,4);
            t = bsxfun(@times,x,t_mean);
            t = reshape(t,1,[]);
            %t = t.^2;
            t(abs(t)>1) = abs(t(abs(t)>1));
            t(abs(t)<1) = t(abs(t)<1).^2;
            outputs{1} = sum(t);
            n = obj.numAveraged ;
            m = n + size(inputs{1},4) ;
            obj.average = (n * obj.average + double(gather(outputs{1}))) / m ;
            obj.numAveraged = m ;
        end
        
        function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
            %x -y ;
            x = gather(inputs{1})*0.1;
            t_mean = mean(x,4);
            t = bsxfun(@minus,x,t_mean);
            Y = t;
            Y(t>1) = 1;
            Y(t<-1) = -1;
            derInputs{1} = gpuArray(bsxfun(@times, derOutputs{1},Y));
            %derInputs{2} = [] ;
            derParams = {} ;
        end
        
        function obj = Batch_Center_Loss(varargin)
            obj.load(varargin) ;
        end
    end
end