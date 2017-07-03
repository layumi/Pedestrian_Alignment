classdef Split < dagnn.ElementWise
  properties
    dim = 3
  end

  properties (Transient)
    inputSizes = {}
  end

  methods
    function outputs = forward(obj, inputs, params)
      x = gather(inputs{1});
      outputs{1} = gpuArray(x(:,:,1:2,:)) ;
      outputs{2} = gpuArray(x(:,:,3:4,:)) ;
      %obj.inputSizes = cellfun(@size, inputs, 'UniformOutput', false) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = gpuArray(vl_nnconcat(derOutputs,3));
      derParams = {} ;
    end

    function reset(obj)
      obj.inputSizes = {} ;
    end

    function obj = Split(varargin)
      obj.load(varargin{:}) ;
    end
  end
end
