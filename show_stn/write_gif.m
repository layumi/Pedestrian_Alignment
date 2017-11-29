function [] = write_gif( im_stack,filename )

for i = 1:size(im_stack,4)
im = im_stack(:,:,:,i);
[A,map] = rgb2ind(im,256); 
	if i == 1;
		imwrite(A,map,filename,'gif','LoopCount',Inf,'DelayTime',0.1);
	else
		imwrite(A,map,filename,'gif','WriteMode','append','DelayTime',0.1);
	end
end

end

