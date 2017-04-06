function [ constants ] = Init_constants( image )
%INIT 
%  中文
if size(image,1)~=size(image,2)
    return
end
constants = struct;
vec_len = size(image,1)^2;

end

