function [ constants ] = Init_constants( image )
%INIT 此处显示有关此函数的摘要
%   输出的常量：
if size(image,1)~=size(image,2)
    return
end
constants = struct;
vec_len = size(image,1)^2;

end

