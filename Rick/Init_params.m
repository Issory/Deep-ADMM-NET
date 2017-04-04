function [ params ] = Init_params( image )
%INIT 此处显示有关此函数的摘要
%   输出的学习参数：q,D,H,rho,eta
if size(image,1)~=size(image,2)
    return
end
params = struct;
vec_len = size(image,1)^2;

end