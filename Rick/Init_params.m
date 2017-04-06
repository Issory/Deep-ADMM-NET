function [ params ] = Init_params( image )
%INIT 
%   q,D,H,rho,eta
if size(image,1)~=size(image,2)
    return
end
params = struct;
vec_len = size(image,1)^2;

end