function [ ini ] = Init( image,n )
%INIT �˴���ʾ�йش˺�����ժҪ
%   n�������
ini = struct;
ini.constants = Init_constants(image);
ini.params = Init_params(image,n);

end

function [ constants ] = Init_constants(image)
%INIT �˴���ʾ�йش˺�����ժҪ
%   ����ĳ�����P,F,y
%   ����һ��struct constants
if size(image,1)~=size(image,2)
    return
end
constants = struct;
vec_len = size(image,1)^2;
constants.P = eye(vec_len);
constants.F = dftmtx(vec_len);
constants.P
x = reshape(image,[],1);
x
constants.y =  constants.P * constants.F * double(x);
end

function [ params ] = Init_params(image,n)
%INIT �˴���ʾ�йش˺�����ժҪ
%   �����ѧϰ������q,D,H,rho,eta
%   ����һ��struct params

%   ע�⣺�����Ԫ�����涼Ҫ�ĳ�n,l
if size(image,1)~=size(image,2)
    return
end
params = struct;
vec_len = size(image,1)^2;
lambda = 0.01;

params.q = cell(n,1);
params.D = cell(n,1);
params.H = cell(n,1);
params.rho = cell(n,1);
params.eta = cell(n,1);

for i = 1:n
   params.rho(i,1) = {0.01};
   params.q(i,1) = {sft_threshold_func(10,lambda,params.rho{i,1})};
   params.D(i,1) = {dctmtx(vec_len)};
   params.H(i,1) = {dctmtx(vec_len)};
   params.eta(i,1) = {0.01};
end

end