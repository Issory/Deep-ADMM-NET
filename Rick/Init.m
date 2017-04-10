function [ constants,params,net,init_grad ] = Init( image,N )
%INIT 初始化函数
%   输入：image图片、N层数(初始化时要多放一层，第一层作为初始值层)
%   输出：constants常量、params待更新变量、net网络
image = double(image);
constants = Init_constants(image);
params = Init_params(image,N+1);
net = Init_net(image,N+1);
init_grad = Init_grad(N,net);

end

function [ constants ] = Init_constants(image)
%INIT 
%   P,F,y
%   struct constants
if size(image,1)~=size(image,2)
    return
end
constants = struct;
vec_len = size(image,1)^2;
constants.P = eye(vec_len);
constants.F = dftmtx(vec_len);
x = reshape(image,[],1);
constants.y =  constants.P * constants.F * double(x);
end

function [ params ] = Init_params(image,n)
%INIT 
%   q,D,H,rho,eta
%  struct params

if size(image,1)~=size(image,2)
    return
end
params = struct;
vec_len = size(image,1)^2;
lambda = 0.01;

params.q = cell(n,1);
params.D = cell(n,1);
params.H = cell(n,1);
params.B = cell(n,1);
params.rho = cell(n,1);
params.eta = cell(n,1);

for i = 1:n
   params.rho(i,1) = {rand(1)};
   params.q(i,1) = {sft_threshold_func(10,lambda,params.rho{i,1})};
   params.D(i,1) = {dctmtx(vec_len)};
   params.H(i,1) = {dctmtx(vec_len)};
   params.eta(i,1) = {0.01};
   
end
end

function [ net ] = Init_net(image,n)
vec_len = size(image,1)^2;
net = struct;

net.beta = cell(n,1);
net.c = cell(n,1);
net.z = cell(n,1);
net.x = cell(n,1);
for i = 1:n
    net.beta(i,1) = {zeros(vec_len,1)};
    net.c(i,1) = {zeros(vec_len,1)};
    net.z(i,1) = {zeros(vec_len,1)};
    net.x(i,1) = {zeros(vec_len,1)};%{reshape(image,[],1)};
end
net.x(1,1) = {real(reshape(image,[],1))};


end

%初始化梯度
function init_grad = Init_grad(N,net)
vec_size = size(net.x(1,1),1);


init_grad = struct;
init_grad.gamma_grad = cell(N,1);
init_grad.eta_grad = cell(N,1);
init_grad.rho_grad = cell(N,1);
init_grad.q_grad = cell(N,1);
init_grad.w_grad = cell(N,1);

init_grad.x_grad = cell(N,1);
init_grad.z_grad = cell(N,1);
init_grad.beta_grad = cell(N,1);
init_grad.c_grad = cell(N,1);

for i = 1:N
    init_grad.gamma_grad(i,1) = {0};
    init_grad.eta_grad(i,1) = {0};
    init_grad.rho_grad(i,1) = {0};
    init_grad.q_grad(i,1) = {zeros(10,1)};
    init_grad.w_grad(i,1) = {0};

    init_grad.x_grad(i,1) = {zeros(vec_size,1)};
    init_grad.z_grad(i,1) = {zeros(vec_size,1)};
    init_grad.beta_grad(i,1) = {zeros(vec_size,1)};
    init_grad.c_grad(i,1) = {zeros(vec_size,1)};
end
end