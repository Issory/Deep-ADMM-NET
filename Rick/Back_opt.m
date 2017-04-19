function [ E,grad_opt ] = Back_opt( grad_opt_unroll,img,constants,params,init_grad,net)
% Authored by Rick~
%BACK 
%   input:
%   constants
%   params
%   net
%   img

%   output:
%   E
%   grad
N = size(net.x,1)-1;

% Constants
F = constants.F;
P = constants.P;
y = constants.y;
% parameters
H = params.H;
D = params.D;
rho = params.rho;
eta = params.eta;
q = params.q;

x_gt = double(reshape(img,[],1));
init_grad_opt = RollGradient(grad_opt_unroll,init_grad);
grad_opt = init_grad_opt;
grad = init_grad;
E = norm(cell2mat(net.x(N,1))-x_gt)/norm(x_gt);
[B,w] = Init_temp_params(N,net);

options = struct('GradObj','on','Display','iter','LargeScale','off','HessUpdate','lbfgs','InitialHessType','identity','GoalsExactAchieve',0);

L = 1;
M =1;


%E_2_x_3 = {(x_3-x_gt)/(norm(x_gt)*norm(x_3-x_gt))};
grad.x_grad(N,1) = {(cell2mat(net.x(N,1))-x_gt)/...
    (norm(x_gt)*norm((cell2mat(net.x(N,1))-x_gt)))};


for i = N-1:-1:2 
fprintf('BP Reconstruction Layer %d\n',i);
tic;
% E gradient
% function [E_2_gamma_n,E_2_rho_n,E_2_beta_nMinus1,E_2_z_nMinus1] = ...
% ReconstructionLayerGradient(E_2_x_n,P,F,y,H_n,B_m,rho_n,Z_n_Minus_1,beta_n_Minus_1,L,M)
[grad_opt.gamma_grad(i+1,1),grad_opt.rho_grad(i+1,1),E_2_beta_nMinus1_first,E_2_z_n_first]=...
    ReconstructionLayerGradient(grad.x_grad(N,1),...
    P,F,y,H(i+1,1),B(i+1,1),rho(i+1,1),net.z(i,1),net.beta(i,1),L,M);

%---
%[new_grad,cost] = fminunc( @(p)opt_rho(E,grad,i+1),real(grad.rho_grad{i+1,1}),options);
%---

if i==N-1
    grad.beta_grad(i,1) = E_2_beta_nMinus1_first;
else
    grad.beta_grad(i,1) = {cell2mat(E_2_beta_nMinus1_first) + ...
        cell2mat(E_2_beta_nMinus1_second) + cell2mat(E_2_beta_nMinus1_third)};
end


% BP Multiplier update Layer-----
fprintf('BP MultiplierUpdate Layer %d\n',i-1);
tic;
% function [E_2_eta_n,E_2_beta_nMinus1,E_2_c_n,E_2_z_n] = MultiplierUpdateLayerGradient(E_2_beta_n,c_n,z_n,eta_n,L)
[grad_opt.eta_grad(i,1),E_2_beta_nMinus1_second,E_2_c_n_first,E_2_z_second] = ...
    MultiplierUpdateLayerGradient(grad.beta_grad(i,1),net.c(i,1),net.z(i,1),eta(i,1),1);
toc;
% BP MultiplierUpdate Layer END-----

% BP Nonlinear Transform Layer-----
fprintf('BP NonlinearTransform Layer %d\n',i-1);
tic;
% E gradient
grad.z_grad(i,1) = {cell2mat(E_2_z_n_first) + cell2mat(E_2_z_second)};
% use function to compute
% [E_2_q_n,E_2_beta_nMinus1,E_2_c_n] =  NonlinearTransformLayerGradient(E_2_z_n,c_n,beta_nMinus1,q,L)
[grad_opt.q_grad(i,1),E_2_beta_nMinus1_third,E_2_c_n_second]= NonlinearTransformLayerGradient...
    (grad.z_grad(i,1),net.c(i,1),net.beta(i-1,1),q(i,1),1);
toc;
% BP Nonlinear Transform Layer END-----

% BP Convolution Layer-----
fprintf('BP Convolution Layer %d\n',i-1);
tic;
% E gradient
grad.c_grad(i,1) = {cell2mat(E_2_c_n_first)+ cell2mat(E_2_c_n_second)};
% function [E_2_w_n,c_n_2_x_n] = ConvolutionLayerGradient(E_2_c_n,B_m,D_n,L,M)
[grad_opt.w_grad(i,1),c_n_2_x_n] = ConvolutionLayerGradient(grad.c_grad(i,1),B(i,1),D(i,1),1,1,net.x{i,1});
toc;
% BP Convolution LayerEND-----
toc;

grad.x_grad(i,1) = {cell2mat(c_n_2_x_n) * cell2mat(grad.c_grad(i,1))};

end

grad_opt = real(UnrollGradient(grad_opt));

end

function [B,w] = Init_temp_params(N,net)
    vec_size = size(net.x(1,1),1);
    B = cell(N,1);
    w = cell(N,1);
    for i=1:N
        B(i,1) = {dctmtx(vec_size)};
        w(i,1) = {1/N};
    end
end

%---------------- for opt function---
function [E_output,grad_gamma] = opt_gamma(E,grad,i)
    E_output = E;
    grad_gamma = grad.gamma_grad(i,1);
end
function [E_output,grad_rho] = opt_rho(E,grad,i)
    E_output = E;
    grad_rho = real(grad.rho_grad{i,1});
end