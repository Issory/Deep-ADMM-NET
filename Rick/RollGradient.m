%
%  RollGradinet.m
%  ADMM-NET
%
%  Created by Wang Han.SCU on 8/4/17.
%  Copyright (C) 2016 Deep ADMM NETWORK. SCU. All rights reserved.

function roll_grad = RollGradinet(unroll_grad,grad)
% This function is aim to unroll the gradient struct 
% 
% input:
% unroll_grad is a vector which is unrolled gradient
% output: 
% roll_grad is a struct which is rolled gradient 
%
roll_grad=struct;

index =1;
% roll_grad.gamma_grad
unroll_gamma_grad=cell(size(grad.gamma_grad,1),1);
for i=1:size(grad.gamma_grad,1)
        length=size(grad.gamma_grad{i},1);
        unroll_gamma_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.gamma_grad=unroll_gamma_grad;

% roll_grad.eta_grad
unroll_eta_grad=cell(size(grad.eta_grad,1),1);
for i=1:size(grad.eta_grad,1)
        length=size(grad.eta_grad{i},1);
        unroll_eta_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.eta_grad=unroll_eta_grad;

% roll_grad.rho_grad
unroll_rho_grad=cell(size(grad.rho_grad,1),1);
for i=1:size(grad.rho_grad,1)
        length=size(grad.rho_grad{i},1);
        unroll_rho_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.rho_grad=unroll_rho_grad;

% roll_grad.q_grad
unroll_q_grad=cell(size(grad.q_grad,1),1);
for i=1:size(grad.q_grad,1)
        length=size(grad.q_grad{i},1);
        unroll_q_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.q_grad=unroll_q_grad;

% roll_grad.w_grad
unroll_w_grad=cell(size(grad.w_grad,1),1);
for i=1:size(grad.w_grad,1)
        length=size(grad.w_grad{i},1);
        unroll_w_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.w_grad=unroll_w_grad;

% roll_grad.x_grad
unroll_x_grad=cell(size(grad.x_grad,1),1);
for i=1:size(grad.x_grad,1)
        length=size(grad.x_grad{i},1);
        unroll_x_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.x_grad=unroll_x_grad;

% roll_grad.z_grad
unroll_z_grad=cell(size(grad.z_grad,1),1);
for i=1:size(grad.z_grad,1)
        length=size(grad.z_grad{i},1);
        unroll_z_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.z_grad=unroll_z_grad;

% roll_grad.beta_grad
unroll_beta_grad=cell(size(grad.beta_grad,1),1);
for i=1:size(grad.beta_grad,1)
        length=size(grad.beta_grad{i},1);
        unroll_beta_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.beta_grad=unroll_beta_grad;

% roll_grad.c_grad
unroll_c_grad=cell(size(grad.c_grad,1),1);
for i=1:size(grad.c_grad,1)
        length=size(grad.c_grad{i},1);
        unroll_c_grad{i}=unroll_grad(index:index+length-1);
        index=index+length;
end
roll_grad.c_grad=unroll_c_grad;

% roll.grad.gamma_beta
%unroll_gamma_beta=cell(size(grad.gamma_beta,1),1);
%for i=1:size(grad.gamma_beta,1)
%        length=size(grad.gamma_beta{i},1);
%        unroll_gamma_beta{i}=unroll_grad(index:index+length-1);
%        index=index+length;
%end
%roll_grad.gamma_beta=unroll_gamma_beta;

end
