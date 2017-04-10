%
%  UnrollGradinet.m
%  ADMM-NET
%
%  Created by Wang Han.SCU on 8/4/17.
%  Copyright (C) 2016 Deep ADMM NETWORK. SCU. All rights reserved.

function unroll_grad = UnrollGradient(roll_grad)
% This function is aim to unroll the gradient struct 
% 
% input: 
% roll_grad is a struct which is rolled gradient 
% ouput:
% unroll_grad is a vector which is unrolled gradient
%

% roll_grad.gamma_grad
unroll_gamma_grad=[];
for i=1:size(roll_grad.x_grad,1)
        unroll_gamma_grad=[unroll_gamma_grad;roll_grad.gamma_grad{i}(:)];
end

% roll_grad.eta_grad
unroll_eta_grad=[];
for i=1:size(roll_grad.eta_grad,1)
        unroll_eta_grad=[unroll_eta_grad;roll_grad.eta_grad{i}(:)];
end

% roll_grad.rho_grad
unroll_rho_grad=[];
for i=1:size(roll_grad.rho_grad,1)
        unroll_rho_grad=[unroll_rho_grad;roll_grad.rho_grad{i}(:)];
end

% roll_grad.q_grad
unroll_q_grad=[];
for i=1:size(roll_grad.q_grad,1)
        unroll_q_grad=[unroll_q_grad;roll_grad.q_grad{i}(:)];
end

% roll_grad.w_grad
unroll_w_grad=[];
for i=1:size(roll_grad.w_grad,1)
        unroll_w_grad=[unroll_w_grad;roll_grad.w_grad{i}(:)];
end

% roll_grad.x_grad
unroll_x_grad=[];
for i=1:size(roll_grad.x_grad,1)
        unroll_x_grad=[unroll_x_grad;roll_grad.x_grad{i}(:)];
end

% roll_grad.z_grad
unroll_z_grad=[];
for i=1:size(roll_grad.z_grad,1)
        unroll_z_grad=[unroll_z_grad;roll_grad.z_grad{i}(:)];
end

% roll_grad.beta_grad
unroll_beta_grad=[];
for i=1:size(roll_grad.beta_grad,1)
        unroll_beta_grad=[unroll_beta_grad;roll_grad.beta_grad{i}(:)];
end

% roll_grad.c_grad
unroll_c_grad=[];
for i=1:size(roll_grad.c_grad,1)
        unroll_c_grad=[unroll_c_grad;roll_grad.c_grad{i}(:)];
end

% roll.grad.gamma_beta
%unroll_gamma_beta=[];
%for i=1:size(roll_grad.gamma_beta,1)
%        unroll_gamma_beta=[unroll_gamma_beta;roll_grad.gamma_beta{i}(:)];
%end

unroll_grad=[unroll_gamma_grad;unroll_eta_grad;unroll_rho_grad;unroll_q_grad;unroll_w_grad;unroll_x_grad;unroll_z_grad;unroll_beta_grad;unroll_c_grad];%unroll_gamma_beta];
end

