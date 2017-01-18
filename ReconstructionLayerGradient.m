%
%  ReconstructionLayerGradient.m
%  ADMM-NET
%
%  Created by Wang Han.SCU on 22/10/16.
%  Copyright (C) 2016 Deep ADMM NETWORK. SCU. All rights reserved.

function [E2gamma_gradient,E2rho_gradient,E2beta_gradient,E2z_gradient,...
    x2gamma_gradient,x2rho_gradient,x2beta_gradient,x2z_gradient] = ...
    ReconstructionLayerGradient(F,L,M,H_n,P,E2x_n_gradient,rho_n,B_m,Z_n_Minus_1,beta_n_Minus_1)
% This function is aim to calculate the beta in the Reconstruction Layer
% input:

% F is a Fourier transform, size N'*N ;
% L is the amount of layer, which has been definited before.
% M is the amount of bais, which has been definited before.
% P is a under-samping matrix;
% while A = P*F,size N'*N; 
% H_n is a transform matrix for a filtering operation
% rho_n is the learnable parameter
% Z_n_Minus_1 
% beta_n_Minus_1

% output:
% E2gamma_gradient is the gradient between cost E and ¦Ã(gamma)
% E2gamma_gradient is the gradient between cost E and ¦Ñ(rho)
% E2beta_gradient is the gradient between cost E and ¦Â(beta)
% E2z_gradient is the gradient between cost E and z
E2gamma_gradient = {zeros(M,L)}; 
E2rho_gradient = {zeros(1,L)};
E2beta_gradient = {zeros(1,L)};
E2z_gradient = {zeros(1,L)};
x2gamma_gradient = {zeros(M,L)};
x2rho_gradient = {zeros(1,L)};
x2beta_gradient = {zeros(1,L)};
x2z_gradient = {zeros(1,L)};
I_n = eye(size(beta_n,1)); %I_n is an identity matrix sized N x N
temp_sum = 0;% sum in the formula
temp_sum_2 = 0;
for l  =1:L
    temp_sum=temp_sum+rho_n{l}*F*H_n{l}'*F';
    temp_sum_2 = temp_sum_2+rho_n{L}*F*H_n{l}'*(Z_n_Minus_1{l}- beta_n_Minus_1{l});
end
Q = (P'*P+temp_sum)^(-1);

for l = 1:L
    for m = 1:M
        x2gamma_gradient{m,l} = -rho_n{1}*F'*(Q^2*(F*B_m{m,l}'*H_n{1}*F'+F*H_n{1}'*B_m{m,l}*F')*(P'*y+temp_sum_2)*(Z_n_Minus_1{1}-beta_n_Minus_1{1})-Q*F*B_m{m,l}'*(Z_n_Minus_1{1}-beta_n_Minus_1{1}));
        E2gamma_gradient{m,l} = x2gamma_gradient{m,l}' * E2x_n_gradient;
    end
    x2rho_gradient{l} = -F'*(Q^2*(F*H_n{1}'*H_n{1}*F')*(P'*y+temp_sum_2)-Q*F*H_n{1}'*(Z_n_Minus_1{1}-beta_n_Minus_1{1}));
    E2rho_gradient{l} = x2rho_gradient{l}' * E2x_n_gradient;
    x2beta_gradient{l} = -rho_n{l}*F'*Q*H_n(l)'*I_n;
    E2beta_gradient{l} = x2beta_gradient{l}' * E2x_n_gradient;
    x2z_gradient{l} = rho_n(l)*F'*Q*H_n(l)'*I_n;
    E2z_gradient{l} = x2z_gradient{l}'* E2x_n_gradient;
end
