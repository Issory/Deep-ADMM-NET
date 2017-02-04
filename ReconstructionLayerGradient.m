%
%  ReconstructionLayerGradient.m
%  ADMM-NET
%
%  Created by Wang Han.SCU on 22/10/16.
%  Copyright (C) 2016 Deep ADMM NETWORK. SCU. All rights reserved.


function [E_2_gamma_n,E_2_rho_n,E_2_beta_nMinus1,E_2_z_nMinus1] = ...
    ReconstructionLayerGradient(E_2_x_n,P,F,y,H_n,B_m,rho_n,Z_n_Minus_1,beta_n_Minus_1,L,M)
% This function is aim to calculate the beta in the Reconstruction Layer

% output:
% E_2_gamma_n is the gradient between cost E and ¦Ã(n-1)
% E_2_rho_n is the gradient between cost E and ¦Ñ(n-1)
% E_2_beta_nMinus1 is the gradient between cost E and ¦Â(n-1)
% E_2_z_nMinus1 is the gradient between cost E and z(n-1)

% input:
% E_2_x_n is the gradient between E and x(n)
% P is a under-samping matrix;
% F is a Fourier transform, size N'*N ;
% y is the output of forward-propagating
% H_n is the transform matrix for a filtering operation
% B_m is the bais 
% rho_n is the learnable parameter
% Z_n_Minus_1 is n-1 layer of z
% beta_n_Minus_1 is n-1 layer of ¦Â
% L is the amount of layer, which has been definited before.
% M is the amount of bais, which has been definited before.

E_2_gamma_n = {zeros(L,M)}; 
E_2_rho_n = {zeros(1,L)};
E_2_beta_nMinus1 = {zeros(1,L)};
E_2_z_nMinus1 = {zeros(1,L)};

x_n_2_gamma_n = {zeros(L,M)};
x_n_2_rho_n = {zeros(1,L)};
x_n_2_beta_nMinus1 = {zeros(1,L)};
x_n_2_zMinus1 = {zeros(1,L)};

% temp parameter
tmp1 = 0;
tmp2 = 0;

for l  =1:L
    tmp1 = tmp1 + rho_n{l}*F*H_n{l}'*F';
    tmp2 = tmp2 + rho_n{L}*F*H_n{l}'*(Z_n_Minus_1{l}- beta_n_Minus_1{l});
end
Q = (P'*P+tmp1)^(-1);

for l = 1:L
    for m = 1:M
        x_n_2_gamma_n{l,m} = -rho_n{l}*F'*(Q^2*(F*B_m{l,m}'*H_n{l}*F'+F*H_n{l}'*B_m{l,m}*F')...
            *(P'*y+tmp2)*(Z_n_Minus_1{l}-beta_n_Minus_1{l})...
            -Q*F*B_m{l,m}'*(Z_n_Minus_1{l}-beta_n_Minus_1{l}));
        E_2_gamma_n{l,m} = x_n_2_gamma_n{l,m}' * E_2_x_n{l};
    end
    x_n_2_rho_n{l} = -F'*(Q^2*(F*H_n{l}'*H_n{l}*F')...
        *(P'*y+tmp2)-Q*F*H_n{l}'*(Z_n_Minus_1{l}-beta_n_Minus_1{l}));
    E_2_rho_n{l} = x_n_2_rho_n{l}' * E_2_x_n{l};
    x_n_2_beta_nMinus1{l} = -rho_n{l}*F'*Q*H_n(l)';
    E_2_beta_nMinus1{l} = x_n_2_beta_nMinus1{l}' * E_2_x_n{l};
    x_n_2_zMinus1{l} = rho_n(l)*F'*Q*H_n(l)';
    E_2_z_nMinus1{l} = x_n_2_zMinus1{l}'* E_2_x_n{l};
end
