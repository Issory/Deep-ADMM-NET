%
%  ConvolutionLayerGradient.m
%  ADMM-NET
%
%  Created by Wang Han.SCU on 22/10/16.
%  Copyright (C) 2016 Deep ADMM NETWORK. SCU. All rights reserved.

function [E_2_w_n,c_n_2_x_n] = ConvolutionLayerGradient(E_2_c_n,B_m,D_n,L,M,x_n)
    % This function is aimed to compute the gradient of convolution layer
    % output:
    % E_2_w_n is the gradient between E and w(n)
    % c_n_2_x_n is the gradient between c(n) and x(n)
    % input:
    % E_2_c_n is the gradient between E and c(n)
    % B_m is the bais
    % D_n is a transform matrix for a filtering operation such as DWT,DCT.
    % L is the number of z in column, which has been definited before.
    % M is the number of bais , which has been definited before.
    E_2_w_n = {zeros(L,M)};
    c_n_2_w_n= {zeros(L,M)};
    c_n_2_x_n = {zeros(1,L)};
    
    for l=1:L
        for m = 1:M
            c_n_2_w_n{l,m} = B_m{l,m}*x_n;
            E_2_w_n{l,m} = c_n_2_w_n{l,m}'* E_2_c_n{l};
        end
        c_n_2_x_n{l} = D_n{l};
    end
end
