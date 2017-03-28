% BP initialization;
m_size = 9;
B(:,:,1)=H;
B(:,:,2)=H;
B(:,:,3)=H;
B_2 = {H};
B_3 = {H};
w_1 = eye(1,3)/3;
H_3 = {H};
w_2 = eye(1,3)/3;
gama_3 = w_2;
gama_2 = gama_3;
gama_1 = gama_2;
x_3 = real(x_3);
x_gt = double(reshape(img,[],1));
E = norm(x_3-x_gt)/norm(x_gt);

%---Third Layer--------------------------
% BP Reconstruction Layer3-----
fprintf('BP Reconstruction Layer3\n');
L = 1;
M =1;
B_m = {H}; 
H_n = {H};
E_2_x_3 = {(x_3-x_gt)/(norm(x_gt)*norm(x_3-x_gt))};
tic;
% E gradient
% function [E_2_gamma_n,E_2_rho_n,E_2_beta_nMinus1,E_2_z_nMinus1] = ReconstructionLayerGradient(E_2_x_n,P,F,y,H_n,B_m,rho_n,Z_n_Minus_1,beta_n_Minus_1,L,M)
[E_2_gamma_3,E_2_rho_3,E_2_beta_2,E_2_z_2_fisrt]=ReconstructionLayerGradient(E_2_x_3,P,F,y,H_3,B_3,rho_3,z_2,beta_2,1,1);
% BP Reconstruction Layer3 END-----
toc;
% BP Reconstruction Layer3 END-----
%---Third Layer END----------------------


%---Second Layer-------------------------
% BP Multiplier update Layer2-----
fprintf('BP MultiplierUpdate Layer2\n');
tic;
% function [E_2_eta_n,E_2_beta_nMinus1,E_2_c_n,E_2_z_n] = MultiplierUpdateLayerGradient(E_2_beta_n,c_n,z_n,eta_n,L)
[E_2_eta_1,E_2_beta_1_first,E_2_c_2_first,E_2_z_2_second] = MultiplierUpdateLayerGradient(E_2_beta_2,c_2,z_2,eta_2,1);
toc;
% BP MultiplierUpdate Layer 2 END-----

% BP Nonlinear Transform Layer 2-----
fprintf('BP NonlinearTransform Layer2\n');
tic;
% E gradient
E_2_z_2 = {cell2mat(E_2_z_2_fisrt) + cell2mat(E_2_z_2_second)};
% use function to compute
% [E_2_q_n,E_2_beta_nMinus1,E_2_c_n] =  NonlinearTransformLayerGradient(E_2_z_n,c_n,beta_nMinus1,q,L)
[E_2_q_2,E_2_beta_1_second,E_2_c_2_second]= NonlinearTransformLayerGradient(E_2_z_2,c_2,beta_1,q_2,1);
toc;
% BP Nonlinear Transform Layer 2 END-----

% BP Convolution Layer 2-----
fprintf('BP Convolution Layer2\n');
tic;
% E gradient
E_2_c_2 = {cell2mat(E_2_c_2_first)+ cell2mat(E_2_c_2_second)};
% function [E_2_w_n] = ConvolutionLayerGradient(E_2_c_n,B_m,D_n,L,M)
E_2_w_2 = ConvolutionLayerGradient(E_2_c_2,B_2,D_2,1,1,x_2);
toc;
% BP Convolution Layer 2 END-----

% BP Reconstruction Layer2-----
fprintf('BP Reconstruction Layer2\n');
tic;
% E gradient
E_2_x_2 = {cell2mat(c_2_2_x_2) * cell2mat(E_2_c_2)};
% function [E_2_gamma_n,E_2_rho_n,E_2_beta_nMinus1,E_2_z_nMinus1] = ReconstructionLayerGradient(E_2_x_n,P,F,y,H_n,B_m,rho_n,Z_n_Minus_1,beta_n_Minus_1,L,M)
% output gradient
[E_2_gamma_2,E_2_rho_2,E_2_beta_1,E_2_z_1] = ReconstructionLayerGradient(E_2_x_2,P,F,y,H_2,B_2,rho_2,z_1,beta_1,1,3);
toc;
% BP Reconstruction Layer2 END-----

