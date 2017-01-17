%load Test_single_layer.m;

% what we need, Theta includes following parameters: {q,D,H,rho,eta } in
% every layer n *plus* {H,rho} in every filter L.

% BP initialization;
m_size = 4096;
B(:,:,1)=H;
B(:,:,2)=H;
B(:,:,3)=H;
B_2 = H;
B_3 = H;
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
tic;
% E gradient
E_2_x3 = (x_3-x_gt)/(norm(x_gt)*norm(x_3-x_gt));
% input gradient
Q = (P'*P+rho_3{1}*F*H_3{1}'*H_3{1}*F')^(-1);
x3_2_gama3 = -rho_3{1}*F'*(Q^2*(F*B_3'*H_3{1}*F'+F*H_3{1}'*B_3*F')*(P'*y+rho_3{1}*F*H_3{1}'*(z_2{1}-beta_2{1}))-Q*F*B_3'*(z_2{1}-beta_2{1}));
x3_2_rho3{1} = -F'*(Q^2*(F*H_3{1}'*H_3{1}*F')*(P'*y+rho_3{1}*F*H_3{1}'*(z_2{1}-beta_2{1}))-Q*F*H_3{1}'*(z_2{1}-beta_2{1}));
x3_2_beta2 = -rho_3{1}*F'*Q*F*H_3{1}';
x3_2_z2 = rho_3{1}*F'*Q*F*H_3{1}';

% output gradient(some of them will be used in upcoming steps in the network while others will be used in L-BFGS)
% these gradients can be used directly, because they are in Theta:
E_2_gama3 = x3_2_gama3' * E_2_x3;
E_2_rho3{1} = x3_2_rho3{1}' * E_2_x3;

% these gradients will be used in upcoming steps:
E_2_beta2_first = x3_2_beta2' * E_2_x3;
E_2_z2_first = x3_2_z2' * E_2_x3;
toc;
% BP Reconstruction Layer3 END-----

%---Third Layer END----------------------

%---Second Layer-------------------------
% BP Multiplier update Layer2-----
fprintf('BP MultiplierUpdate Layer2\n');
tic;
% E gradient
E_2_beta2 = E_2_beta2_first;

% input gradient
beta2_2_eta2 = c_2{1} - z_2{1};
beta2_2_c2 = eta_2{1}*eye(m_size);
beta2_2_z2 = -beta2_2_c2;
beta2_2_beta1 = eye(m_size);

% output gradient
% these gradients can be used directly, because they are in Theta:
E_2_eta2 = beta2_2_eta2'* E_2_beta2;
% these gradients will be used in upcoming steps:
E_2_c2_first = beta2_2_c2' * E_2_beta2;
E_2_z2_second = beta2_2_z2' * E_2_beta2;
E_2_beta1_first = beta2_2_beta1' * E_2_beta2;
toc;
% BP MultiplierUpdate Layer 2 END-----

% BP Nonlinear Transform Layer 2-----
fprintf('BP NonlinearTransform Layer2\n');
tic;
% E gradient
E_2_z2 = E_2_z2_first + E_2_z2_second;
% input gradient
z2_2_q2 = sft_func_bp_z2q(size(q_2{1},2),c_2{1}+beta_1{1});
z2_2_beta1 = sft_func_bp_z2beta(size(q_2{1},2),c_2{1}+beta_1{1},q_2{1});
z2_2_c2 = z2_2_beta1;

% output gradient
% these gradients can be used directly, because they are in Theta:
E_2_q2 = z2_2_q2' * E_2_z2;
% these gradients will be used in upcoming steps:
E_2_beta1_second = z2_2_beta1' * E_2_beta2;
E_2_c2_second = z2_2_c2' * E_2_beta2;
toc;
% BP Nonlinear Transform Layer 2 END-----

% BP Convolution Layer 2-----
fprintf('BP Convolution Layer2\n');
tic;
% E gradient
E_2_c2 = E_2_c2_first + E_2_c2_second;
% input gradient
c2_2_w2= zeros(size(x,1),size(B,3));
for i = 1:size(B,3)
    c2_2_w2(:,i) = B(:,:,i)*x_2;
end
c2_2_x2 = D_2{1};
% output gradient
% these gradients can be used directly, because they are in Theta:
E_2_w2 = c2_2_w2'* E_2_c2;
% these gradients will be used in upcoming steps:
% NULL
toc;
% BP Convolution Layer 2 END-----

% BP Reconstruction Layer2-----
tic;
% E gradient
E_2_x2 = c2_2_x2 * E_2_c2;
% input gradient
Q = (P'*P+rho_2{1}*F*H_2{1}'*H_2{1}*F')^(-1);
x2_2_gama2 = -rho_2{1}*F'*(Q^2*(F*B_2'*H_2{1}*F'+F*H_2{1}'*B_2*F')*(P'*y+rho_2{1}*F*H_2{1}'*(z_1{1}-beta_1{1}))-Q*F*B_2'*(z_1{1}-beta_1{1}));
x2_2_rho2{1} = -F'*(Q^2*(F*H_2{1}'*H_2{1}*F')*(P'*y+rho_2{1}*F*H_2{1}'*(z_1{1}-beta_1{1}))-Q*F*H_2{1}'*(z_1{1}-beta_1{1}));
x2_2_beta1 = -rho_2{1}*F'*Q*F*H_2{1}';
x2_2_z1 = rho_2{1}*F'*Q*F*H_2{1}';

% output gradient(some of them will be used in upcoming steps in the network while others will be used in L-BFGS)
% these gradients can be used directly, because they are in Theta:
E_2_gama2 = x2_2_gama2' * E_2_x2;
E_2_rho2{1} = x2_2_rho2{1}' * E_2_x2;

% these gradients will be used in upcoming steps:
E_2_beta1_first = x2_2_beta1' * E_2_x2;
E_2_z1_first = x2_2_z1' * E_2_x2;
toc;
% BP Reconstruction Layer2 END-----