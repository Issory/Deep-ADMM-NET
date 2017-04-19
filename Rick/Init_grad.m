%初始化梯�?
function [init_grad,init_grad_opt] = Init_grad(N,net)
vec_size = size(net.x(1,1),1);


init_grad = struct;
init_grad_opt = struct;
init_grad_opt.gamma_grad = cell(N,1);
init_grad_opt.eta_grad = cell(N,1);
init_grad_opt.rho_grad = cell(N,1);
init_grad_opt.q_grad = cell(N,1);
init_grad_opt.w_grad = cell(N,1);

init_grad.x_grad = cell(N,1);
init_grad.z_grad = cell(N,1);
init_grad.beta_grad = cell(N,1);
init_grad.c_grad = cell(N,1);

for i = 1:N
    init_grad_opt.gamma_grad(i,1) = {rand(1,1)};
    init_grad_opt.eta_grad(i,1) = {rand(1,1)};
    init_grad_opt.rho_grad(i,1) = {rand(1,1)};
    init_grad_opt.q_grad(i,1) = {rand(10,1)};
    init_grad_opt.w_grad(i,1) = {rand(1,1)};

    init_grad.x_grad(i,1) = {rand(vec_size,1)};
    init_grad.z_grad(i,1) = {rand(vec_size,1)};
    init_grad.beta_grad(i,1) = {rand(vec_size,1)};
    init_grad.c_grad(i,1) = {rand(vec_size,1)};
end
end