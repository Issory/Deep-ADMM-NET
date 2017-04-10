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