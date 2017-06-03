% 卷积反投影重建;庄天戈《CT原理与算法》/卷积反投影重建/f3.27;笔束平移旋转扫描
%% 
clear;
clc;
close all;
tic;
%% initial
I=phantom(256);
subplot(2,2,1)
imshow(I,[]);
title('256*256原始图像');
[N,N]=size(I);
z=2*ceil(norm(size(I)-floor((size(I)-1)/2)-1))+3;% radon变换默认平移点数/角度
Nt=360;% 角度采样点数
Nd=N;% 平移数
x=pi/180;% 角度增量
d=N/Nd;% 平移步长
theta = 1:Nt;
a=zeros(N);
%%
% 产生无噪声投影数据
[R,xp] = radon(I,theta);% 产生I投影,默认z点/角度,即使指定N点也是z点.
                        % 所以为避免重建图像放大或缩小,下面计算取投影时需补偿,补偿量e
                        % 如对256的图像,补偿为55,即pm的第55个点作为计算用的第一个投影
e=floor((z-Nd)/2)+2;
R=R(e:(Nd+e-1),:);
R1=reshape(R,256,360);
% 添加噪声
[mm,nn]=size(R1);
di=lognrnd(0,0.15,mm,nn);
R1= 10*(R1-min(R1(:)))/( max(R1(:))-min(R1(:)));
I0 = 1.5e5; % incident photons; decrease this for simulating "low dose" scans
rand('state', 0), randn('state', 0);
yi= poissrnd(I0 * di.*exp(-R1))+3*randn(size(R1));

if any(yi(:) == 0)
  warn('%d of %d values are 0 in sinogram!', ...
       sum(yi(:)==0), length(yi(:)));
end
R1 = log(I0 ./ max(yi,0.01)); % noisy sinogram
R1=max(R1,0); % R1 加噪的投影数据

% load R3.MAT
ff=2;
uu=22000;
v=ff*exp(R1/uu);
subplot(2,2,2)
imagesc(R1);
title('256*360有噪声平行投影');
colormap(gray)
colorbar
Q=reshape(R1,256,360);
%%  

% designing RL filter 
g=-(Nd/2-1):(Nd/2);
for i=1:256
    if g(i)==0
        hl(i)=1/(4*d^2);
    else if  mod(g(i),2)==0
            hl(i)=0;
        else
            hl(i)=(-1)/(pi^2*d^2*(g(i)^2));
        end
    end
end
k=Nd/2:(3*Nd/2-1);% 取卷积结果时用
%% 
% 重建过程
for m=1:Nt
    % reading projection
    pm=Q(:,m);% 读取投影数据
    u=conv(hl,pm);% 卷积
    pm=u(k);% 取卷积结果
    Cm=((N-1)/2)*(1-cos((m-1)*x)-sin((m-1)*x));
    for i=1:N
        for j=1:N
            Xrm=Cm+(j-1)*cos((m-1)*x)+(i-1)*sin((m-1)*x);
            if Xrm<1
                n=1;
                t=abs(Xrm)-floor(abs(Xrm));
            else
                n=floor(Xrm);
                t=Xrm-floor(Xrm);
            end
            if n>(Nd-1)
                n=Nd-1;
            end
            p=(1-t)*pm(n)+t*pm(n+1);
            a(N+1-i,j)=a(N+1-i,j)+p;
        end
    end
end
%%
I=I-min(min(I));
I=1/(max(max(I)))*I;
% 将a的元素值变换到0-255之间，利于显示
a=a-min(min(a));
a=1/(max(max(a)))*a;
subplot(2,2,3);
imshow(a);
title('重建图像');
%%  
m60=I(230,:);%取原始图像的某一行灰度值
n60=a(230,:);%取重建后图像的某一行灰度值
subplot(2,2,4);
plot(m60,'b','linewidth',2,'linestyle','--');
axis([0,130,0,1.5]);
grid on;  % 控制分隔符
hold on   % 使图形在一个界面上画
plot(n60,'r','linewidth',2,'linestyle',':');
xlabel('第230行的像素','fontsize',10);
ylabel('灰度值','fontsize',10);
legend('原始图像灰度','重建图像灰度',1);
%%
%评价参数
%归一化均方距离判据dd
m1=0;
n1=0;
tt=0;
for j=1:256
    for i=1:256
        tt=tt+I(i,j);
    end
end
tt=tt/(256*256);
for j=1:256
    for i=1:256
        m1=m1+(I(i,j)-a(i,j))^2;
        n1=n1+(I(i,j)-tt)^2;
    end
end
dd=sqrt(m1/n1);

%归一化平均绝对距离判据rr
m2=0;
n2=0;
for j=1:256 
    for i=1:256
        m=m+abs(I(i,j)-a(i,j));
        n=n+abs(I(i,j));
    end
end
rr=m/n;

% %最坏情况距离判据ee
sprintf('归一化均方距离判据: %f',dd)  % 输出数
sprintf('归一化平均绝对距离判据: %f',rr)  % 输出数
%%
toc
s=int2str(toc);
s=['totaltime = ' s ' s'];% 字符串组合
text(0,300,s,'FontSize',14);


