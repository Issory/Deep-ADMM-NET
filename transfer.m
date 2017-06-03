I=imread('image1/IM-0001-0001.jpg');
I=rgb2gray(I);
subplot(2,2,1)
imshow(I,[]);
title('512*512原始图像');
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
R1=reshape(R,512,360);
subplot(2,2,2);
imshow(R1,[]);
title('512*360投影图像');
