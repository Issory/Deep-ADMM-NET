% Authored by Rick~
clc;
clear all;
img = imread('IM-0001-0001.jpg');
img = rgb2gray(img);
img = img(256:258,256:258);

N = 10;
[constants,params,net]  = Init( img,N );
[ x_output,net_forwarded ] = Forward( N,constants,params,net);
[ E,grad ] = Back( img,constants,params,net);