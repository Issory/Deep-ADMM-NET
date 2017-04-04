clear all;
img = imread('IM-0001-0001.jpg');
img = rgb2gray(img);
img = img(256:258,256:258);

ini  = Init( img,2 );