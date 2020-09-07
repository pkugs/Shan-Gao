
function y=qudian()
n=300;          % the number of the point
a=rand(n,1)*2*pi;         % ????N??
b=asin(rand(n,1)*2-1);          
r0=[cos(a).*cos(b),sin(a).*cos(b),sin(b)];
v0=zeros(size(r0));
G=1e-2;               %????
wucha1=zeros(1,200);
wucha2=zeros(1,200);
for ii=1:200              %??200?            
    [rn,vn,fv1]=countnext(r0,v0,G);         %more effective
    r0=rn;
    v0=vn;
    wucha1(ii)=fv1;%lingjia
end

r0=[cos(a).*cos(b),sin(a).*cos(b),sin(b)];
v0=zeros(size(r0));
G=1e-2;     
for ii=1:200              %??200?            
    [r1,v1,fv2]=countnext2(r0,v0,G);         %??????????
    r0=r1;
    v0=v1;
    wucha2(ii)=fv2;%lingjia
end


figure(1);
% plot3(rn(:,1),rn(:,2),rn(:,3),'.');
% figure(2);
plot3(rn(:,1),rn(:,2),rn(:,3),'kd');
%实验

%实验
xlabel('x');
ylabel('y');
zlabel('z');
hold on;
% figure(2)
dt = DelaunayTri(rn);  %利用Delaunay将点划分为空间4面体
[ch] = convexHull(dt); %利用convexHull求凸包表面和凸包体积
trisurf(ch,rn(:,1),rn(:,2),rn(:,3),'FaceColor','w','facealpha',0.8);%画凸多面体网格
% [xx,yy,zz]=sphere(30);
% h2=surf(xx,yy,zz);
% set(h2,'edgecolor','k','facecolor','w','facealpha',0.8);
axis equal;
axis([-1 1 -1 1 -1 1]);
% title('');
hold off;

figure(3)
ii=1:200;
plot(ii,wucha1,ii,wucha2,'r');    %compare;
legend('1','2')



y=rn;

end