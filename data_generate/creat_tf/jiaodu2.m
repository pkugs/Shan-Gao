function y=jiaodu2(theta,phi)

y=mic_angle;
rh=0.042;
x=y;
for i=1:32
    [xx,yy,zz]=sph2cart(theta,phi,20);
    c=acos((xx*cos(x(i,1))*rh*cos(x(i,2))+yy*sin(x(i,1))*rh*cos(x(i,2))+zz*rh*sin(x(i,2)))/(sqrt(xx^2+yy^2+zz^2)*rh));
    y(i,3)=c;  %????????????????
end

end
