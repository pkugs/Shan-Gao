clear
hoa_folder='/data/gs/mic_to_hoa_new/signal/test/uniform_mic(2)/horizontal-same/hoa_sig/';
% tet_folder='/data/gs/mic_to_hoa_new/signal/test/uniform_mic(2)/horizontal-same/est_final-8f(noise-with-norm-1s-4s(3)/';
tet_folder='/data/gs/mic_to_hoa_new/signal/test/uniform_mic(2)/horizontal-same/est_3_31(2)/';
mic_folder='/data/gs/mic_to_hoa_new/signal/test/uniform_mic(2)/horizontal-same/mic_sig/';
file_list=dir(tet_folder);

fs=48000;
fft_len=1024;
% freq_sign_list=ceil(10.^((1:fft_len/2)/50)/(10^(fft_len/100))*(fft_len/2-1));
freq_sign_list=1:12:fft_len/2;
freq_list=(1:fft_len/2)/fft_len*fs;

rad=0.042;
Y=juzhen_new(4);                   %Y matrix;
E=(Y'*Y)^(-1)*Y';
for freq_ii=1:length(freq_list)
    freq=freq_list(freq_ii);
    tW=matrixEQ_old(4,freq,rad);
    T1{freq_ii}=tW*E;
end
order_list1=[1:4];
order_list2=[1:9];
order_list3=[1:16];
order_list4=[1:25];
order_all_list={order_list1,order_list2,order_list3,order_list4};
p_tst_out=zeros(length(file_list)-2,length(freq_sign_list),25);
p_hoa_out=zeros(length(file_list)-2,length(freq_sign_list),25);
p_cal_out=zeros(length(file_list)-2,length(freq_sign_list),25);
sig_num=10;
for file_ii=1:length(file_list)-2
    if mod(file_ii,10)==0
        disp(file_ii)
    end
% for file_ii=13:14
    file_name=num2str(file_ii);
    hoa_file=[hoa_folder,file_name];
    tst_file=[tet_folder,file_name];
    mic_file=[mic_folder,file_name];
    load(hoa_file);
    hoa=data;
    load(tst_file);
    tst=data;
    tst=squeeze(tst);
%     tst=tst.';
    load(mic_file)
    mic=data;
%     for sig_ii= 1:length(hoa(:,1))
    for sig_ii=1
        t_hoa=hoa(sig_ii,:);
        t_tst=tst(sig_ii,:);
        t_mic=mic(sig_ii,:);
        pro_hoa=inv_shape(t_hoa,25);
        pro_tst=inv_shape(t_tst,25);
        pro_mic=inv_shape(t_mic,32);
        for freq_i=1:length(freq_sign_list)
            freq_ii=freq_sign_list(freq_i);
            p_mic=pro_mic(:,freq_ii+1);
            p_cal=T1{freq_ii}*p_mic;
            for order_ii=4
                p_hoa=pro_hoa(order_all_list{order_ii},freq_ii+1);
                p_tst=pro_tst(order_all_list{order_ii},freq_ii+1);       
                p_cal1=p_cal(order_all_list{order_ii});
                p_tst_out(file_ii,freq_i,:)=abs(p_tst)/1.8;
                p_hoa_out(file_ii,freq_i,:)=abs(p_hoa)/1.8;
                p_cal_out(file_ii,freq_i,:)=abs(p_cal1)/7;
%                 p_tst_out(file_ii,freq_i,:)=real(p_tst.'/p_hoa(1));
%                 p_hoa_out(file_ii,freq_i,:)=real(p_hoa.'/p_hoa(1));
%                 p_cal_out(file_ii,freq_i,:)=real(p_cal1.'/p_cal1(1));
            end
        end
    end
end
azi_list=(0:6:360)/180*pi;
[f_mat,azi_mat,]=meshgrid(freq_list(freq_sign_list),azi_list);
wanted_order=[4,9,16,25];

% for order_ii=1:length(wanted_order)
%     figure(order_ii)
%     pat_hoa=p_hoa_out(:,:,wanted_order(order_ii));
%     c=squeeze(abs(pat_hoa));
%     c=c/max(max(c));
%     [x,y,z]=pol2cart(azi_mat,abs(pat_hoa)*1.1,f_mat);
%     x=-x;
%     y=-y;
%     h2=surf(x,y,z,c);
%     alpha(h2,0.4);
%     colormap jet
%     shading interp
%     hold on;
%     pat_tst=p_cal_out(:,:,wanted_order(order_ii));
%     [x,y,z]=pol2cart(azi_mat,abs(pat_tst),f_mat);
%     c=squeeze(abs(pat_tst));
%     sign_c=max(max(c(:,1:20)));
%     c(find(c>sign_c))=1;
%     h1=surf(x,y,z,c/2,'EdgeColor',[0,0,0]);
% %     shading faceted
%     alpha(h1,0.4);
% %     colormap jet
%     xl=max(max(x(:,1:20)))*1.1;
%     axis([-xl xl -xl xl 0 12000])
%     hold off
%     set(gca,'Fontname','Time newman','Fontsize',20)
% end

for order_ii=1:length(wanted_order)
    figure(order_ii)
    pat_hoa=p_hoa_out(:,:,wanted_order(order_ii));
    c=squeeze(abs(pat_hoa));
    c=c/max(max(c));
    [x,y,z]=pol2cart(azi_mat,abs(pat_hoa)*1,f_mat);
    x=-x;
    y=-y;
    h2=surf(x,y,z,c);
    alpha(h2,0.4);
    colormap jet
    shading interp
    hold on;
    pat_tst=p_tst_out(:,:,wanted_order(order_ii));
    [x,y,z]=pol2cart(azi_mat,abs(pat_tst),f_mat);
    c=squeeze(abs(pat_tst));
    sign_c=max(max(c(:,1:20)));
    c(find(c>sign_c))=1;
    h1=surf(x,y,z,c/2,'EdgeColor',[0,0,0]);
%     shading faceted
    alpha(h1,0.7);
%     colormap jet
    xl=max(max(x(:,480:500)))*1.1;
    axis([-xl xl -xl xl 0 24000])
    hold off
    set(gca,'Fontname','Time newman','Fontsize',20)
    grid on
%     set(gca,'ZScale','log')
end

% set(findobj(get(gca,'Children'),'LineWidth','0.5'),'LineWidth',2)
% set(gca,'Fontname','Time newman','Fontsize',15)

function out=inv_shape(signal,ch_num)
   t1=reshape(signal,[2,length(signal)/2]);
   px_data=t1(1,:)+1i*t1(2,:);
   out=reshape(px_data,[ch_num,length(px_data)/ch_num]);
end