clear
hoa_folder='/data/gs/mic_to_hoa_new/signal/test/uniform_mic(2)/3D_result_same/hoa_sig/';
% tet_folder='/data/gs/mic_to_hoa_new/signal/test/uniform_mic(2)/3D_result_same/est_final-8f(noise-with-norm-1s-4s(3)/';
tet_folder='/data/gs/mic_to_hoa_new/signal/test/uniform_mic(2)/3D_result_same/est_3_31(2)/';
mic_folder='/data/gs/mic_to_hoa_new/signal/test/uniform_mic(2)/3D_result_same/mic_sig/';
file_list=dir(tet_folder);

fs=48000;
fft_len=1024;
freq_list=(1:fft_len/2)/fft_len*fs;
rad=0.042;
Y=juzhen_new(4);                   %Y matrix;
E=(Y'*Y)^(-1)*Y';
for freq_ii=1:length(freq_list)
    freq=freq_list(freq_ii);
    tW=matrixEQ_old(4,freq,rad);
    T1{freq_ii}=tW*E;
end
load('cal_T');
T2=T;
corr_est=zeros(4,fft_len/2);
corr_cal1=zeros(4,fft_len/2);
corr_cal2=zeros(4,fft_len/2);
order_list1=[1:4];
order_list2=[1:9];
order_list3=[1:16];
order_list4=[1:25];
order_all_list={order_list1,order_list2,order_list3,order_list4};
p_tst_out=zeros(2,length(file_list)-2,25);
p_hoa_out=zeros(2,length(file_list)-2,25);
p_cal_out=zeros(2,length(file_list)-2,25);
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
    for sig_ii=3
        t_hoa=hoa(sig_ii,:);
        t_tst=tst(sig_ii,:);
        t_mic=mic(sig_ii,:);
        pro_hoa=inv_shape(t_hoa,25);
        pro_tst=inv_shape(t_tst,25);
        pro_mic=inv_shape(t_mic,32);
        f_list=[100,200];
        for freq_i=1:length(f_list)
            freq_ii=f_list(freq_i);
            p_mic=pro_mic(:,freq_ii);
            p_cal=T1{freq_ii-1}*p_mic;
            for order_ii=4
                p_hoa=pro_hoa(order_all_list{order_ii},freq_ii);
                p_tst=pro_tst(order_all_list{order_ii},freq_ii);       
                p_cal1=p_cal(order_all_list{order_ii});
                p_tst_out(freq_i,file_ii,:)=abs(p_tst)/1.8;
                p_hoa_out(freq_i,file_ii,:)=abs(p_hoa)/1.8;
                p_cal_out(freq_i,file_ii,:)=abs(p_cal1)/7;
            end
        end
    end
end
azi_list=(0:5:355)/180*pi;
ele_list=(90:-5:-90)/180*pi;
[azi_mat,ele_mat]=meshgrid(azi_list,ele_list);
wanted_order=[8,9,15,25];
% figure(3)
for order_ii=1:length(wanted_order)
    for f_ii=1:2
        figure(order_ii*2-2+f_ii)
%         subplot(2,4,order_ii+(f_ii-1)*4)
        pat_hoa=p_hoa_out(f_ii,:,wanted_order(order_ii));
        pat_hoa=reshape(pat_hoa,length(azi_list),length(ele_list));
        [x,y,z]=sph2cart(azi_mat,ele_mat,abs(pat_hoa')*1);
        h2=surf(x,y,z,abs(pat_hoa)');
        alpha(h2,0.6);
        shading interp
        axis equal;
        colormap jet
        hold on;
        pat_tst=p_tst_out(f_ii,:,wanted_order(order_ii));
        pat_tst=reshape(pat_tst,length(azi_list),length(ele_list));
        [x,y,z]=sph2cart(azi_mat,ele_mat,abs(pat_tst'));
        h1=surf(x,y,z,abs(pat_tst'),'EdgeColor','k');
    %     shading faceted
        alpha(h1,0.2);
        hold off
        axis off
    end
end
% figure(4)
for order_ii=1:length(wanted_order)
    for f_ii=1:2
        figure
%         subplot(2,4,order_ii+(f_ii-1)*4)
        pat_hoa=p_hoa_out(f_ii,:,wanted_order(order_ii));
        pat_hoa=reshape(pat_hoa,length(azi_list),length(ele_list));
        [x,y,z]=sph2cart(azi_mat,ele_mat,abs(pat_hoa')*1.1);
        h2=surf(x,y,z,abs(pat_hoa)');
        alpha(h2,0.8);
        shading interp
        axis equal;
        colormap jet
        hold on;
        pat_tst=p_cal_out(f_ii,:,wanted_order(order_ii));
        pat_tst=reshape(pat_tst,length(azi_list),length(ele_list));
        [x,y,z]=sph2cart(azi_mat,ele_mat,abs(pat_tst'));
        h1=surf(x,y,z,abs(pat_tst')/3,'EdgeColor','k');
    %     shading faceted
        alpha(h1,0.2);
        hold off
        axis off
    end
end


% set(findobj(get(gca,'Children'),'LineWidth','0.5'),'LineWidth',2)
% set(gca,'Fontname','Time newman','Fontsize',15)

function out=inv_shape(signal,ch_num)
   t1=reshape(signal,[2,length(signal)/2]);
   px_data=t1(1,:)+1i*t1(2,:);
   out=reshape(px_data,[ch_num,length(px_data)/ch_num]);
end