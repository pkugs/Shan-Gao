clear
hoa_folder='/data/gs/mic_to_hoa_new/signal/test/uniform_mic_20dB/single_source/hoa_sig/';
tet_folder='/data/gs/mic_to_hoa_new/signal/test/uniform_mic_20dB/single_source/est_final-8f(noise)/';
mic_folder='/data/gs/mic_to_hoa_new/signal/test/uniform_mic_20dB/single_source/mic_sig/';
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
for file_ii=3:length(file_list)
    file_ii
% for file_ii=13:14
    file_name=file_list(file_ii).name;
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
    for sig_ii= 1:length(hoa(:,1))
        t_hoa=hoa(sig_ii,:);
        t_tst=tst(sig_ii,:);
        t_mic=mic(sig_ii,:);
        pro_hoa=inv_shape(t_hoa,25);
        pro_tst=inv_shape(t_tst,25);
        pro_mic=inv_shape(t_mic,32);
        for freq_ii=2:length(pro_hoa(1,:))
            p_mic=pro_mic(:,freq_ii);
            p_cal=T1{freq_ii-1}*p_mic;
            for order_ii=1:4
                p_hoa=pro_hoa(order_all_list{order_ii},freq_ii);
                p_tst=pro_tst(order_all_list{order_ii},freq_ii);       
                p_cal1=p_cal(order_all_list{order_ii});
                corr_est(order_ii,freq_ii)=corr_est(order_ii,freq_ii)+abs(p_hoa'*p_tst)/...
                    sqrt(p_hoa'*p_hoa)/sqrt(p_tst'*p_tst); 
                corr_cal1(order_ii,freq_ii)=corr_cal1(order_ii,freq_ii)+abs(p_hoa'*p_cal1)/...
                    sqrt(p_hoa'*p_hoa)/sqrt(p_cal1'*p_cal1);
            end
        end
    end
end
figure(2)
% semilogx(freq_list,corr_result2/(file_num-2))
plot(freq_list,corr_cal1/51)

hold on
plot(freq_list,corr_est/51,'.-')
hold off
xlabel('Frequency(Hz)')
axis([30 20000 -inf inf])
ylabel('Sptial Correlation')
legend('1^{th}order','2^{th}order',...
           '3^{th}order','4^{th}order',...
            'nn-1^{th}order','nn-2^{th}order',...
           'nn-3^{th}order','nn-4^{th}order')
legend('location','northeast')

set(findobj(get(gca,'Children'),'LineWidth','0.5'),'LineWidth',2)
set(gca,'Fontname','Time newman','Fontsize',15)

function out=inv_shape(signal,ch_num)
   t1=reshape(signal,[2,length(signal)/2]);
   px_data=t1(1,:)+1i*t1(2,:);
   out=reshape(px_data,[ch_num,length(px_data)/ch_num]);
end