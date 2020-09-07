function [corr,ld]=cal_sh_corr_func(sign_data,input)

am_coe=abs(input(:,1)'*sign_data(:,1))/(sign_data(:,1)'*sign_data(:,1));
sign_data=sign_data/am_coe;

fft_sign_data=fft(sign_data);
fft_input=fft(input);
fft_len=1024;
corr=zeros(1,fft_len/2);
ld=zeros(1,fft_len/2);
        
for freq_ii=2:fft_len/2+1
    t_sign=fft_sign_data(freq_ii,:);
    t_hoa=fft_input(freq_ii,:);
    temp_corr=abs(t_hoa*t_sign')/sqrt(t_hoa*t_hoa')/sqrt(t_sign*t_sign');
    corr(freq_ii-1)=temp_corr;
    ld(freq_ii-1)=10*log10(sqrt(t_hoa*t_hoa')/sqrt(t_sign*t_sign'));

end



end