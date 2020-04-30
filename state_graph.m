j=0;
k=1;
for i=1:size(states,2)
    if sum(states{i})==0
        j=j+1;
        i=i+1;
        k=1;
    end
    s_plot(j,k)=states{i}(1);
    k=k+1;
end
s_plot_reduced=s_plot;

remove=[];
for i=1:size(s_plot_reduced,1)
    if s_plot_reduced(i,151)==0
        remove=[remove,i]
    end
end



%s_plot_reduced(remove,:)=[];
display(length(remove));

s_plot_reduced=s_plot_reduced(size(s_plot_reduced,1)-50:end,:);

s_bar(1)=sum(sum(s_plot_reduced<-0.1));
s_bar(2)=sum(sum(s_plot_reduced<0))- s_bar(1)   ;
s_bar(3)=sum(sum(s_plot_reduced<0.1)) - s_bar(1) - s_bar(2) -sum(sum(s_plot_reduced==0));
s_bar(4)=sum(sum(s_plot_reduced>0.1));

figure
plot(s_plot_reduced')
grid()

figure
bar(s_bar/sum(s_bar)*100)
ylabel('Percentage of time [%]')
xticklabels({'x<-0.1','-0.1 \leq x < 0',' 0 \leq x < 0.1',' x > 0.1'})
x0=100;
y0=100;
width=550;
height=250;
set(gcf,'position',[x0,y0,width,height])

figure()
stdshade(s_plot_reduced,0.2,[0.6350, 0.0780, 0.1840]);
xlim([1,size(s_plot_reduced,2)-1])
xticks([1,26,51,76,101,126,151,176,201])
xticklabels({0,25,50,75,100,125,150,175,200})
xlabel('Time (s)')
ylabel('x [m]')
ylim([-0.2,0.2001])
x0=100;
y0=100;
width=550;
height=250;
set(gcf,'position',[x0,y0,width,height])
grid()

reward=0;
for j=1:size(s_plot_reduced,1)
    for i=1:151
        x=s_plot_reduced(j,i);
        reward=reward+(-0.5+2*(x<0))*(i<=100)+(-0.5+4.5*(x>0))*(i>100);
    end
end
reward=reward/(size(s_plot_reduced,1))

ylim([-0.2,1])
xline(101,'--k')
xline(156,'.-b')
    