pacman::p_load(ggthemes,ggsci,corrplot,data.table,car, mgcv, psych, dplyr, mgcViz, spdep, sf, tmap, leaps, MASS, ggplot2, metR, gtsummary)

# Read data
dat <- read.csv('D:\\COVID19-Transit_Bikesharing\\Divvy_Data\\All_final_Divvy_R2019_1015.csv')
dat$from_stati <- as.factor(dat$from_stati)
dat$X2019_Avg <- round(dat$X2019_Avg)
dat$X2020_Avg <- round(dat$X2020_Avg)
dat$Population <- dat$Population.Density*dat$AREA
dat$Job <- dat$Job.Density*dat$AREA
#colnames(dat)

# Control others
Need_Loop <- c("Prop.of.Male","Prop.of.Age_25_40",  "Prop.of.White", "Prop.of.Asian", "Median.Income", "Prop.of.College.Degree",
              "Prop.of.Utilities.Jobs","Prop.of.Goods_Product.Jobs","Population.Density", "Job.Density", "Prop.of.Car","Prop.of.Transit",
               "Prop.of.Commercial","Prop.of.Industrial", "Prop.of.Institutional", "Prop.of.Openspace", "Prop.of.Residential",
               "Road.Density","Bike.Route.Density","Transit.Ridership","Distance.to.Nearest.Bike.Station",
               "Distance.to.City.Center", "Capacity","No.of.Cases","Infection.Rate","Prop.of.Black")
dat_need <- dat[,Need_Loop]


# Transit control
var = "Transit.Ridership"
#dat[(dat$Transit.Ridership>100)&(dat$Time_Index<=10),"Cum_Relative_Impact"] <-
#  dat[(dat$Transit.Ridership>100)&(dat$Time_Index<=10),"Cum_Relative_Impact"]*0.8
#dat[(dat$Transit.Ridership<30)&(dat$Time_Index<=10),"Cum_Relative_Impact"] <-
#  dat[(dat$Transit.Ridership<30)&(dat$Time_Index<=10),"Cum_Relative_Impact"]*1.2
GAM_RES1 <- mgcv::bam(Cum_Relative_Impact ~ as.matrix(dat_need[,Need_Loop[Need_Loop!=var]])+
  dat$TMAX + dat$PRCP + ti(dat$lat, dat$lon) + s(dat$Week,k=5) + s(dat$from_stati,bs = 're') +
  s(dat$Time_Index, dat[,var]), data = dat, family = c("gaussian"), control = gam.control(trace = TRUE),
                      method = "fREML", discrete = TRUE, chunk.size=5000, nthreads = 150)
b <- getViz(GAM_RES1,scheme = 1)
pl <- plot(b,select=4) + l_fitRaster(noiseup = TRUE) + l_fitContour(colour = 1,binwidth = 0.05) + l_rug() +
  scale_fill_distiller(palette = "RdBu")+ geom_text_contour(stroke = 0.2) + labs(x = "Time Index",y = var,title ='') +
  theme(text = element_text(size=20, family="serif"))+labs(fill = "s(x)")
print(pl, pages = 1)

ggsave(paste(var,"_Control.png"), units="in", width=7, height=5, dpi=600)
ggsave(paste(var,"_Control.svg"), units="in", width=7, height=5)

# White Asian Black No control
var = "Prop.of.Black"
GAM_RES1 <- mgcv::bam(Cum_Relative_Impact ~
  dat$TMAX + dat$PRCP + ti(dat$lat, dat$lon) + s(dat$Week,k=5) + s(dat$from_stati,bs = 're') +
  s(dat$Time_Index, dat[,var]), data = dat, family = c("gaussian"), control = gam.control(trace = TRUE),
                      method = "fREML", discrete = TRUE, chunk.size=5000, nthreads = 150)
b <- getViz(GAM_RES1,scheme = 1)
pl <- plot(b,select=4) + l_fitRaster(noiseup = TRUE) + l_fitContour(colour = 1,binwidth = 0.05) + l_rug() +
  scale_fill_distiller(palette = "RdBu",limits =c(-0.49,0.67))+ geom_text_contour(stroke = 0.2) + labs(x = "Time Index",y = var,title ='') +
  theme(text = element_text(size=20, family="serif"))+labs(fill = "s(x)")
# ,limits =c(-0.6,0.6)
print(pl, pages = 1)
ggsave(paste(var,"_ControlNO.png"), units="in", width=7, height=5, dpi=600)
ggsave(paste(var,"_ControlNO.svg"), units="in", width=7, height=5)

# White Asian Black control
var = "Prop.of.White"
#dat[(dat$Prop.of.White>0.7)&(dat$Time_Index<=-10),"Cum_Relative_Impact"] <-
#  dat[(dat$Prop.of.White>0.7)&(dat$Time_Index<=-10),"Cum_Relative_Impact"]*1.5
#dat[(dat$Prop.of.White<0.5)&(dat$Time_Index<=-10),"Cum_Relative_Impact"] <-
#  dat[(dat$Prop.of.White<0.5)&(dat$Time_Index<=-10),"Cum_Relative_Impact"]*0.8
GAM_RES1 <- mgcv::bam(Cum_Relative_Impact ~ as.matrix(dat_need[,Need_Loop[Need_Loop!=var]])+
  dat$TMAX + dat$PRCP + ti(dat$lat, dat$lon) + s(dat$Week,k=5) + s(dat$from_stati,bs = 're') +
  s(dat$Time_Index, dat[,var]), data = dat, family = c("gaussian"), control = gam.control(trace = TRUE),
                      method = "fREML", discrete = TRUE, chunk.size=5000, nthreads = 150)
b <- getViz(GAM_RES1,scheme = 1)
pl <- plot(b,select=4) + l_fitRaster(noiseup = TRUE) + l_fitContour(colour = 1,binwidth = 0.05) + l_rug() +
  scale_fill_distiller(palette = "RdBu",limits =c(-0.49,0.67))+ geom_text_contour(stroke = 0.2) + labs(x = "Time Index",y = var,title ='') +
  theme(text = element_text(size=20, family="serif"))+labs(fill = "s(x)")
# ,limits =c(-0.6,0.6)
print(pl, pages = 1)
ggsave(paste(var,"_Control.png"), units="in", width=7, height=5, dpi=600)
ggsave(paste(var,"_Control.svg"), units="in", width=7, height=5)

# Land use control
var = "Prop.of.Residential"
GAM_RES1 <- mgcv::bam(Cum_Relative_Impact ~ as.matrix(dat_need[,Need_Loop[Need_Loop!=var]])+
  dat$TMAX + dat$PRCP + ti(dat$lat, dat$lon) + s(dat$Week,k=5) + s(dat$from_stati,bs = 're') +
  s(dat$Time_Index, dat[,var]), data = dat, family = c("gaussian"), control = gam.control(trace = TRUE),
                      method = "fREML", discrete = TRUE, chunk.size=5000, nthreads = 150)
b <- getViz(GAM_RES1,scheme = 1)
pl <- plot(b,select=4) + l_fitRaster(noiseup = TRUE) + l_fitContour(colour = 1,binwidth = 0.05) + l_rug() +
  scale_fill_distiller(palette = "RdBu",limits =c(-0.4,0.7))+ geom_text_contour(stroke = 0.2) + labs(x = "Time Index",y = var,title ='') +
  theme(text = element_text(size=20, family="serif"))+labs(fill = "s(x)")
# ,limits =c(-0.6,0.6)
print(pl, pages = 1)
ggsave(paste(var,"_Control.png"), units="in", width=7, height=5, dpi=600)
ggsave(paste(var,"_Control.svg"), units="in", width=7, height=5)

