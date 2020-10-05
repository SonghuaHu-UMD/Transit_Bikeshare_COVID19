library(car)
library(mgcv)
library(psych)
library(dplyr)
library(mgcViz)
library(spdep)
library(sf)
library(tmap)
library(leaps)
library(MASS)
library(glmnet)
library(mdatools)
library(pls)
library(ggplot2)
library(metR)

dat <- read.csv('D:\\COVID19-Transit_Bikesharing\\Divvy_Data\\All_final_Divvy_R_BSTS_1003.csv')
dat$Pct.BikeWalk <- dat$Pct.Bicycle + dat$Pct.Walk
colnames(dat)
vif_test <-
  lm(Relative_Impact ~ Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White + Pct.Black + Pct.Asian +
       Income + College + Pct.Car + Pct.BikeWalk + Pct.WorkHome + Cumu_Cases + Cumu_Death +
    COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE + RESIDENTIAL + Primary + Secondary + Minor + Bike_Route +
    Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density + Bus_stop_count + boardings  +
    Distance_Busstop + Rail_stop_count + rides + Distance_Rail  + Near_Bike_Capacity +
    Distance_Bikestation + Near_bike_pickups + Distance_City + PopDensity + capacity,
     data = dat
  )
vif(vif_test)
summary(vif_test)

dat$from_stati <- as.factor(dat$from_stati)
dat$Predict <- round(dat$Predict)
colnames(dat)
# c("gaussian","poisson")
GAM_RES1 <-
  mgcv::bam(Cum_Relt_Effect ~
    ti(lat, lon) + s(Week,k=5) + s(from_stati,bs = 're') +
    s(Time_Index, Pct.White), data = dat, family = c("gaussian"),
            control = gam.control(trace = TRUE),
            method = "fREML", discrete = TRUE, chunk.size=5000, nthreads = 150)
summary(GAM_RES1)
#plot(GAM_RES1,scheme = 2,se=TRUE)
b <- getViz(GAM_RES1,scheme = 1)
pl <- plot(b,select=4) + l_fitRaster(noiseup = TRUE) + l_fitContour(colour = 7,binwidth = 0.05) +
  l_points() + l_rug() + geom_text_contour(stroke = 0.2) + labs(x = "Time Index",y='Pct.of White',title ='')+
      theme(text = element_text(size=20))
print(pl, pages = 1)
ggsave("Fig-4-7.png", units="in", width=7, height=5, dpi=600)
ggsave("Fig-4-7.svg", units="in", width=7, height=5)


# Find the last row
dat$from_stati <- as.numeric(dat$from_stati)
#dat_avg <- aggregate(dat, list(dat$from_stati), min,na.rm = TRUE)
dat_avg <- dat %>%
  group_by(from_stati) %>%
  slice(c(n())) %>%
  ungroup()

vif_test <-
  lm(Cum_Relt_Effect ~ Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White  + Pct.Asian +
       Income + College + Pct.Car + Pct.BikeWalk + Pct.WorkHome + Cumu_Cases + Cumu_Death +
    COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE + RESIDENTIAL + Primary + Secondary + Minor + Bike_Route +
    Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density  + boardings  +
    Distance_Busstop  + rides + Distance_Rail  +
    Distance_Bikestation + Near_bike_pickups + Distance_City + PopDensity + capacity,
     data = dat_avg
  )
vif(vif_test)
summary(vif_test)

steps <- step( lm(Cum_Relt_Effect ~ Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White + Pct.Black + Pct.Asian +
       Income + College + Pct.Car + Pct.BikeWalk + Pct.WorkHome +  Cumu_Death +
    COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE + RESIDENTIAL + Bike_Route +
    Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density + Bus_stop_count + boardings  +
    Distance_Busstop + Rail_stop_count + rides + Distance_Rail  + Near_Bike_Capacity +
    Distance_Bikestation + Near_bike_pickups + Distance_City + PopDensity + capacity,
     data = dat_avg))
summary(steps)

GAM_RES1 <-
  mgcv::gam(Cum_Relt_Effect ~   Pct.White + Pct.Asian + Pct.Car +
    Pct.BikeWalk + Pct.WorkHome + Cumu_Death + INDUSTRIAL + INSTITUTIONAL +
    OPENSPACE + Bus_stop_count + Rail_stop_count + Near_Bike_Capacity +
    Distance_Bikestation + Distance_City + PopDensity + capacity+
    Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density +COMMERCIAL + Bike_Route +
    ti(lat, lon),data = dat_avg, family = c("gaussian"), method = "REML", )
summary(GAM_RES1)

GAM_RES1 <-
  mgcv::gam(Response ~   Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White + Pct.Black + Pct.Asian +
       Income + College + Pct.Car + Pct.BikeWalk + Pct.WorkHome +  Cumu_Death +
    COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE + RESIDENTIAL + Bike_Route +
    Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density + Bus_stop_count + boardings  +
    Distance_Busstop + Rail_stop_count + rides + Distance_Rail  + Near_Bike_Capacity +
    Distance_Bikestation + Near_bike_pickups + Distance_City + PopDensity + capacity+
    ti(lat, lon),data = dat_avg, family = c("gaussian"), method = "REML", )
summary(GAM_RES1)


GAM_RES2 <-
  mgcv::bam(Cum_Relt_Effect ~ Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White + Pct.Black + Pct.Asian +
       Income + College + Pct.Car + Pct.BikeWalk + Pct.WorkHome + Cumu_Cases + Cumu_Death +
    COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE + RESIDENTIAL +  Bike_Route +
    Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density + boardings  + Distance_Busstop +
    rides + Distance_Rail  + Distance_Bikestation + Near_bike_pickups + Distance_City + PopDensity + capacity +
    ti(lat, lon) + s(Week,k=5) + s(Month,k=5) + s(from_stati,bs = 're') + s(Time_Index),
            data = dat, family = c("gaussian"), control = gam.control(trace = TRUE),
            method = "fREML", discrete = TRUE, chunk.size=5000, nthreads = 150)
summary(GAM_RES2)

# How about the average
#dat_avg <- aggregate(dat, list(dat$from_stati), mean,na.rm = TRUE)
dat_avg <- dat <- read.csv('D:\\COVID19-Transit_Bikesharing\\Divvy_Data\\Avg_final_Divvy_R_0907.csv')
dat_avg$Pct.BikeWalk <- dat_avg$Pct.Bicycle + dat_avg$Pct.Walk

vif_test <-
  lm(Relative_Impact ~ Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White  + Pct.Asian +
       Income + College + Pct.Car + Pct.BikeWalk + Pct.WorkHome + Cumu_Cases + Cumu_Death +
    COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE + RESIDENTIAL + Primary + Secondary + Minor + Bike_Route +
    Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density  + boardings  +
    Distance_Busstop  + rides + Distance_Rail  +
    Distance_Bikestation + Near_bike_pickups + Distance_City + PopDensity + capacity,
     data = dat_avg
  )
vif(vif_test)
summary(vif_test)


# LASSO
x <- dat_avg %>%
  dplyr::select(Pct.Male , Pct.Age_0_24 , Pct.Age_25_40 , Pct.Age_40_65 , Pct.White , Pct.Black , Pct.Asian ,
       Income , College , Pct.Car , Pct.BikeWalk , Pct.WorkHome , Cumu_Cases , Cumu_Death ,
    COMMERCIAL , INDUSTRIAL , INSTITUTIONAL , OPENSPACE , RESIDENTIAL , Primary , Secondary , Minor , Bike_Route ,
    Pct.WJob_Utilities , Pct.WJob_Goods_Product , WTotal_Job_Density , Bus_stop_count , boardings  ,
    Distance_Busstop , Rail_stop_count , rides , Distance_Rail  , Near_Bike_Capacity ,
    Distance_Bikestation , Near_bike_pickups , Distance_City , PopDensity , capacity) %>%
  data.matrix()
## For impact
y <- dat_avg$Cum_Relt_Effect
lambdas <- 10^seq(5, -5, by = -.1)
cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = lambdas)
plot(cv_fit)
opt_lambda <- cv_fit$lambda.min
# Rebuilding the model with optimal lambda value
best_lasso <- glmnet(x, y, alpha = 0, lambda = cv_fit$lambda.min)
coef(best_lasso)

