pacman::p_load(car, mgcv, psych, dplyr, mgcViz, spdep, sf, tmap, leaps, MASS, ggplot2, metR)

# Read data
dat <- read.csv('D:\\COVID19-Transit_Bikesharing\\Divvy_Data\\All_final_Divvy_R2019_1005.csv')
dat$from_stati <- as.factor(dat$from_stati)
dat$X2019_Avg <- round(dat$X2019_Avg)
dat$Transit.Ridership <- dat$alightings+dat$boardings+dat$rides
dat$Prop.of.Car <- dat$Pct.Car
dat$Prop.of.Transit <- dat$Pct.Transit
colnames(dat)

# Plot interaction
# Control others
Need_Loop <- c("Prop.of.Male","Prop.of.Age_25_40",  "Prop.of.White", "Prop.of.Asian", "Median.Income", "Prop.of.College.Degree",
              "Prop.of.Utilities.Jobs","Prop.of.Goods_Product.Jobs","Population.Density", "Job.Density", "Prop.of.Car","Prop.of.Transit",
               "Prop.of.Commercial","Prop.of.Industrial", "Prop.of.Institutional", "Prop.of.Openspace", "Prop.of.Residential",
               "Road.Density","Bike.Route.Density","Transit.Ridership","Distance.to.Nearest.Bike.Station",
               "Distance.to.City.Center", "Capacity","No.of.Cases","Infection.Rate")
dat_need <- dat[,Need_Loop]
for (var in Need_Loop){
  GAM_RES1 <- mgcv::bam(Cum_Relative_Impact ~ as.matrix(dat_need[,Need_Loop[Need_Loop!=var]])+
    dat$TMAX + dat$PRCP + ti(dat$lat, dat$lon) + s(dat$Week,k=5) + s(dat$from_stati,bs = 're') +
    s(dat$Time_Index, dat[,var]), data = dat, family = c("gaussian"), control = gam.control(trace = TRUE),
                        method = "fREML", discrete = TRUE, chunk.size=5000, nthreads = 150)
  b <- getViz(GAM_RES1,scheme = 1)
  pl <- plot(b,select=4) + l_fitRaster(noiseup = TRUE) + l_fitContour(colour = 7,binwidth = 0.05) + l_rug() +
    geom_text_contour(stroke = 0.2) + labs(x = "Time Index",y=var,title ='') + theme(text = element_text(size=20))
  print(pl, pages = 1)
  ggsave(paste(var,"_Control.png"), units="in", width=7, height=5, dpi=600)
  ggsave(paste(var,"_Control.svg"), units="in", width=7, height=5)
}

# Without Control
Need_Loop <- c("Prop.of.Male","Prop.of.Age_0_24","Prop.of.Age_25_40", "Prop.of.Age_40_65", "Prop.of.White",
               "Prop.of.Black","Prop.of.Indian", "Prop.of.Asian", "Median.Income", "Prop.of.College.Degree",
               "No.of.Cases","Infection.Rate","No.of.Death", "Death.Rate","Prop.of.Commercial","Prop.of.Industrial",
               "Prop.of.Institutional", "Prop.of.Openspace", "Prop.of.Residential", "Primary.Road.Density",
               "Secondary.Road.Density","Minor.Road.Density","Bike.Route.Density", "Prop.of.Goods_Product.Jobs",
               "Prop.of.Utilities.Jobs", "Prop.of.Other.Jobs","Job.Density", "boardings", "alightings",
               "Distance.to.Nearest.Busstop","No.of.Nearby.Rail.Stations", "rides", "Distance.to.Nearest.Rail.Station",
               "No.of.Nearby.Bike.Stations", "Capacity.of.Nearby.Bike.Stations", "Distance.to.Nearest.Bike.Station",
               "Nearby.Bike.Pickups", "Distance.to.City.Center", "Population.Density", "Capacity","Prop.of.Car",
               "Prop.of.Transit","Pct.Bicycle","Pct.Walk", "Pct.WorkHome","Transit.Ridership")
for (var in Need_Loop){
  GAM_RES1 <-
  mgcv::bam(Cum_Relative_Impact ~
    dat$TMAX + dat$PRCP + ti(dat$lat, dat$lon) + s(dat$Week,k=5) + s(dat$from_stati,bs = 're') +
    s(dat$Time_Index, dat[,var]), data = dat, family = c("gaussian"), control = gam.control(trace = TRUE),
            method = "fREML", discrete = TRUE, chunk.size=5000, nthreads = 150)
  b <- getViz(GAM_RES1,scheme = 1)
  pl <- plot(b,select=4) + l_fitRaster(noiseup = TRUE) + l_fitContour(colour = 7,binwidth = 0.05) + l_rug() +
    geom_text_contour(stroke = 0.2) + labs(x = "Time Index",y=var,title ='') + theme(text = element_text(size=20))
  print(pl, pages = 1)
  ggsave(paste(var,"_Single.png"), units="in", width=7, height=5, dpi=600)
  ggsave(paste(var,"_Single.svg"), units="in", width=7, height=5)
}

# Cross Estimation
# Find the last row
dat$from_stati <- as.numeric(dat$from_stati)
dat_avg <- dat %>% group_by(from_stati) %>% slice(c(n())) %>% ungroup()


GAM_RES1 <- mgcv::gam(X2019_Avg ~  Prop.of.Male + Prop.of.Age_25_40  + Prop.of.White + Prop.of.Asian + Median.Income +
  Prop.of.College.Degree + Prop.of.Utilities.Jobs + Prop.of.Goods_Product.Jobs + Population.Density + Job.Density +
  Prop.of.Car + Prop.of.Transit +
  Prop.of.Commercial + Prop.of.Industrial + Prop.of.Institutional + Prop.of.Openspace + Prop.of.Residential +
  Road.Density + Bike.Route.Density  + Transit.Ridership  + Distance.to.Nearest.Bike.Station +  Distance.to.City.Center +
  Capacity + ti(lat, lon), data = dat_avg, family = c("nb"), method = "REML")
summary(GAM_RES1)

GAM_RES2 <- mgcv::gam(Cum_Relative_Impact ~  Prop.of.Male + Prop.of.Age_25_40  + Prop.of.White + Prop.of.Asian + Median.Income +
  Prop.of.College.Degree + Prop.of.Utilities.Jobs + Prop.of.Goods_Product.Jobs + Population.Density + Job.Density +
  Prop.of.Car + Prop.of.Transit +
  Prop.of.Commercial + Prop.of.Industrial + Prop.of.Institutional + Prop.of.Openspace + Prop.of.Residential +
  Road.Density + Bike.Route.Density  + Transit.Ridership  + Distance.to.Nearest.Bike.Station +  Distance.to.City.Center +
  Capacity + No.of.Cases + Infection.Rate + ti(lat, lon), data = dat_avg, family = c("gaussian"), method = "REML")
summary(GAM_RES2)

dat_start <- dat[dat$Date == '2020-03-10',]
GAM_RES3 <- mgcv::gam(Cum_Relative_Impact ~  Prop.of.Male + Prop.of.Age_25_40  + Prop.of.White + Prop.of.Asian + Median.Income +
  Prop.of.College.Degree + Prop.of.Utilities.Jobs + Prop.of.Goods_Product.Jobs + Population.Density + Job.Density +
  Prop.of.Car + Prop.of.Transit +
  Prop.of.Commercial + Prop.of.Industrial + Prop.of.Institutional + Prop.of.Openspace + Prop.of.Residential +
  Road.Density + Bike.Route.Density  + Transit.Ridership  + Distance.to.Nearest.Bike.Station +  Distance.to.City.Center +
  Capacity + No.of.Cases + Infection.Rate + ti(lat, lon), data = dat_start, family = c("gaussian"), method = "REML")
summary(GAM_RES3)

# VIF TEST
vif_test <-
  lm(Cum_Relative_Impact ~ Prop.of.Male + Prop.of.Age_25_40  + Prop.of.White + Prop.of.Asian + Median.Income +
  Prop.of.College.Degree + Prop.of.Utilities.Jobs + Prop.of.Goods_Product.Jobs + Population.Density + Job.Density +
  Pct.Car + Pct.Transit +
  Prop.of.Commercial + Prop.of.Industrial + Prop.of.Institutional + Prop.of.Openspace + Prop.of.Residential +
  Road.Density + Bike.Route.Density  + Transit.Ridership  + Distance.to.Nearest.Bike.Station +  Distance.to.City.Center +
  Capacity + No.of.Cases + Infection.Rate,
     data = dat_avg)
vif(vif_test)
summary(vif_test)

vif_test <-
  lm(X2019_Avg ~ Prop.of.Male + Prop.of.Age_25_40  + Prop.of.White + Prop.of.Asian + Median.Income +
  Prop.of.College.Degree + Prop.of.Utilities.Jobs + Prop.of.Goods_Product.Jobs + Population.Density + Job.Density +
  Pct.Car + Pct.Transit +
  Prop.of.Commercial + Prop.of.Industrial + Prop.of.Institutional + Prop.of.Openspace + Prop.of.Residential +
  Road.Density + Bike.Route.Density  + Transit.Ridership  + Distance.to.Nearest.Bike.Station +  Distance.to.City.Center +
  Capacity, data = dat_avg)
vif(vif_test)
summary(vif_test)

# Stepwise select variables
steps <- step(lm(Cum_Relative_Impact ~ Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White + Pct.Asian +
       Income + College +  Cumu_Cases_Rate + Cumu_Cases  + COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE +
  Bike_Route + Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density +  alightings   +
  rides   + Near_Bike_Capacity + Distance_Bikestation +  Distance_City + PopDensity + capacity,
     data = dat_avg))
summary(steps)

# Use the selected variables to build GAM
GAM_RES1 <- mgcv::gam(Cum_Relative_Impact ~  Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White + Pct.Asian +
       Income + College +  Cumu_Cases_Rate + Cumu_Cases  + COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE +
  Bike_Route + Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density    +
  transit   + Near_Bike_Capacity + Distance_Bikestation +  Distance_City + PopDensity + capacity+
    ti(lat, lon),data = dat_avg, family = c("gaussian"), method = "REML", )
summary(GAM_RES1)

# VIF TEST
vif_test <-
  lm(X2019_Avg ~ Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White + Pct.Black + Pct.Asian +
       Income + College +  Cumu_Cases_Rate + Cumu_Cases + Cumu_Death+Cumu_Death_Rate+COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE + RESIDENTIAL +
    Bike_Route + Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density +  alightings  + Distance_Busstop +
    rides + Distance_Rail  + Near_Bike_Capacity + Distance_Bikestation +  Distance_City + PopDensity + capacity,
     data = dat_avg)
vif(vif_test)
summary(vif_test)

# Stepwise select variables
steps <- step(lm(X2019_Avg ~ Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White + Pct.Asian +
       Income + College + COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE +
  Bike_Route + Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density +  alightings   +
  rides + Distance_Rail  + Near_Bike_Capacity + Distance_Bikestation +  Distance_City + PopDensity + capacity,
     data = dat_avg))
summary(steps)

GAM_RES1 <- mgcv::gam(X2019_Avg ~  Pct.Male + Pct.Age_25_40  + Pct.White + Pct.Asian +
       Income + College + COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE +
  Bike_Route + Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density    +
  transit   + Near_Bike_Capacity + Distance_Bikestation +  Distance_City + PopDensity + capacity+
    ti(lat, lon),data = dat_avg, family = c("nb"), method = "REML", )
summary(GAM_RES1)

GAM_RES1 <-
  mgcv::gam(Response ~  Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White + Pct.Black + Pct.Asian +
       Income + College + Pct.Car + Pct.BikeWalk + Pct.WorkHome +  Cumu_Death +
    COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE + RESIDENTIAL + Bike_Route +
    Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density + Bus_stop_count + boardings  +
    Distance_Busstop + Rail_stop_count + rides + Distance_Rail  + Near_Bike_Capacity +
    Distance_Bikestation + Near_bike_pickups + Distance_City + PopDensity + capacity+
    ti(lat, lon),data = dat_avg, family = c("gaussian"), method = "REML", )
summary(GAM_RES1)

colnames(dat)
# Time_varying
GAM_RES2 <-
  mgcv::bam(Cum_Relative_Impact ~
              Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White + Pct.Asian +
       Income + College +  Cumu_Cases +
    COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE  + Bike_Route +
    Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density +  alightings  +
    Distance_Busstop +  rides + Distance_Rail  + Near_Bike_Capacity +
    Distance_Bikestation +  Distance_City + PopDensity + capacity + TMAX + PRCP +
                ti(lat, lon) + s(Week,k=5) + s(Month,k=5) + s(from_stati,bs = 're') + s(Time_Index),
            data = dat, family = c("gaussian"), control = gam.control(trace = TRUE),
            method = "fREML", discrete = TRUE, chunk.size=5000, nthreads = 150)
summary(GAM_RES2)
plot(GAM_RES2)

GAM_RES2 <-
  mgcv::bam(Response ~
              Pct.Male + Pct.Age_0_24 + Pct.Age_25_40 + Pct.Age_40_65 + Pct.White + Pct.Black + Pct.Asian +
                Income + College  + Cumu_Cases + Cumu_Death + COMMERCIAL + INDUSTRIAL + INSTITUTIONAL + OPENSPACE +
                RESIDENTIAL +  Bike_Route + Pct.WJob_Utilities + Pct.WJob_Goods_Product + WTotal_Job_Density +
                alightings   + rides   + Distance_Bikestation + Distance_City + PopDensity + capacity + TMAX_2020 + PRCP_2020 +
                ti(lat, lon) + s(Week,k=5) + s(Month,k=5) + s(from_stati,bs = 're') + s(Time_Index),
            data = dat, family = c("nb"), control = gam.control(trace = TRUE),
            method = "fREML", discrete = TRUE, chunk.size=5000, nthreads = 150)
summary(GAM_RES2)
plot(GAM_RES2)

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

