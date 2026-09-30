# Preserved R scratch code; requires the original R-session objects.
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
