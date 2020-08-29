# need to install package first: install.packages("CausalImpact")
library(CausalImpact)
library(foreach)
library(doParallel)
library(mgcv)
library(car)
library(ggplot2)
library(lattice)
library(spdep)
library(sf)
library(tmap)
library(scales)
library(reshape2)
library(forecast)
library(zoo)
library(bsts)
library(dplyr)

# One time-series
dat <- read.csv('D:\\COVID19-Transit_Bikesharing\\Divvy_Data\\All_Day_count_Divvy.csv') # Daily_Lstaion_Final_0806.csv
dat$startdate <- as.Date(dat$startdate)
rownames(dat) <- NULL
first_enforce_day <- as.numeric(rownames(dat[dat$startdate == as.Date('2020-03-02'),]))
pre.period <- c(1, first_enforce_day - 1)
post.period <- c(first_enforce_day, nrow(dat))
# We keep a copy of the actual observed response in "post.period.response
post.period.response <- dat$trip_id[post.period[1]:post.period[2]]
dat$trip_id[post.period[1]:post.period[2]] <- NA
response <- zoo(dat$trip_id, dat$startdate)
#plot(response)


# Build a bsts model
ss <- AddSemilocalLinearTrend(list(), response)
ss <- AddSeasonal(ss, response, nseasons = 7)
ss <- AddMonthlyAnnualCycle(ss, response)
bsts.model1 <- bsts(
  response ~ PRCP + TMAX + Holidays,
  state.specification = ss, niter = 2000, data = dat, expected.model.size = 2)
#plot(bsts.model1)
#plot(bsts.model1, "components") + scale_x_date(date_breaks = "1 month", labels = date_format("%b-%Y"), limits = as.Date(c('2019-01-01', '2020-05-01')))
#plot(bsts.model1, "coef")
# Estiamting counterfactual and compare with actual post period response
impact <- CausalImpact(
  bsts.model = bsts.model1,
  post.period.response = post.period.response)
plot(impact)
impact.plot <- plot(impact) +
  theme_bw(base_size = 20) +
  scale_x_date(date_breaks = "1 month", labels = date_format("%b-%Y"), limits = as.Date(c('2019-01-01', '2020-08-01')))
plot(impact.plot)
### Get the number of burn-ins to discard
burn <- SuggestBurn(0.1, bsts.model1)
### Get the components
components.withreg <- cbind.data.frame(
  colMeans(bsts.model1$state.contributions[-(1:burn), "trend",]),
  colMeans(bsts.model1$state.contributions[-(1:burn), "seasonal.7.1",]),
  colMeans(bsts.model1$state.contributions[-(1:burn), "Monthly",]),
  colMeans(bsts.model1$state.contributions[-(1:burn), "regression",]),
  as.Date((dat$startdate)))
names(components.withreg) <- c("Trend", "Seasonality", "Monthly", "Regression", "Date")
components.withreg$Predict <- impact$series$point.pred
components.withreg$Response <- impact$series$response
components.withreg$Predict_Lower <- impact$series$point.pred.lower
components.withreg$Predict_Upper <- impact$series$point.pred.upper
components.withreg$Raw_Response <- dat$trip_id
#plot(impact$series$response)
#plot(components.withreg$Raw_Response)
components.withreg <- melt(components.withreg, id.vars = "Date")
names(components.withreg) <- c("Date", "Component", "Value")
# Plot different components
ggplot(data = components.withreg, aes(x = Date, y = Value)) +
  geom_line() +
  theme_bw() +
  theme(legend.title = element_blank()) +
  ylab("") +
  xlab("") +
  facet_grid(Component ~ ., scales = "free") +
  guides(colour = FALSE) +
  scale_x_date(date_breaks = "12 month", labels = date_format("%b-%Y")) +
  theme(axis.text.x = element_text(angle = -30, hjust = 0))
#, limits = as.Date(c('2010-01-01', '2020-05-01'))
# Coefficient
colMeans(bsts.model1$coefficients)
plot(bsts.model1, "coef")
components.withreg$CTNAME <- eachstation


AllCounty <- unique(dat$station_id)
# Setup parallel backend
cores <- detectCores()
cl <- makeCluster(cores[1] - 1)
registerDoParallel(cl)
finalMatrix <- data.frame()

# length(AllCounty)-1800
finalMatrix <-
  foreach(
    ccount = 1:(length(AllCounty)),
    .combine = rbind,
    .packages = c("CausalImpact", "reshape2", "lattice", "ggplot2", "forecast")
  ) %dopar%
  {
    eachstation <- AllCounty[ccount]
    # eachstation <- 40440
    # eachstation <- 40090
    print(ccount)
    dat_Each <- dat[dat$station_id == eachstation,]
    rownames(dat_Each) <- NULL
    first_enforce_day <- as.numeric(rownames(dat_Each[dat_Each$date == as.Date('2020-03-02'),]))
    pre.period <- c(1, first_enforce_day - 1)
    post.period <- c(first_enforce_day, nrow(dat_Each))
    # We keep a copy of the actual observed response in "post.period.response
    post.period.response <- dat_Each$rides[post.period[1]:post.period[2]]
    dat_Each$rides[post.period[1]:post.period[2]] <- NA
    response <- zoo(dat_Each$rides, dat_Each$date)
    #plot(response)
    # drop outliers
    response_cl <- tsclean(response, replace.missing = TRUE, lambda = NULL)
    response_cl[post.period[1]:post.period[2]] <- NA
    response <- zoo(response_cl, dat_Each$date)
    #plot(response1)

    # Build a bsts model
    ss <- AddSemilocalLinearTrend(list(), response)
    ss <- AddSeasonal(ss, response, nseasons = 7)
    ss <- AddMonthlyAnnualCycle(ss, response)
    bsts.model1 <- bsts(
      response ~ PRCP + TMAX + IsWeekend + Holidays,
      state.specification = ss, niter = 2000, data = dat_Each, expected.model.size = 2)
    #plot(bsts.model1)
    #plot(bsts.model1, "components") + scale_x_date(date_breaks = "1 month", labels = date_format("%b-%Y"), limits = as.Date(c('2019-01-01', '2020-05-01')))
    #plot(bsts.model1, "coef")
    # Estiamting counterfactual and compare with actual post period response
    impact <- CausalImpact(
      bsts.model = bsts.model1,
      post.period.response = post.period.response)
    #impact.plot <- plot(impact) +
    #  theme_bw(base_size = 20) +
    #  scale_x_date(date_breaks = "1 month", labels = date_format("%b-%Y"), limits = as.Date(c('2019-01-01', '2020-05-01')))

    #plot(impact.plot)
    ### Get the number of burn-ins to discard
    burn <- SuggestBurn(0.1, bsts.model1)
    ### Get the components
    components.withreg <- cbind.data.frame(
      colMeans(bsts.model1$state.contributions[-(1:burn), "trend",]),
      colMeans(bsts.model1$state.contributions[-(1:burn), "seasonal.7.1",]),
      colMeans(bsts.model1$state.contributions[-(1:burn), "Monthly",]),
      colMeans(bsts.model1$state.contributions[-(1:burn), "regression",]),
      as.Date((dat_Each$date)))
    names(components.withreg) <- c("Trend", "Seasonality", "Monthly", "Regression", "Date")
    components.withreg$Predict <- impact$series$point.pred
    components.withreg$Response <- impact$series$response
    components.withreg$Predict_Lower <- impact$series$point.pred.lower
    components.withreg$Predict_Upper <- impact$series$point.pred.upper
    components.withreg$Raw_Response <- dat_Each$rides
    #plot(impact$series$response)
    #plot(components.withreg$Raw_Response)
    components.withreg <- melt(components.withreg, id.vars = "Date")
    names(components.withreg) <- c("Date", "Component", "Value")
    # Plot different components
    #ggplot(data = components.withreg, aes(x = Date, y = Value)) +
    #  geom_line() +
    #  theme_bw() +
    #  theme(legend.title = element_blank()) +
    #  ylab("") +
    #  xlab("") +
    #  facet_grid(Component ~ ., scales = "free") +
    #  guides(colour = FALSE) +
    #  scale_x_date(date_breaks = "12 month", labels = date_format("%b-%Y")) +
    #  theme(axis.text.x = element_text(angle = -30, hjust = 0))
    #, limits = as.Date(c('2010-01-01', '2020-05-01'))
    # Coefficient
    #colMeans(bsts.model1$coefficients)
    #plot(bsts.model1, "coef")
    components.withreg$CTNAME <- eachstation
    components.withreg
    #write.csv(components.withreg, 'finalMatrix_Transit_temp.csv')
  }

stopCluster(cl)
write.csv(finalMatrix, 'finalMatrix_Transit_0810.csv')

GetInclusionProbabilities <- function(bsts.object) {
  # Pulls code from
  # - BoomSpikeSlab::PlotMarginalInclusionProbabilities
  # - bsts::PlotBstsCoefficients
  burn <- SuggestBurn(0.1, bsts.object)
  beta <- bsts.object$coefficients
  beta <- beta[-(1:burn), , drop = FALSE]
  inclusion.prob <- colMeans(beta != 0)
  index <- order(inclusion.prob)
  inclusion.prob <- inclusion.prob[index]
  # End from BoomSpikeSlab/bsts.
  return(data.frame(predictor = names(inclusion.prob),
                    inclusion.prob = inclusion.prob))
}

# Coeffi
# Setup parallel backend
cores <- detectCores()
cl <- makeCluster(cores[1] - 1)
registerDoParallel(cl)
finalMatrix <- data.frame()


# length(AllCounty)-1800
finalMatrix <-
  foreach(
    ccount = 1:(length(AllCounty)),
    .combine = rbind,
    .packages = c("CausalImpact", "reshape2", "lattice", "ggplot2", "forecast")
  ) %dopar%
  {
    eachstation <- AllCounty[ccount]
    # eachstation <- 40440
    # eachstation <- 40090
    print(ccount)
    dat_Each <- dat[dat$station_id == eachstation,]
    rownames(dat_Each) <- NULL
    first_enforce_day <- as.numeric(rownames(dat_Each[dat_Each$date == as.Date('2020-03-13'),]))
    pre.period <- c(1, first_enforce_day - 1)
    post.period <- c(first_enforce_day, nrow(dat_Each))
    # We keep a copy of the actual observed response in "post.period.response
    post.period.response <- dat_Each$rides[post.period[1]:post.period[2]]
    dat_Each$rides[post.period[1]:post.period[2]] <- NA
    response <- zoo(dat_Each$rides, dat_Each$date)
    #plot(response)
    # drop outliers
    response_cl <- tsclean(response, replace.missing = TRUE, lambda = NULL)
    response_cl[post.period[1]:post.period[2]] <- NA
    response <- zoo(response_cl, dat_Each$date)
    #plot(response1)

    # Build a bsts model
    ss <- AddSemilocalLinearTrend(list(), response)
    ss <- AddSeasonal(ss, response, nseasons = 7)
    ss <- AddMonthlyAnnualCycle(ss, response)
    bsts.model1 <- bsts(
      response ~ PRCP + TMAX + Holidays + IsWeekend,
      state.specification = ss, niter = 2000, data = dat_Each)
    #plot(bsts.model1, "coef")
    # Estiamting counterfactual and compare with actual post period response
    impact <- CausalImpact(
      bsts.model = bsts.model1,
      post.period.response = post.period.response)
    #impact.plot <- plot(impact) +
    #  theme_bw(base_size = 20) +
    #  scale_x_date(date_breaks = "1 month", labels = date_format("%b-%Y"), limits = as.Date(c('2019-01-01', '2020-05-01')))
    #GetInclusionProbabilities(bsts.model1)
    components.withreg <- as.data.frame(bsts.model1$coefficients)
    # Coefficient
    #colMeans(bsts.model1$coefficients)
    #plot(bsts.model1, "coef")
    components.withreg$CTNAME <- eachstation
    components.withreg
    #write.csv(components.withreg, 'finalMatrix_Transit_temp.csv')
  }

stopCluster(cl)
write.csv(finalMatrix, 'finalCoeff_Transit_0810.csv')

# Causal impact

# Setup parallel backend
cores <- detectCores()
cl <- makeCluster(cores[1] - 1)
registerDoParallel(cl)
finalMatrix <- data.frame()
# length(AllCounty)-1800
finalMatrix <-
  foreach(
    ccount = 1:(length(AllCounty)),
    .combine = rbind,
    .packages = c("CausalImpact", "reshape2", "lattice", "ggplot2", "forecast")
  ) %dopar%
  {
    eachstation <- AllCounty[ccount]
    # eachstation <- 40440
    # eachstation <- 40090
    print(ccount)
    dat_Each <- dat[dat$station_id == eachstation,]
    rownames(dat_Each) <- NULL
    first_enforce_day <- as.numeric(rownames(dat_Each[dat_Each$date == as.Date('2020-03-13'),]))
    pre.period <- c(1, first_enforce_day - 1)
    post.period <- c(first_enforce_day, nrow(dat_Each))
    # We keep a copy of the actual observed response in "post.period.response
    post.period.response <- dat_Each$rides[post.period[1]:post.period[2]]
    dat_Each$rides[post.period[1]:post.period[2]] <- NA
    response <- zoo(dat_Each$rides, dat_Each$date)
    #plot(response)
    # drop outliers
    response_cl <- tsclean(response, replace.missing = TRUE, lambda = NULL)
    response_cl[post.period[1]:post.period[2]] <- NA
    response <- zoo(response_cl, dat_Each$date)
    #plot(response1)

    # Build a bsts model
    ss <- AddSemilocalLinearTrend(list(), response)
    ss <- AddSeasonal(ss, response, nseasons = 7)
    ss <- AddMonthlyAnnualCycle(ss, response)
    bsts.model1 <- bsts(
      response ~ PRCP + TMAX + Holidays + IsWeekend,
      state.specification = ss, niter = 2000, data = dat_Each)
    #plot(bsts.model1, "coef")
    # Estiamting counterfactual and compare with actual post period response
    impact <- CausalImpact(
      bsts.model = bsts.model1,
      post.period.response = post.period.response)
    #summary(impact)
    # plot(impact)
    #summary(impact, "report")
    components.withreg <- as.data.frame(impact$summary)
    components.withreg$CTNAME <- eachstation
    components.withreg
    #write.csv(components.withreg, 'finalMatrix_Transit_temp.csv')
  }

stopCluster(cl)
write.csv(finalMatrix, 'finalImpact_Transit_0810_old.csv')

