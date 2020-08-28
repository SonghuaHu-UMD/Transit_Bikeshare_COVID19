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

# Read data
#dat <- read.csv('D:/Transit/All_final_Transit_R_0810.csv')
#dat <- read.csv('D:/Transit/All_final_Transit_R_0812.csv')
dat <- read.csv('D:/Transit/All_final_Transit_R_0822.csv')
dat$rides <- round(dat$rides)

vif_test <-
  lm(rides ~ COMMERCIAL + # rides
    INDUSTRIAL +
    INSTITUTIONAL +
    OPENSPACE +
    RESIDENTIAL +
    Cumu_Cases +
    Pct.Male +
    Pct.Age_0_24 +
    Pct.Age_25_40 +
    Pct.Age_40_65 +
    Pct.White +
    Pct.Black +
    Income +
    PopDensity +
    College +
    Freq +
    Num_trips +
    Pct.WJob_Utilities +
    Pct.WJob_Goods_Product +
    WTotal_Job_Density,
     data = dat
  )
vif(vif_test)
summary(vif_test)

# PLS
# Cumu_Cases_Rate, Cumu_Death_Rate,
x <- dat %>%
  dplyr::select(COMMERCIAL, INDUSTRIAL, INSTITUTIONAL, OPENSPACE, RESIDENTIAL, LUM,
                Cumu_Cases, Cumu_Death, Pct.Male, Pct.Age_0_24, Pct.Age_25_40, Pct.Age_40_65,
                Pct.White, Pct.Black, Income, College, Pct.WJob_Goods_Product, Pct.WJob_Utilities, WTotal_Job_Density,
                PopDensity, Num_trips, Freq,) %>%
  data.matrix()
## For impact
y <- dat$Relative_Impact
# Find optimal ncomp
PLSR_ <- plsr(y ~ x, ncomp = 10, data = dat, validation = "LOO", scale = F) # method = "oscorespls",
summary(PLSR_)
loading.weights(PLSR_)
png('Figure/NCOM-1.png', units = "in", width = 5, height = 5, res = 600)
ncomp.onesigma <- selectNcomp(PLSR_, method = "onesigma", plot = TRUE)
dev.off()
svg('Figure/NCOM-1.svg', width = 5, height = 5)
ncomp.onesigma <- selectNcomp(PLSR_, method = "onesigma", plot = TRUE)
dev.off()
#ggsave("1-NOCOM.png", units = "in", width = 3.1, height = 3, dpi = 600)

# Check model
plot(PLSR_, ncomp = 4, asp = 1, line = TRUE)
plot(PLSR_, plottype = "scores", comps = 1:6)
plot(PLSR_, "loadings", comps = 1:6, legendpos = "topleft")
explvar(PLSR_)
plot(PLSR_, plottype = "coef", ncomp = 1:6, legendpos = "bottomleft")
plot(PLSR_, plottype = "correlation")
df_coef <- as.data.frame(coef(PLSR_, ncomp = 1:6, intercept = TRUE))
# Calculate p-value
m <- pls(x, y, 4, cv = 10, scale = T)
summary(m)
summary(m$coeffs)

## For ride
y <- dat$rides
PLSR_ <- plsr(y ~ x, ncomp = 10, data = dat, validation = "LOO", scale = F) # method = "oscorespls",
summary(PLSR_)
loading.weights(PLSR_)
png('Figure/NCOM-2.png', units = "in", width = 5, height = 5, res = 600)
ncomp.onesigma <- selectNcomp(PLSR_, method = "onesigma", plot = TRUE)
dev.off()
svg('Figure/NCOM-2.svg', width = 5, height = 5)
ncomp.onesigma <- selectNcomp(PLSR_, method = "onesigma", plot = TRUE)
dev.off()

m <- pls(x, y, 3, cv = 10, scale = T, info = "Shoesize prediction model")
summary(m)
summary(m$coeffs)

# For relative
y <- dat$RELIMP
PLSR_ <- plsr(y ~ x, ncomp = 10, data = dat, validation = "LOO", scale = F) # method = "oscorespls",
ncomp.onesigma <- selectNcomp(PLSR_, method = "onesigma", plot = TRUE)

m <- pls(x, y, 5, cv = 10, scale = T, info = "Shoesize prediction model")
summary(m)
summary(m$coeffs)

# PCA/PLS
# LASSO
cv_fit <- cv.glmnet(x, y, alpha = 1, lambda = lambdas)
plot(cv_fit)
opt_lambda <- cv_fit$lambda.min
# Rebuilding the model with optimal lambda value
best_lasso <- glmnet(x, y, alpha = 1, lambda = cv_fit$lambda.min)
coef(best_lasso)

# Use the selection to build gam
dat$station_id <- as.factor(dat$station_id)
colnames(dat)
GAM_RES1 <-
  mgcv::gam(Relative_Impact ~
              COMMERCIAL +
                Pct.Age_25_40 +
                Pct.Age_40_65 +
                Pct.White +
                Pct.Black +
                Pct.Asian +
                Income +
                PopDensity +
                ti(LAT, LNG),
            data = dat, family = c("gaussian"), method = "REML")
summary(GAM_RES1)

## For rides
y <- dat$rides
# RIDGE
cv_fit <- cv.glmnet(x, y, alpha = 0, lambda = lambdas, family = "poisson")
plot(cv_fit)
opt_lambda <- cv_fit$lambda.min
# Rebuilding the model with optimal lambda value
best_ridge <- glmnet(x, y, alpha = 0, lambda = cv_fit$lambda.min, family = "poisson")
coef(best_ridge)
# LASSO
cv_fit <- cv.glmnet(x, y, alpha = 1, lambda = lambdas, family = "poisson")
plot(cv_fit)
opt_lambda <- cv_fit$lambda.min
# Rebuilding the model with optimal lambda value
best_lasso <- glmnet(x, y, alpha = 1, lambda = cv_fit$lambda.min, family = "poisson")
coef(best_lasso)

# VIF/ALIAS
colnames(dat)
alias(
  lm(
    Relative_Impact ~ Primary +
      Secondary +
      Minor +
      COMMERCIAL +
      INDUSTRIAL +
      INSTITUTIONAL +
      OPENSPACE +
      RESIDENTIAL +
      LUM +
      Cumu_Cases +
      Pct.Male +
      Pct.Age_0_24 +
      Pct.Age_25_40 +
      Pct.Age_40_65 +
      Pct.White +
      Pct.Black +
      Pct.Indian +
      Pct.Asian +
      Pct.Unemploy +
      Income +
      College +
      Pct.Car +
      Pct.Transit +
      Pct.WorkHome +
      PopDensity +
      EmployDensity,
    data = dat
  )
)

vif_test <-
  lm(
    Relative_Impact ~
      COMMERCIAL +
        INDUSTRIAL +
        INSTITUTIONAL +
        OPENSPACE +
        RESIDENTIAL +
        LUM +
        Cumu_Cases +
        Pct.Male +
        Pct.Age_0_24 +
        Pct.Age_25_40 +
        Pct.Age_40_65 +
        Pct.White +
        Pct.Black +
        Pct.Indian +
        Pct.Asian +
        Pct.Unemploy +
        Income +
        College +
        Pct.Car +
        Pct.Transit +
        Pct.WorkHome +
        PopDensity,
    data = dat
  )
vif(vif_test)
summary(vif_test)