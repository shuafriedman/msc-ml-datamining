### Linear Regression###
str(regress1)
model1=lm(regress1$grade~regress1$studytime) #linear model of one feature's effect on another
summary(model1)
model1$residuals
model1$fitted.values

### multiple regression###
head(regress2)
str(regress2)
model2=lm(regress2$cost~.,data=regress2) #Cost here is the Y variable (target),
summary(model2)
anova(model2)
new=data.frame(meters=c(100),temp=c(25),floor=c(2),oldest=c(40),roof=c(0),baby=c(0),windows=c(5))
predict(model2,new) #given a model, make a prediction on the new row.
model21=lm(regress2$cost~.-windows,data=regress2)
summary(model21)

### Stepwise regression (aic) - forward### 
# intercept-only model: the formula for the intercept-only model
# direction: the mode of stepwise search, can be either “both”, “backward”, or “forward”
# scope: a formula that specifies which predictors we’d like to attempt to enter into the model
regress2
intercept_only=lm(cost~1,data=regress2)
summary(intercept_only)
model5=step(intercept_only)
forward=step(intercept_only,direction = "forward",scope=formula(model2),trace = 1) #set trace = 0 for shorter results
summary(forward)
anova(forward)

#stepwise regression (aic) - backwards
backwards=step(model2,direction = "backward",scope=formula(model2),trace = 1)
summary(backwards)

#stepwise regression (aic) - both
both=step(intercept_only,direction = "both",scope=formula(model2),trace = 1)
summary(both)

#r square
install.packages("olsrr")
library(olsrr)
model5=lm(cost~.,data=regress2)
summary(model5)
ols_step_best_subset(model5)

#leaps
install.packages("leaps")
library(leaps)
regfit_full=regsubsets(regress2$cost~.,data=regress2)
regsummary=summary(regfit_full)
regsummary

