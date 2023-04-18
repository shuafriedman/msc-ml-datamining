
#Q1
# ידוע שאקמול עוזר לכאב ראש ב-30 דקות. כדי לקבל רישיון לשווק תרופה חדשה לשוק עליך להראות (0.05=α)
# שהתרופה החדשה יותר טובה. אם התרופה החדשה תעזור לכאב ראש בתוך 25 דקות, אתה רוצה לגלות את זה
# בהסתברות של 0.95 . תכנן 
# ניסיון סטטיסטי שיעמוד בדרישות אם σ=10
#לגלות את זה בהסתברות של 0.95 ז"א עצמת המבחן היא 95%
mean0 = 30
mean1 = 25
alfa = 0.05
beta = 0.05 #beta = 1-power = 1-0.95
sd=10
#חשוב לשים לב האם בהשערת האפס מסתכלים על זנב שמאלי או ימני
#חשוב לשים לב האם בהשערת האחד מסתכלים על זנב שמאלי או ימני
#כאן בגלל שהשערת האפס גדולה מהשערת האחד נתסכל על הזנב השמאלי של השערת האפס , ועל הזנב הימני של השערת האחד 
n =  (sd*(qnorm(alfa,lower.tail = TRUE)-(qnorm(1- beta,lower.tail = TRUE)))/(mean1-mean0))^2 # חשוב לשים לב האם בהשערת האפס מסתכלים על זנב שמאלי או ימני 
cat("size of the test:", n)
Xc = mean0+(qnorm(alfa,lower.tail = TRUE)*sd)/sqrt(n)
cat("Xc (or k) is equal to:", Xc)
####
# You believe that the average cat weighs 7 kl. I believe it weighs more, 
# The standard deviation is known to be = 1.0 kl. 
# How many samples are needed in order to test the hypothesis with α=0.05 and β=0.1 if the true mean is 7.5 kl.
mean0 = 7
mean1 = 7.5
alfa = 0.05
beta = 0.1
sd=1
#חשוב לשים לב האם בהשערת האפס מסתכלים על זנב שמאלי או ימני
#חשוב לשים לב האם בהשערת האחד מסתכלים על זנב שמאלי או ימני
#כאן בגלל שהשערת האפס קטנה מהשערת האחד נתסכל על הזנב הימני של השערת האפס , ועל הזנב השמאלי של השערת האחד 
n =  (sd*(qnorm(alfa,lower.tail = TRUE)-(qnorm(1-beta,lower.tail = TRUE)))/(mean1-mean0))^2
cat("size of the test:", n)
Xc = mean0+(qnorm(alfa,lower.tail = TRUE)*sd)/sqrt👎
cat("Xc (or k) is equal to:", Xc)


#q2
t.test(final1data1$A, final1data1$B)

#q3
x = c(150, 100, 50)
y = c(100, 400, 200)
data = rbind(x, y)
chisq.test(data)
#therefore we reject the null hypothesis that there is no connection between the
## two categorical variables, and we say that the two variables are 
## indeed Dependent (have different distributions)

#q4


stack(final1data2)
summary(aov(values~ind, data = stack(final1data2)))
TukeyHSD(aov(values~ind, data = stack(final1data2)))
#זן ג-זן ב -26.465212 -51.88327 -1.047154 0.0390230

#q5
#PCA?

#Q6
data=iris
head(data)
# install.packages("MASS")
library(MASS)
qda.model=qda(Species~.,data=data)
X = data[,1:4]
y = data[,5]
qda.pred=predict(qda.model, X)
qda.pred$posterior # (probabilities)
qda.pred$class #predicted labels
table(qda.pred$class,y)
mean(qda.pred$class==y)

#Q7

model=lm(mpg~.,data=mtcars)
summary(model)
intercept_only=lm(mpg~1,data=mtcars)
both=step(intercept_only,direction = "both",scope=formula(model),trace = 1)
#wt + cyl + hp with aic of 62.66

##part3: Cyl+ Disp

### Method1
m1 = lm(mpg~., data=mtcars)
m2 = lm(mpg~.-cyl-disp, data=mtcars)
anova(m1, m2)
#Null hypothesis: The variables we removed have no significance.
#Alternative Hypothesis: Those variables are significant.
## If the new model is an improvement of the original model, then we fail to reject H0 (the removed variables don't help)
## If that is not the case, it means that those variables were significant; hence we reject H0 (the removed variables help)

### Method2
summary(m1)
summary(m2)
#The adjusted R squared is higher without the two variables, than it is with the two variables.
 
## https://www.datacamp.com/tutorial/multiple-linear-regression-r-tutorial

###Method 3
ssr_full = sum((fitted(model2) - mean(data$mpg))^2)
ssr_paritally = sum((fitted(model_test) - mean(data$mpg))^2)
mse_full =  sum(residuals(model2)^2)/(nrow(data)-ncol(data))
derduce_features_num = 2
F_statistic = ((ssr_full-ssr_paritally)/derduce_features_num)/mse_full
F_critical = qf(0.95, df1=derduce_features_num, df2 = (nrow(data)-ncol(data))) 
F_statistic
F_critical

#Q8
wilcox.test(final1data1$A, final1data1$B)

