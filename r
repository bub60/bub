Rcodes

# ####################Practical 3

library("TSA")

data(airpass)
plot(decompose(airpass, type = "additive")) #returns additive
plot(decompose(airpass, type = "multiplicative"))  #returns multiplicative

#Moving Average

library(fpp)

data(elecsales)
head(elecsales)

ma3 = filter(elecsales, filter = rep(1/3, 3), method = "convolution", sides = 2)
ma5 = filter(elecsales, filter = rep(1/5, 5), method = "convolution", sides = 2)

plot(elecsales, main = "Elecsales MA")
lines(ma3, lty = 2, lwd = 1, col = "blue")
lines(ma5, lty = 2, lwd = 1, col = "red")
legend("topleft", legend = c("side = 1", "side = 2"), lty = c(1,3))

#Least Square

df = data.frame(hours = c(1,2,4,5,5,6,6,7,8,10,11,11,12,12,14),
                score = c(64, 66, 76, 73, 74,81, 83, 82, 80, 88, 84,82, 91, 93,89 ))

model = lm(score ~ hours, data = df)
summary(model)

plot(df$hours, df$score, pch = 16, col = "steelblue")
abline(model)

data("mtcars")

model1 = lm(mpg ~ hp, data = mtcars)
summary(model1)

plot(mtcars$mpg, mtcars$hp, pch = 16, col = "green")
abline(model)

#Curve Fitting

df1 = data.frame(x = 1:15, y = c(3,14,23,25,23,15,9,5,9,13,17,24,32,36,46))
plot(df1$x , df1$y, pch = 19, xlab = "x", ylab = "y")

fit1 = lm(y ~ x, data = df1)
fit2 = lm(y ~ poly(x, 2, raw = TRUE), data = df1)
fit3 = lm(y ~ poly(x, 3, raw = TRUE), data = df1)
fit4 = lm(y ~ poly(x, 4, raw = TRUE), data = df1)
fit5 = lm(y ~ poly(x, 5, raw = TRUE), data = df1)

x_axis = seq(1, 15, length = 15)

lines(x_axis, predict(fit1, data.frame(x = x_axis)), col = "green")
lines(x_axis, predict(fit2, data.frame(x = x_axis)), col = "red")
lines(x_axis, predict(fit3, data.frame(x = x_axis)), col = "blue")
lines(x_axis, predict(fit4, data.frame(x = x_axis)), col = "orange")
lines(x_axis, predict(fit5, data.frame(x = x_axis)), col = "yellow")

summary(fit1)$adj.r.squared
summary(fit2)$adj.r.squared
summary(fit3)$adj.r.squared
summary(fit4)$adj.r.squared
summary(fit5)$adj.r.squared

summary(fit4)

#Exponential

x = 1:20
y = c(1,3,5,7,9,12,15,19,23,28,33,38,44,50,56,64,73,84,97, 113)

plot(x, y)

model = lm(log(y) ~ x)
summary(model)

# Smoothing

alpha = 0.3

#Initialize the vector for smoothed values
smoothed = numeric(length(x))

#The first smoothed value is the same as first data point
smoothed[1] = x[1]

#Apply the simple Exponential smoothing formula
for (t in 2: length(x)){
  smoothed[t] = alpha * x[t] + (1 - alpha) * smoothed[t-1]
}

print(smoothed)

#---------------------------------------------------------------------------------

# Practical 3: Stationarity

#Creating random data

t = 0:300
y_stationary = rnorm(length(t), mean = 1, sd = 1) #stationary time series
y_trend = cumsum(rnorm(length(t), mean = 1, sd = 4)) + t/100   # time series with tread

#normalize each for simplicity
y_stationary = y_stationary / max(y_stationary)
y_trend = y_trend / max(y_trend)

#Plotting
plot.new()
frame()
par(mfcol = c(2,2))

#the stationary signal and acf
plot(t, y_stationary,
     type = "l", col = "red", xlab = "Time(t)", ylab = "Y(t)",
     main = "Stationary Signal")

acf(y_stationary, lag.max = length(y_stationary),
    xlab = "lag #", ylab = "ACF", main = "ACF")

#the trend signal and acf
plot(t, y_trend,
     type = "l", col = "blue", xlab = "Time(t)", ylab = "Y(t)",
     main = "Trend signal")

acf(y_trend, lag.max = length(y_trend),
    xlab = "lag #", ylab = "ACF", main = "ACf")

#Augmented Dickey-Fuller test
#Null hypothesis: data's mean is not Stationary

library("tseries")

#for y_stationary data
adf.test(y_stationary)

#for y_trend data
adf.test(y_trend)

#Ljung-Box test for independence
#Null hypothesis : a series of residuals is uncorrelated / No autocorrelation

lag.length = 25
Box.test(y_stationary, lag = lag.length, type = "Ljung-Box")

# KPSS test for level or trend stationarity
#Null hypothesis : Data is Stationary

kpss.test(y_stationary, null = "Trend")

#----------------------------------------------------------------------------------

############################Practical 5: Time series Modelling

library(forecast)
library(TTR)

#Q1
data = read.csv("p1s1 (1).csv")
class(data())

#Converting data to time series
TS = ts(data, start = 1963, frequency = 12)
TS

#timeplot
plot(TS)

#identify components
decomposedTS = decompose(TS)

#Check for seasonality
plot(decomposedTS)

#estimate trend
trendestimate = SMA(TS, n = 5)
plot(trendestimate)

trendestimate2 = SMA(TS, n = 8)
plot(trendestimate2)

#removing trend
detrendTS = lm(TS ~ c(1:length(TS)))
detrendTS

plot(resid(detrendTS), type = "l")

#plot 1st difference
firstdf = diff(TS, differences = 1)
firstdf

plot(firstdf, type = "l", main = "First Difference")

#acf and pacf plots
acf(TS)

pacf(TS)

#Q2
data = read.csv("p1s2 (1).csv")
class(data)

#converting to timeseries
TS = ts(data, start = 1948, frequency = 4)
TS

#timeplot
plot(TS)

#identify components
decomposedTS = decompose(TS)

#Check seasonality
plot(decomposedTS)

#Checking if TS is stationary or not
#H0: TS is not stationary
#H1 : TS is stationary

adf.test(TS)

#Plotting 1st difference and check stationarity
firstdiff = diff(TS, differences = 1)
firstdiff

adf.test(firstdiff)

seconddiff = diff(TS, differences = 2)
seconddiff

adf.test(seconddiff)

#plot second difference
plot(seconddiff, type = "l", main = "Second Differencing")

#acf and pacf plots 
#On original data
acf(TS, main = "Income")
pacf(TS)
#on 1st differencing
acf(firstdiff)
pacf(firstdiff)
#on 2nd diff
acf(seconddiff)
pacf(seconddiff)

#fit appropriate model and forecast
model = auto.arima(firstdiff)
model

Forecast = forecast(model)
Forecast
plot(Forecast)

#Q3
#importing stock data
data = read.csv("jindal.csv")
head(data)
class(data)

#Convert to TS data
TS = ts(data$Close, start = 1, frequency = 12)
class(TS)

#Timeplot
plot(TS)

#identify components
decomposedTS = decompose(TS)
decomposedTS

#check seasonality
plot(decomposedTS)

#Check of TS is stationary
adf.test(TS)

#plot 1st difference and check stationarity
firstdiff = diff(TS, differences = 1)
firstdiff

adf.test(firstdiff)

#plot 2nd difference and check stationarity
seconddiff = diff(TS, differences = 2)
seconddiff

adf.test(seconddiff)

#acf and pacf plots
#for original data
acf(TS, main = "Jindal stock price")
pacf(TS)

#first difference
acf(firstdiff)
pacf(firstdiff)

#second difference
acf(seconddiff)
pacf(seconddiff)

#Fit model and forecast
model = auto.arima(seconddiff)
model

Forecast = forecast(model)
Forecast

plot(Forecast)


#####yahoo finance

library("tidyquant")

options("getSymbols.warning4.0" = FALSE)
options("getSymbols.yahoo.warning" = FALSE)
getSymbols("TATAMOTORS.NS",from = "2023-09-01", to = "2024-08-30", warnings = FALSE, auto.assign = TRUE)
head(TATAMOTORS.NS)

class(TATAMOTORS.NS)

# b. Convert it into time series data

TS = ts(TATAMOTORS.NS$TATAMOTORS.NS.Adjusted, start = 1, frequency = 12)

class(TS)



############################################################################


Prac1 

RMSE - (actual-predicted)^2
MAPE-(actual-predicted)/actual

final reults
1)rmse- square root of average 
2)mape- multiply average by 100 

least number will be the best prediction(model)


least square

x range- date 
y- data (meantemp)

formula for prediction is 
intercept+x variable*date


2nd degree parabolic curve 
formula
∑y=na+b∑x+c∑x^2

∑xy=a∑x+b∑x^2+c∑x^3

∑x^2y=a∑x^2+b∑x^3+c∑x^4



pract 2 

Q1

ratio to moving average for 4 yearly 

(actual/predicted )*100

Q2

ratio to trend Quaterly(4 yearly)

columns- average(yt)
t(shift of origin)
t^2
t*yt
ythat

for calculating a 
yt/actual data
for calculating b
summation(tyt/t)

b/4= quarterly for predicting trend values

predicted values in center (anuualy)= a +bx 
add and subtract 1.5 and 3 for Q! Q2 Q3 Q4 accordingly

(actual/predicted )- seasonal indices 
and calculate average of every column

Q3 

take avg of years 
average of average 
then make a column seasonal indices 
formula= actual/total avg*100
highest value highlighted 

Q4



Q6)

ft + 1(first order) = alpha value x baju wala + (1- alpha value) x upar wala

ft +2 = same as above

2ft + 1 - Ft + 2 (second order)

acf . pacf2 find c0 c1 c2 r1 r2

find sum and mean for data
column c0 = (data-mean)^2
column c1 =(data-mean)*(next year data -mean)
column c2=(data-mean)*(alternate(skip 1) year data -mean)
column c3=(data-mean)*(alternate(skip 2 ) year data -mean)

take sum and mean of c0 c1 c2 and c3

now 
r0=mean c0/mean co
r1=mean c1/mean c0
r2=mean c2/mean c0
r3=mean c3/mean c0
