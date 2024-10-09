---
title: "Distributional Properties of Stock Prices"

## R Markdown


```{r}
# Executive Summary

#Many financial models assume that stock prices are log-normally distributed, which is why we aimed to test the validity of this assumption. First, we visualized the log-transformed stock prices, and then we conducted parametric tests to assess normality. Based on our analysis, we conclude that the logs of the stock prices are not normally distributed.
```
# Read the data

## Loading necessary libraries and extracting data using yahoo finance API
```{r}
library("tidyquant")
options("getSymbols.warning4.0" = FALSE)
options("getSymbols.yahoo.warning" = FALSE)
getSymbols("WELSPUNLIV.NS",from = "2023-09-06", to = "2024-09-06", warnings = FALSE, auto.assign = TRUE)
head(WELSPUNLIV.NS)
```

# Structure of Stock Prices
```{r}
str(WELSPUNLIV.NS)
```

# Descriptive Statistics
```{r}
summary(WELSPUNLIV.NS)
```

# Visualization

##Plot of stock price
```{r}
plot(WELSPUNLIV.NS$WELSPUNLIV.NS.Adjusted, type = "l", main = "Plot of Stock prices of Welspun Living", xlab = "Period from 06/09/2023 to 06/09/2024", ylab = "Stock Prices", col = "blue")
```
#Test Log normality of Stock Prices
#To test log normality we check if the logs of Stock prices are normally distributed.
```{r}
logStockPrices = log(WELSPUNLIV.NS$WELSPUNLIV.NS.Adjusted)
head(logStockPrices)
```
##Histogram
```{r}
hist(logStockPrices, freq = FALSE, breaks = 50, main = "Histogram of Logs of Stock Prices", xlab = "log of Stock Prices", ylab = "Density", col = "grey")
lines(density(logStockPrices), col = "red", lwd = 2)
```

##QQplot
```{r}
library("car")
qqnorm(logStockPrices, col = "red", lwd = 4)
qqline(logStockPrices, col = "blue", lwd = 2)

```
# Parametric Test

## Jarque-Berra test for Normality
#the null hypothesis of the JB test is that the data is normal
```{r}
library("tseries")
jarque.bera.test(logStockPrices)
```
#As the p-value is more than 0.05, we conclude that the logs of Stock prices are normally distributed

## Anderson Darling Test for Normality
#the null hypothesis of the Anderson Darling test is that data is normal
```{r}
library(nortest)
ad.test(logStockPrices)
```
#the p-value of Anderson Darling test is less than 0.05, hence the log of stock price are not normally distributed

#Conclusion

##Based on visualization

# 1. Histogram
The histogram shows that the log of stock prices has a somewhat bimodal distribution, with two peaks around values 5.0 and 5.2.
The red density line indicates that the distribution is not strictly normal, with some visible skewness and multiple modes, which further supports the rejection of normality.
There are tails on both ends, suggesting potential outliers or a heavier-tailed distribution than normal.
# 2. QQ Plot
The Q-Q plot compares the sample quantiles of the log stock prices with the theoretical quantiles of a normal distribution.
The points deviate from the straight blue line, especially at the lower and upper extremes, indicating deviation from normality.
The "S" shape in the Q-Q plot further suggests that the log stock prices are not normally distributed, particularly in the tails, where both the lower and upper quantiles do not follow the expected normal behavior.

##Based on parametric tests

# 1.Jarque-Bera Test:
The JB test is used to evaluate whether the sample has skewness and kurtosis matching that of a normal distribution. The test statistic is based on the sample's third and fourth moments. In this case, the JB test produced a p-value of 0.4236, which is greater than 0.05. This means that there is not enough evidence to reject the null hypothesis of normality. Therefore, based on the JB test, the log of the stock prices does not significantly deviate from normality.

# 2. Anderson-Darling Test:
The Anderson-Darling test is another test for normality, focusing on the tails of the distribution. Here, the test resulted in a very low p-value (< 2.2e-16), strongly rejecting the null hypothesis of normality. This indicates that the log of the stock prices is not normally distributed, especially with more focus on the extreme values or tails of the distribution.

# Summary of conclusion
The time-series plot of the log-transformed stock prices shows fluctuations and trends, indicating non-stationary behavior. Visual inspection suggests that the stock price movements might not follow a simple pattern, which could impact the distribution's normality.
Based on the Jarque-Bera test, the log-transformed stock prices exhibit normal skewness and kurtosis, implying no major deviation from normality in these specific characteristics. However, the Anderson-Darling test shows significant departures from normality, especially in the tails of the distribution. This suggests that while the overall shape of the distribution might appear normal, extreme stock price movements deviate from a normal distribution.
```
####################################################################################################################################################################################



---
title: "Distributional Properties of Stock Returns"

## R Markdown

```{r}
# Executive Summary

#Many financial models assume that stock returns are log-normally distributed, which is why we aimed to test the validity of this assumption. First, we visualized the log-transformed stock prices, and then we conducted parametric tests to assess normality. Based on our analysis, we conclude that the logs of the stock prices are **not normally distributed.
```
# Read the data

## Loading necessary libraries and extracting data using yahoo finance API
```{r}
library("tidyquant")

options("getSymbols.warning4.0" = FALSE)
options("getSymbols.yahoo.warning" = FALSE)
getSymbols("VMART.NS",from = "2023-09-01", to = "2024-09-30", warnings = FALSE, auto.assign = TRUE)
head(VMART.NS)

```
# Calculating Simple Returns of the adjusted close of daily stock prices.

```{r}
adjusted_prices <- VMART.NS$VMART.NS.Adjusted

# Calculating simple returns
#simple_returns <- (adjusted_prices / lag(adjusted_prices)) - 1
n = length(adjusted_prices)
# Removing NA values from the first row
simple_returns <- na.omit((as.numeric(adjusted_prices[-1])/as.numeric(adjusted_prices[-n]))-1)

head(simple_returns)

```
# Structure of Stock Returns
```{r}
str(simple_returns)

```
# Descriptive Statistics
```{r}
summary(simple_returns)

```
# Visualization

##Plot of stock returns
```{r}
plot(simple_returns, type = "l", main = "Plot of Stock returns of V-Mart Retail Ltd", xlab = "Period from 01/09/2023 to 30/09/2024", ylab = "Stock Returns", col = "blue")

```

#Test Log normality of Stock Returns
#To test log normality, we check if the logs of Stock returns are normally distributed.
```{r}
positive_returns <- simple_returns[simple_returns > 0]

# Applying the log transformation to positive returns
logStockReturns <- log(positive_returns)

head(logStockReturns)
```

##Histogram
```{r}
hist(logStockReturns, freq = FALSE, breaks = 50, main = "Histogram of Logs of Stock Returns", xlab = "log of Stock returns", ylab = "Density", col = "grey")
lines(density(logStockReturns), col = "red", lwd = 2)

```


##QQplot
```{r}
library("car")
qqnorm(logStockReturns, col = "red", lwd = 4)
qqline(logStockReturns, col = "blue", lwd = 2)

```

# Parametric Test

## Jarque-Berra test for Normality
#the null hypothesis of the JB test is that the data is normal
```{r}
library("tseries")
jarque.bera.test(logStockReturns)

```
#As the p-value is less than 0.05, we conclude that the logs of Stock prices are not normally distributed

## Anderson Darling Test for Normality
#the null hypothesis of the Anderson Darling test is that data is normal
```{r}
library(nortest)
ad.test(logStockReturns)

```
#the p-value of Anderson Darling test is less than 0.05, hence the log of stock price are not normally distributed

# Conclusion

## Based on visualization

# 1. Histogram

The histogram shows that the log of stock prices has a somewhat bimodal distribution,
with two peaks around values 5.0 and 5.2. The red density line indicates that the
distribution is not strictly normal, with some visible skewness and multiple modes, which
further supports the rejection of normality. There are tails on both ends, suggesting
potential outliers or a heavier-tailed distribution than normal.

# 2. QQ Plot

The Q-Q plot compares the sample quantiles of the log stock prices with the theoretical
quantiles of a normal distribution. The points deviate from the straight blue line, especially
at the lower and upper extremes, indicating deviation from normality. The “S” shape in the
Q-Q plot further suggests that the log stock prices are not normally distributed, particularly
in the tails, where both the lower and upper quantiles do not follow the expected normal
behavior.

## Based on parametric tests

# Jarque-Bera Test:

The JB test is used to evaluate whether the sample has skewness and kurtosis matching
that of a normal distribution. The test statistic is based on the sample’s third and fourth
moments. In this case, the JB test produced a p-value of 0.4236, which is greater than 0.05.
This means that there is not enough evidence to reject the null hypothesis of normality.
Therefore, based on the JB test, the log of the stock prices does not significantly deviate
from normality.

# Anderson-Darling Test:

The Anderson-Darling test is more sensitive to deviations from normality, particularly in the tails of the distribution.
Here, the test provides a p-value less than 2.2e-16, which is much smaller than 0.05, strongly rejecting the null hypothesis of normality. This indicates that the log of stock returns is not normally distributed, especially when considering extreme values or the tails of the distribution.

# Summary of Conclusion:
The time-series plot of the log-transformed stock prices shows fluctuations and trends,
indicating non-stationary behavior. Visual inspection suggests that the stock price
movements might not follow a simple pattern, which could impact the distribution’s
normality. Based on the Jarque-Bera test, the log-transformed stock prices exhibit normal
skewness and kurtosis, implying no major deviation from normality in these specific
characteristics. However, the Anderson-Darling test shows significant departures from
normality, especially in the tails of the distribution. This suggests that while the overall
shape of the distribution might appear normal, extreme stock price movements deviate
from a normal distribution.

################################################################################################################################################################################################



---
title: "Financial Time Series for Stock Prices"

## R Markdown

# Executive Summary

The analysis aimed to examine the stock price behavior of Can Fin Homes Ltd (CANFINHOME.NS) from September 1, 2023, to September 12, 2024. Using graphical techniques (stock price plot and autocorrelation function plot) and parametric tests (Augmented Dickey-Fuller for stationarity and Ljung-Box for autocorrelation), the stock prices were found to be non-stationary with significant autocorrelation. Despite applying various transformations, including first-order differencing and multiple-order differencing, the stock prices remained non-stationary. The Ljung-Box test confirmed autocorrelation with a p-value < 2.2e-16. Overall, the results indicate that the stock prices cannot be effectively modeled using stationary assumptions, and further advanced time series techniques are necessary to address these characteristics.

# Read the data

## Loading necessary libraries and extracting data using yahoo finance API

```{r}
library("tidyquant")
library("aTSA")

options("getSymbols.warning4.0" = FALSE)
options("getSymbols.yahoo.warning" = FALSE)
getSymbols("CANFINHOME.NS",from = "2023-09-01", to = "2024-09-12", warnings = FALSE, auto.assign = TRUE)
head(CANFINHOME.NS)

```
# Structure 

```{r}
str(CANFINHOME.NS)

```

# Descriptive Statistics

```{r}
summary(CANFINHOME.NS)

```
# Visualization

## Plot of stock price

```{r}
plot(CANFINHOME.NS$CANFINHOME.NS.Adjusted, type = "l", main = "Plot of Stock prices of Can Fin Home", xlab = "Period from 01/09/2023 to 12/09/2024", ylab = "Stock Prices", col = "blue")

lines(density(CANFINHOME.NS$CANFINHOME.NS.Adjusted), col = "red", lwd = 2)

```

## Plot of AutoCorrelation

```{r}
acf(CANFINHOME.NS$CANFINHOME.NS.Adjusted, main = "Auto Correlation of Can Fin Homes Ltd")

```

# Parametric Test for Stationarity

Null hypothesis : Data is not Stationary
```{r}
stationary.test(CANFINHOME.NS$CANFINHOME.NS.Adjusted)

```
The augmented Dickey-Fuller (ADF) test assume that the null hypothesis is that the series has a unit root and is therefore non-stationary. 
Time Series is not stationary, hence null hypothesis cannot be rejected since p-value > 0.05

# Parametric Test for Autocorrelation

The Ljung-Box test is a hypothesis test that determines if a time series contains autocorrelation. 

Null hypothesis: The residuals are independently distributed/ Data is uncorrelated.

```{r}
library(tseries)
Box.test(CANFINHOME.NS$CANFINHOME.NS.Adjusted, type = "Ljung-Box")

```
Since p-value < 2.2e-16, we reject the null hypothesis. Hence autocorrelation is present

# Transformation of Stock Prices

```{r}
transformedStockPrice = na.omit(diff(CANFINHOME.NS$CANFINHOME.NS.Adjusted))

```
# Visualization of transformed Stock Prices

## Plot of transformed stock price

```{r}
plot(transformedStockPrice, type = "l", main = "Plot of transformed Stock prices of Can Fin Home", xlab = "Period from 01/09/2023 to 12/09/2024", ylab = "Stock Prices", col = "blue")

```

## Plot of AutoCorrelation

```{r}

acf(transformedStockPrice, main = "Auto Correlation of Can Fin Homes Ltd")

```

# Parametric Test for Stationarity

Null hypothesis : Data is not Stationary
```{r}
stationary.test(transformedStockPrice)

```
Since data is still not stationary, transforming the data again

```{r}
transformedStockPrice1 = na.omit(diff(diff(CANFINHOME.NS$CANFINHOME.NS.Adjusted)))

stationary.test(transformedStockPrice1)

```

```{r}
transformedStockPrices = diff(CANFINHOME.NS$CANFINHOME.NS.Adjusted, differences = 10)
stationary.test(transformedStockPrices)

```

# Conclusion:

Graphical Analysis: 
The stock price plot revealed a clear trend over time, indicating that the stock prices were likely non-stationary. The ACF plot further suggested significant autocorrelation in the data.

Stationarity Test: 
The ADF test consistently failed to reject the null hypothesis, confirming that the stock prices were non-stationary. Even after applying transformations (first-order differencing, second-order differencing, and up to the 10th order), the series remained non-stationary.

Autocorrelation Test: 
The Ljung-Box test resulted in a p-value of less than 2.2e-16, leading to the rejection of the null hypothesis. This indicates the presence of significant autocorrelation in the stock prices.

# Summary of Conclusion: 

The analysis concludes that the stock prices of Can Fin Homes Ltd exhibit non-stationary behavior, even after multiple transformations. Additionally, there is significant autocorrelation present in the data. These findings suggest that the stock prices cannot be modeled effectively using simple stationary models, and further techniques, such as advanced time series modeling, may be required to address the non-stationarity and autocorrelation present in the data.


#############################################################################################################################################################################################



---
title: "Financial Time series for Stock Returns (Autocorrelation)"

## R Markdown

# Executive Summary

The analysis aimed to examine the stock return behavior of Can Fin Homes Ltd (CANFINHOME.NS) from September 1, 2023, to September 12, 2024. Using graphical techniques (stock price plot and autocorrelation function plot) and parametric tests (Augmented Dickey-Fuller for stationarity and Ljung-Box for autocorrelation), the stock returns were found to be non-stationary with significant autocorrelation. Despite applying various transformations, including first-order differencing and multiple-order differencing, the stock returns remained non-stationary. The Ljung-Box test confirmed autocorrelation with a p-value = 0.03032. Overall, the results indicate that the stock returns cannot be effectively modeled using stationary assumptions, and further advanced time series techniques are necessary to address these characteristics.

# Read the data

## Loading necessary libraries and extracting data using yahoo finance API

```{r}
library("tidyquant")
library("aTSA")

options("getSymbols.warning4.0" = FALSE)
options("getSymbols.yahoo.warning" = FALSE)
getSymbols("CANFINHOME.NS",from = "2023-09-01", to = "2024-09-12", warnings = FALSE, auto.assign = TRUE)
head(CANFINHOME.NS)

```

#Calculating Stock Returns

```{r}
StockReturns = na.omit(diff(CANFINHOME.NS$CANFINHOME.NS.Adjusted)/CANFINHOME.NS$CANFINHOME.NS.Adjusted)

```
# Structure 

```{r}
str(StockReturns)

```
# Descriptive Statistics

```{r}
summary(StockReturns)

```
# Visualization

## Plot of stock return

```{r}
plot(StockReturns, type = "l", main = "Plot of Stock returns of Can Fin Home", xlab = "Period from 01/09/2023 to 12/09/2024", ylab = "Stock Prices", col = "blue")

```
## Plot of AutoCorrelation

```{r}
acf(StockReturns, main = "Auto Correlation of Can Fin Homes Ltd")

```
# Parametric Test for Stationarity

Null hypothesis : Data is not Stationary
```{r}
stationary.test(StockReturns)

```
The augmented Dickey-Fuller (ADF) test assume that the null hypothesis is that the series has a unit root and is therefore non-stationary. 
Time Series is not stationary, hence null hypothesis cannot be rejected since p-value > 0.05

# Parametric Test for Autocorrelation
The Ljung-Box test is a hypothesis test that determines if a time series contains autocorrelation. 

Null hypothesis: The residuals are independently distributed/ Data is uncorrelated.

```{r}
Box.test(StockReturns, type = "Ljung-Box")

```
Since p-value > 0.05, we reject the null hypothesis. Hence autocorrelation is present

# Transformation of Stock Returns

```{r}
transformedStockReturns = na.omit(diff(StockReturns))

```
# Visualization of transformed Stock Returns

## Plot of stock returns

```{r}
plot(transformedStockReturns, type = "l", main = "Plot of transformed Stock returns of Can Fin Home", xlab = "Period from 01/09/2023 to 12/09/2024", ylab = "Stock Prices", col = "blue")


```

## Plot of AutoCorrelation of First Differences of Stock Returns

```{r}

acf(transformedStockReturns, main = "Auto Correlation of Can Fin Homes Ltd")

```
# Parametric Test for Stationarity

Null hypothesis : Data is not Stationary
```{r}
stationary.test(transformedStockReturns)

```


```{r}
TransformedStockReturns = diff(transformedStockReturns, differences = 10)
stationary.test(TransformedStockReturns)

```
```{r}
Box.test(TransformedStockReturns, type = "Ljung-Box")

```

# Conclusion:

The time series analysis of Can Fin Homes Ltd stock returns from September 1, 2023, to September 12, 2024, indicates that the returns exhibit non-stationary behavior, as evidenced by the results of the Augmented Dickey-Fuller (ADF) test. The p-values consistently showed that we fail to reject the null hypothesis of non-stationarity. Additionally, the Ljung-Box test confirmed the presence of autocorrelation in the stock returns with a p-value of less than 0.05, suggesting that the data is not independently distributed.

Despite attempts at multiple transformations, including first-order and higher-order differencing, the stock returns remained non-stationary. This suggests that standard ARIMA modeling techniques may not be suitable for effectively modeling these returns. Advanced time series methods, such as GARCH models or other volatility models, may be necessary to better capture the underlying characteristics of the stock returns and provide more accurate forecasts.


##########################################################################################################################################################################################



---
title: "ARIMA Modelling"

## R Markdown
# Executive Summary

This report details the ARIMA modeling process for analyzing and forecasting the stock prices of Can Fin Homes Limited over the period from September 1, 2022, to September 30, 2023. Using Yahoo Finance data and various time series analysis tools in R, the study identifies key trends, tests for stationarity, and applies transformations where necessary to achieve stationarity for accurate ARIMA model specification.

Initially, the Augmented Dickey-Fuller (ADF) test indicated non-stationarity in the stock price data, as the p-values were high, leading us to fail to reject the null hypothesis. This prompted the transformation of the data through first differencing. After differencing, the data was tested again for stationarity, with the results still indicating non-stationarity, despite the transformation.

An ARIMA (AutoRegressive Integrated Moving Average) model was fitted using auto.arima(). The diagnostic evaluation, including residual checks with the Ljung-Box test, confirmed that the residuals were uncorrelated, indicating that the model was appropriate for forecasting purposes. Finally, stock price forecasting was conducted, providing insights into future trends with confidence intervals.

## Loading necessary libraries and extracting data using yahoo finance API

```{r}
library("tidyquant")
library("tseries")
library("forecast")
library("aTSA")

options("getSymbols.warning4.0" = FALSE)
options("getSymbols.yahoo.warning" = FALSE)
getSymbols("CANFINHOME.NS",from = "2022-09-01", to = "2023-09-30", warnings = FALSE, auto.assign = TRUE)
head(CANFINHOME.NS)

```
# Structure 

```{r}
str(CANFINHOME.NS)

```

# Descriptive Statistics

```{r}
summary(CANFINHOME.NS)

```
# Exploring Stock Prices

```{r}
plot(CANFINHOME.NS$CANFINHOME.NS.Adjusted, type = "l", main = "Plot of Stock prices of Can Fin Home", xlab = "Period from 01/09/2022 to 30/09/2023", ylab = "Stock Prices", col = "blue")

```

# ACF

```{r}
acf(CANFINHOME.NS$CANFINHOME.NS.Adjusted)

```

# Augmented Dickey Fuller Test

Null Hypothesis: Time series has unit roots ie. data is not stationary
```{r}
adf.test(CANFINHOME.NS$CANFINHOME.NS.Adjusted)

```
In all cases, the p-values are very high, meaning we fail to reject the null hypothesis of non-stationarity in all variations of the test (no drift, with drift, and with trend). This indicates that the time series data is likely non-stationary.


# Transform Stock Prices to make it Stationary

```{r}
StockPrices = na.omit(diff(CANFINHOME.NS$CANFINHOME.NS.Adjusted, differences = 1))

```
# Plotting transformed Stock Prices

```{r}
plot(StockPrices, type = "l", main = "Plot of  transformed Stock prices of Can Fin Home", xlab = "Period from 01/09/2022 to 30/09/2023", ylab = "Stock Prices", col = "blue")

``` 

#ACF

```{r}
acf(StockPrices)

```

# Augmented Dickey-Fuller Test on transformed data

Null Hypothesis: Time series has unit roots ie. data is not stationary
```{r}
adf.test(StockPrices)

```
In all cases, the p-values are very high, meaning we fail to reject the null hypothesis of non-stationarity in all variations of the test (no drift, with drift, and with trend). This indicates that the time series data is likely non-stationary.


# Model Specification

```{r}
auto.arima(StockPrices)

```

#Estimation of Coefficients

```{r}
StockPricesCoeff = arima(StockPrices, order = c(0,0,0))
StockPricesRes = residuals(StockPricesCoeff)
tsdiag(StockPricesCoeff)
plot(StockPricesRes, type = "l")

```

# Diagnose the Model

## Using Ljung-Box test
Null Hypothesis: Time Series is not autocorrelated
```{r}
Box.test(StockPricesRes, lag = 10, type = "Ljung-Box")

```
Since the p-value (0.9229) is much greater than common significance levels (e.g., 0.05 or 0.01), we fail to reject the null hypothesis.

# Forecasting

```{r}
library(forecast)
StockPriceForecast = forecast(StockPricesCoeff)

plot(StockPriceForecast)

```

# Conclusion

The ARIMA model developed for Can Fin Homes Limited's stock prices demonstrated its suitability based on residual diagnostics. Despite initial non-stationarity, the transformation and differencing steps provided a model that could be reliably used for short-term forecasting. Future work could involve refining the model further by exploring other transformation techniques or adding external variables to improve accuracy. The model's residuals showed no significant autocorrelation, reinforcing the robustness of the model for prediction purposes.

###############################################################################################################################################################################################



---
title: "Portfolio Analysis"

## R Markdown

# Executive Summary

This portfolio analysis focuses on four stocks: BF Utilities, Adani Enterprises, Coal India, and Himatsingka Seide, using historical stock prices from September 1, 2022, to September 30, 2023. The study examines key performance indicators such as returns, standard deviations, and correlations between the stocks to assess the portfolio’s risk and return. The analysis calculates the portfolio's risk (standard deviation) and expected returns, providing valuable insights for portfolio management.

The descriptive statistics for the individual stocks reveal significant variability in stock returns, with mean returns ranging from -0.0012 for Adani Enterprises to 0.0022 for BF Utilities. The calculated variance-covariance matrix enables us to understand the correlation between the stocks and their joint impact on overall portfolio risk.

Using equal weights (25% allocation for each stock), the portfolio's annualized risk (standard deviation) is 30.85%, and the expected annualized return is 38.76%. These metrics provide a basis for evaluating the performance and volatility of this diversified portfolio.

# Reading the Data
```{r}
shhh = suppressPackageStartupMessages
shhh(library(quantmod))
library("tidyquant")
library("aTSA")

options("getSymbols.warning4.0" = FALSE)
options("getSymbols.yahoo.warning" = FALSE)
getSymbols("BFUTILITIE.NS",from = "2022-09-01", to = "2023-09-30", warnings = FALSE, auto.assign = TRUE)
getSymbols("ADANIENT.NS",from = "2022-09-01", to = "2023-09-30", warnings = FALSE, auto.assign = TRUE)
getSymbols("COALINDIA.NS",from = "2022-09-01", to = "2023-09-30", warnings = FALSE, auto.assign = TRUE)
getSymbols("HIMATSEIDE.NS",from = "2022-09-01", to = "2023-09-30", warnings = FALSE, auto.assign = TRUE)

```

# Calculation of Stock Returns and Descriptive Statistics 

## BF Utilities (BFUTILITIE.NS)
```{r}
StockPrice_BFUtil = na.omit(BFUTILITIE.NS$BFUTILITIE.NS.Adjusted)
len_BFUtil = length(StockPrice_BFUtil)
len_BFUtil

Returns_BFUtil = na.omit((as.numeric(StockPrice_BFUtil[-1]/
                                       as.numeric(StockPrice_BFUtil[-len_BFUtil])))-1)

# Descriptive Stats
mean_BFUtil = mean(Returns_BFUtil)
sd_BFUtil = sd(Returns_BFUtil)
max_BFUtil = max(Returns_BFUtil)
min_BFUtil = min(Returns_BFUtil)

```
## BF Utilities Plot
```{r}
par(bg = "grey")

plot.ts(Returns_BFUtil, type = "l", main = "BF Utilities Plot",
        xlab = "Time", ylab = "Stock Returns", col = "blue")
abline(h = mean_BFUtil, col = "red", lwd = 2, lty = 3)

```

## Adani Enterprises (ADANIENT.NS)
```{r}
StockPrice_Adani = na.omit(ADANIENT.NS$ADANIENT.NS.Adjusted)
len_Adani = length(StockPrice_Adani)
len_Adani

Returns_Adani = na.omit((as.numeric(StockPrice_Adani[-1] / as.numeric(StockPrice_Adani[-len_Adani])))-1)

# Descriptive Stats
mean_Adani = mean(Returns_Adani)
sd_Adani = sd(Returns_Adani)
max_Adani = max(Returns_Adani)
min_Adani = min(Returns_Adani)

```

## Adani Enterprise Plot
```{r}
par(bg = "grey")

plot.ts(Returns_Adani, type = "l", main = "Adani Enterprise Plot",
        xlab = "Time", ylab = "Stock Returns", col = "blue")
abline(h = mean_Adani, col = "red", lwd = 2, lty = 3)

```

## Coal India (COALINDIA.NS)
```{r}
StockPrice_Coal = na.omit(COALINDIA.NS$COALINDIA.NS.Adjusted)
len_Coal = length(StockPrice_Coal)
len_Coal

Returns_Coal = na.omit((as.numeric(StockPrice_Coal[-1] / as.numeric(StockPrice_Coal[-len_Coal])))-1)

# Descriptive Stats
mean_Coal = mean(Returns_Coal)
sd_Coal = sd(Returns_Coal)
max_Coal = max(Returns_Coal)
min_Coal = min(Returns_Coal)

```
## Coal India Plot
```{r}
par(bg = "grey")

plot.ts(Returns_Coal, type = "l", main = "Coal India Plot",
        xlab = "Time", ylab = "Stock Returns", col = "blue")
abline(h = mean_Coal, col = "red", lwd = 2, lty = 3)

```

## Himatsingka Seide (HIMATSEIDE.NS)
```{r}
StockPrice_Himat = na.omit(HIMATSEIDE.NS$HIMATSEIDE.NS.Adjusted)
len_Himat = length(StockPrice_Himat)
len_Himat

Returns_Himat = na.omit((as.numeric(StockPrice_Himat[-1] / as.numeric(StockPrice_Himat[-len_Himat])))-1)

#Descriptive Stats
mean_Himat = mean(Returns_Himat)
sd_Himat = sd(Returns_Himat)
max_Himat = max(Returns_Himat)
min_Himat = min(Returns_Himat)

```
## Himatsingka Seide Plot
```{r}
par(bg = "grey")

plot.ts(Returns_Himat, type = "l", main = "Himatsingka Seide Plot",
        xlab = "Time", ylab = "Stock Returns", col = "blue")
abline(h = mean_Himat, col = "red", lwd = 2, lty = 3)

```

## Creating a dataframe
```{r}
MyPortfolio = data.frame(Returns_BFUtil, Returns_Adani, Returns_Coal, Returns_Himat)
head(MyPortfolio)

colnames(MyPortfolio) = c("BF_Utilities", "Adani_Enterprise", "Coal_India", "Himatsingka_Seide")

```

# Plotting Stock-wise Correlation
```{r}
pairs(MyPortfolio, labels = colnames(MyPortfolio), pch = 20, col = "blue")

```

## Descriptive Statistics dataframe
```{r}
DescStats <- data.frame(
  Stocks = c("BF_Utilities", "Adani_Enterprise", "Coal_India", "Himatsingka_Seide"),
  Mean = NA , Standard_Dev = NA, Max = NA, Min = NA, Length = NA)

DescStats$Mean =  c(mean_BFUtil, mean_Adani, mean_Coal, mean_Himat)
DescStats$Standard_Dev = c(sd_BFUtil, sd_Adani, sd_Coal, sd_Himat)
DescStats$Max = c(max_BFUtil, max_Adani, max_Coal, max_Himat)
DescStats$Min = c(min_BFUtil, min_Adani, min_Coal, min_Himat)
DescStats$Length = c(len_BFUtil, len_Adani, len_Coal, len_Himat)

head(DescStats)

```

# Preparing a Variance Covariance Matrix
```{r}
VarCovMat = matrix(cov(MyPortfolio), nrow = 4)
VarCovMat

```

## Assigning weights
```{r}
weights = matrix(c(0.25, 0.25, 0.25, 0.25), nrow = 4)

```

# Calculating Portfolio Risk
```{r}
PortfolioRisk = ((t(weights)%*% VarCovMat)%*% weights)^0.5
PortfolioRisk

```

# Calculation of Annual Portfolio Risk
```{r}
AnnualPortfolioRisk = PortfolioRisk * sqrt(269)
AnnualPortfolioRisk

print(paste("Portfolio Standard Deviation (Risk) : ",round(AnnualPortfolioRisk*100, digits = 2),"%"))
```

# Calculating Portfolio Returns
```{r}
Returns = matrix(c(mean_BFUtil,mean_Adani, mean_Coal, mean_Himat), nrow = 4)

PortfolioReturn = t(weights)%*% Returns
PortfolioReturn

AnnualPortfolioReturns = ((PortfolioReturn + 1)^len_Adani)-1
AnnualPortfolioReturns

print(paste("Portfolio Returns : ", round(AnnualPortfolioReturns*100, digits = 2),"%"))

```

# Conclusion

The portfolio consisting of BF Utilities, Adani Enterprises, Coal India, and Himatsingka Seide demonstrates a strong expected annual return of 38.76%. However, the associated risk, as reflected by the standard deviation of 30.85%, indicates considerable volatility. Investors should balance this high return potential with the elevated risk levels. Diversifying further or adjusting the portfolio’s allocation may help reduce risk while maintaining returns.

In summary, while the portfolio offers a compelling return, risk management strategies should be considered to align with investor risk tolerance, especially given the observed volatility across the included stocks.
