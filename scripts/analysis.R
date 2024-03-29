library(dplyr)

landing_all = read.csv('landing.csv', fileEncoding = 'Shift_JIS', header = T)
landing_all[is.na(landing_all)] <- 0
landing = select(landing_all, YEAR='年', COUNT='年間')

landing_all = read.csv('landing.csv', fileEncoding = 'Shift_JIS', header = T)
landing_all[is.na(landing_all)] <- 0
landing.train = select(landing_all, YEAR='年', COUNT='年間')

# 台風発生数と本土への接近数
generation_all = read.csv('generation.csv', fileEncoding = 'Shift_JIS', header = T)
generation = select(generation_all, YEAR='年', generate='年間')
accession_all = read.csv('hondo.csv', fileEncoding = 'Shift_JIS', header = T)
accession = select(accession_all, YEAR='年', access='年間')

# 北太平洋の海面水温平年差の推移
npac = read.csv('npac.csv', header = T)
landing.train = left_join(landing.train, npac, by='YEAR')

landing.train = left_join(landing.train, generation, by='YEAR')
landing.train = left_join(landing.train, accession, by='YEAR')

lm_anomaly = lm(ANOMALY ~ YEAR, npac)
summary(lm_anomaly)
plot(npac$YEAR, npac$ANOMALY)
abline(lm_anomaly)

landing.test = data.frame(
    'YEAR' = 2019,
    'generate' = 22,
    'access' = 8,
    'ANOMALY' = predict(lm_anomaly, data.frame('YEAR'=2019))

)

# modeling
lm = lm(COUNT ~ .^3, landing.train)
lm = step(lm)
summary(lm)

glm = glm(COUNT ~ .^3, landing.train, family = poisson(link = 'log'))
glm = step(glm)
summary(glm)

predict(lm, landing.test)
exp(predict(glm, landing.test))


# read other data
# equator_temp: the equatorial Pacific Ocean characterized by a five consecutive 3-month running mean of sea surface temperature (SST) anomalies
equator_temp = read.table('equator_temp.txt', sep='', header=T)
equator_temp = equator_temp %>%
    group_by(YR) %>%
    summarise(NINO4_mean = mean(NINO4),
              ANOM.2_mean = mean(ANOM.2))

# create test data
landing.test = data.frame(
    'YEAR' = 2019,
    'NINO4_mean' = subset(equator_temp, YR == 2019)['NINO4_mean'],
    'ANOM.2_mean' = subset(equator_temp, YR == 2019)['ANOM.2_mean']
)
equator_temp = equator_temp[equator_temp$YR != 2019, ]

# landing = left_join(landing, npac, by='YEAR')
landing.train = left_join(landing, equator_temp, by=c('YEAR' = 'YR'))

landing.train <- na.omit(landing.train)

# plot
plot(landing.train)

# modeling
lm2 = lm(COUNT ~ ., landing.train)
summary(lm2)

glm2 = glm(COUNT ~ ., landing.train, family = poisson)
summary(glm2)

# predict
predict(lm2, landing.test)
exp(predict(glm2, landing.test))

