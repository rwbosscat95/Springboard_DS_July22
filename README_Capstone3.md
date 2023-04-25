# CAPSTONE3 - Application of Deep Learning for the Stability Prediction of the Smart Grids
# 1 Introduction
## 1.1 Problem Statement
Decentral Smart Grid Control (**DSGC**) is a new system implementing demand response without significant changes of the power grid infrastructure. It does so by binding the electricity price to the grid frequency, which is one of the most important factors of the power grid stability[1][2]. The theoretical model of the power grid system made some simplifications to monitor and control the operation of the power grids. With the development of data collection and data mining, it is possible to use deep learning methods to predict the stability of a four-node-star electrical grid with a centralized production system. This project will develop a deep learning model for this four-node-star electrical grid with a centralized production that the precision is more than 90% to predict the stability of this power grid.
## 1.2 Dataset
The dataset, created by KIT, contains results from simulations of grid stability for a reference 4-node star power grid. This power grid system is the most common power grid system in the world. The original dataset contains 10,000 observations and includes 12 primary predictive features and two dependent variables. The parameters and status were recorded for about 2 months with a ten-minute interval, which include stable and unstable scenarios.

The original dataset contains 10,000 observations. As the reference grid is symetric, the dataset can be augmented in 3! (3 factorial) times, or 6 times, representing a permutation of the three consumers occupying three consumer nodes. The augmented version has then 60,000 observations. It also contains 12 primary predictive features and two dependent variables.

**Predictive features:**<br />
*'tau1'* to *'tau4'*: the reaction time of each network participant;<br />
*'p1'* to *'p4'*: nominal power produced (positive) or consumed (negative) by each network participant;<br />
*'g1'* to *'g4'*: price elasticity coefficient for each network participant;<br />

**Dependent variables:**<br />
*'stab'*: the maximum real part of the characteristic differentia equation root (if positive, the system is linearly unstable; if negative, linearly stable);<br />
*'stabf'*: a categorical (binary) label ('stable' or 'unstable').<br />

## 1.3 Study object and goal
In a smart grid, consumer demand information is collected, centrally evaluated against current supply conditions and the resulting proposed price information is sent back to customers for them to decide about usage. As the whole process is time-dependent, dinamically estimating **grid stability** becomes not only a concern but a major requirement.

Put simply, the objective is to understand and plan for both energy production and/or consumption disturbances and fluctuations introduced by system participants in a dynamic way, taking into consideration not only technical aspects but also how participants respond to changes in the associated economic aspects (energy price).

The work of researchers cited in foreword focuses on **Decentral Smart Grid Control (DSGC)** systems, a methodology strictly tied to monitoring one particular property of the grid - its frequency.

The term 'frequency' refers to the alternate current (AC) frequency, measured in cycles per second or its equivalent unit, Hertz (Hz). Around the world AC frequencies of either 50 or 60 Hz are utilized in electric power generation-distribution systems.

It is known that the electrical signal frequency "increases in times of excess generation, while it decreases in times of underproduction" [1]. Therefore, **measuring the grid frequency** at the premise of each customer would suffice to provide the network administrator with all required information about the current **network power balance**, so that it can price its energy offering - and inform consumers - accordingly.

The DSGC differential equation-based mathematical model described in [1] and assessed in [2] aims at identifying grid instability for a reference **4-node star** architecture, comprising one power source (a centralized generation node) supplying energy to three consumption nodes. The model takes into consideration inputs (features) related to:

- the total **power balance** (nominal power produced or consumed at each grid node);
- the response time of participants to adjust consumption and/or production in response to price changes (referred to as "**reaction time**");
- energy **price elasticity**.

![hvmW0cg](https://user-images.githubusercontent.com/50253416/234172747-d5212c62-4631-43b6-aebb-8f3df9aa2b03.png)
**Figure 1: 4-node star power grid model**

# 2 Data wrangling
## 2.1 Data Collection
Goal: Read the raw data and investigate the data structure of data frame. 
### 2.1.1 Import the Libraries

```
# Import pandas, matplotlib.pyplot, and seaborn, datetime below
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.layers import Dense

from datetime import datetime
```
### 2.1.2 Load the Raw Data
Read two datasets: one is the original dataset with 10,000 observations and the other is the augmented dataset with 60,000 observations that were permutated (3!=6) by the original dataset.
```
df = pd.read_csv('../data/smart_grid_stability_augmented.csv')
df1 = pd.read_csv('../data/Data_for_UCI_named.csv')
```
### 2.1.3 Check the Raw Data
There are 12 predictive features:

*'tau1'* to *'tau4'*: the reaction time of each network participant. It is a real value within the range 0.5 to 10 (*'tau1'* corresponds to the supplier node, *'tau2'* to *'tau4'* to the consumer nodes) from the results of df.describe() above.

*'p1'* to *'p4'*: nominal power produced (positive) or consumed (negative) by each network participant, a real value within the range -2.0 to -0.5 for consumers (*'p2'* to *'p4'*). As the total power consumed equals the total power generated, p1 (supplier node) = - (p2 + p3 + p4).

*'g1'* to *'g4'*: price elasticity coefficient for each network participant. It is a real value within the range 0.05 to 1.00 (*'g1'* corresponds to the supplier node, *'g2'* to *'g4'* to the consumer nodes; *'g'* stands for *'gamma'*) from the results of df.describe() above. As defined in [1], gamma = c_1 dot c_2

![2023-04-25 003611](https://user-images.githubusercontent.com/50253416/234183570-d50b7e59-9964-4873-a8d2-6e8a6237baae.png)
 
where γi is proportional to the price elasticity of each node i, i.e., measures how much a producer or consumer is willing to adapt their consumption or production. In general, such an adaptation will not be instantaneous but will be delayed by a certain time τ by a measurement and the following reaction.

![figure 2 in notebook 2](https://user-images.githubusercontent.com/50253416/234183713-490fd5f9-14c8-4272-9db7-da3bc7e0bedd.png)
**Figure 2: Using linear relations the power becomes a linear function of the frequency deviation dθi/dt.** (a) We assume a linear price-frequency relation to motivate consumers to stabilize the grid. For example, if the production is larger than consumption, the power grid frequency increases. Hence, decreasing prices should motivate additional consumption. (b) Although consumers might react non-linearly towards price-changes (dark blue), we assume a linear relationship (light green) close to the operational frequency Ω which corresponds to dθi/dt = 0.

There are two dependent variables:

*'stab'*: the maximum real part of the characteristic differentia equation root (if positive, the system is linearly unstable; if negative, linearly stable);

*'stabf'*: a categorical (binary) label ('stable' or 'unstable').

As there is a direct relationship between 'stab' and 'stabf' ('stabf' = 'stable' if 'stab' <= 0, 'unstable' otherwise), 'stab' will be dropped and 'stabf' will remain as the sole dependent variable.
## 2.2 Data Definition
Goal: Gain an understanding of your data features to inform the next steps of EDA.
### 2.2.1 Clean the dataframe
As the dataset content comes from simulation exercises, there are no missing values. Also, all features are originally numerical, no feature coding is required. Such dataset properties allow for a direct jump to machine modeling without the need of data preprocessing or feature engineering.
```
# replace the 'unstable' with '0' and 'stable' with '1' for column stabf
map1 = {'unstable': 0, 'stable': 1}
df['stabf'] = df['stabf'].replace(map1)
df1['stabf'] = df1['stabf'].replace(map1)

# shuffle the data
df = df.sample(frac=1)
df1 = df1.sample(frac=1)
```
### 2.2.2 Variable distribution
Through the codes below, we can see the distributions of the variables that *p1* is a normal distribution between 1 and 6. Other dependent variables are almost even distribution. The predictive variable *stabf* has the ratio of unstable and stable about 2.
```
# plot the histgrams for each variable (12 predictive features and 2 dependent variables) to look at the distribution 
df.hist(figsize=(25,20))
plt.subplots_adjust(hspace=0.5);
```
![download](https://user-images.githubusercontent.com/50253416/234192119-90193b9e-b20c-4946-bb81-3cdfa5f7445a.png)

