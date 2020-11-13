
# Importing all necessary libraries


```python
import pandas as pd
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

```

# Loading The Dataset


```python
dataset= pd.read_csv(r"C:\Users\nEW u\Desktop\all shortcuts\ALL\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 5 - Multiple Linear Regression\Multiple_Linear_Regression\data.csv")
dataset.head(50)
#dataset.drop(dataset[dataset.is_goal== 'NaN'].index, axis=0, inplace=True)
dataset

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>match_event_id</th>
      <th>location_x</th>
      <th>location_y</th>
      <th>remaining_min</th>
      <th>power_of_shot</th>
      <th>knockout_match</th>
      <th>game_season</th>
      <th>remaining_sec</th>
      <th>distance_of_shot</th>
      <th>...</th>
      <th>lat/lng</th>
      <th>type_of_shot</th>
      <th>type_of_combined_shot</th>
      <th>match_id</th>
      <th>team_id</th>
      <th>remaining_min.1</th>
      <th>power_of_shot.1</th>
      <th>knockout_match.1</th>
      <th>remaining_sec.1</th>
      <th>distance_of_shot.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>10.0</td>
      <td>167.0</td>
      <td>72.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>27.0</td>
      <td>38.0</td>
      <td>...</td>
      <td>45.539131, -122.651648</td>
      <td>shot - 30</td>
      <td>NaN</td>
      <td>20000012</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>50.608</td>
      <td>54.2000</td>
      <td>38.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>12.0</td>
      <td>-157.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>22.0</td>
      <td>35.0</td>
      <td>...</td>
      <td>45.539131, -122.651648</td>
      <td>shot - 45</td>
      <td>NaN</td>
      <td>20000012</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>28.800</td>
      <td>22.0000</td>
      <td>35.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>35.0</td>
      <td>-101.0</td>
      <td>135.0</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>45.0</td>
      <td>36.0</td>
      <td>...</td>
      <td>45.539131, -122.651648</td>
      <td>shot - 25</td>
      <td>NaN</td>
      <td>20000012</td>
      <td>1610612747</td>
      <td>92.64</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>63.7216</td>
      <td>54.400</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>43.0</td>
      <td>138.0</td>
      <td>175.0</td>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>52.0</td>
      <td>42.0</td>
      <td>...</td>
      <td>45.539131, -122.651648</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>20000012</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>122.608</td>
      <td>52.0000</td>
      <td>42.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>155.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>19.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>45.539131, -122.651648</td>
      <td>NaN</td>
      <td>shot - 1</td>
      <td>20000012</td>
      <td>1610612747</td>
      <td>42.64</td>
      <td>2.00</td>
      <td>0.000</td>
      <td>19.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>244.0</td>
      <td>-145.0</td>
      <td>-11.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>...</td>
      <td>45.539131, -122.651648</td>
      <td>shot - 17</td>
      <td>NaN</td>
      <td>20000012</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>34.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>251.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>52.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>45.539131, -122.651648</td>
      <td>NaN</td>
      <td>shot - 4</td>
      <td>20000012</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>112.2000</td>
      <td>89.400</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>254.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>8.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>...</td>
      <td>45.539131, -122.651648</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>20000012</td>
      <td>1610612747</td>
      <td>68.64</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>5.0000</td>
      <td>22.000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>265.0</td>
      <td>-65.0</td>
      <td>NaN</td>
      <td>6.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>12.0</td>
      <td>32.0</td>
      <td>...</td>
      <td>45.539131, -122.651648</td>
      <td>shot - 36</td>
      <td>NaN</td>
      <td>20000012</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>12.0000</td>
      <td>32.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>294.0</td>
      <td>-33.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>36.0</td>
      <td>32.0</td>
      <td>...</td>
      <td>45.539131, -122.651648</td>
      <td>shot - 44</td>
      <td>NaN</td>
      <td>20000012</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>52.2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>309.0</td>
      <td>-94.0</td>
      <td>238.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>56.0</td>
      <td>45.0</td>
      <td>...</td>
      <td>45.539131, -122.651648</td>
      <td>shot - 7</td>
      <td>NaN</td>
      <td>20000012</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>3.00</td>
      <td>80.928</td>
      <td>56.0000</td>
      <td>45.000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>4.0</td>
      <td>121.0</td>
      <td>127.0</td>
      <td>11.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>0.0</td>
      <td>37.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>1.00</td>
      <td>106.608</td>
      <td>64.7856</td>
      <td>16.400</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>27.0</td>
      <td>-67.0</td>
      <td>NaN</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 44</td>
      <td>NaN</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>77.36</td>
      <td>0.000</td>
      <td>9.0000</td>
      <td>32.000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>66.0</td>
      <td>-94.0</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>44.0</td>
      <td>29.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>68.36</td>
      <td>0.000</td>
      <td>44.0000</td>
      <td>29.000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>80.0</td>
      <td>-23.0</td>
      <td>47.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>16.0</td>
      <td>25.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 12</td>
      <td>NaN</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>16.0000</td>
      <td>25.000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>86.0</td>
      <td>NaN</td>
      <td>192.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>NaN</td>
      <td>40.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>94.64</td>
      <td>70.36</td>
      <td>0.000</td>
      <td>70.7856</td>
      <td>40.000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 4</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>1.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>138.0</td>
      <td>-117.0</td>
      <td>226.0</td>
      <td>8.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>50.0</td>
      <td>45.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 6</td>
      <td>NaN</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>38.64</td>
      <td>2.00</td>
      <td>0.000</td>
      <td>50.0000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>244.0</td>
      <td>NaN</td>
      <td>97.0</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>29.0</td>
      <td>36.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>29.0000</td>
      <td>106.400</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>249.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>46.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 4</td>
      <td>NaN</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>70.36</td>
      <td>0.000</td>
      <td>46.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20</td>
      <td>255.0</td>
      <td>3.0</td>
      <td>144.0</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>8.0</td>
      <td>34.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 20</td>
      <td>NaN</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>8.0000</td>
      <td>34.000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21</td>
      <td>265.0</td>
      <td>134.0</td>
      <td>127.0</td>
      <td>9.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>38.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 24</td>
      <td>NaN</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>43.36</td>
      <td>0.000</td>
      <td>4.0000</td>
      <td>38.000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>274.0</td>
      <td>-16.0</td>
      <td>110.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>57.0</td>
      <td>31.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 44</td>
      <td>NaN</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>57.0000</td>
      <td>31.000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>299.0</td>
      <td>-109.0</td>
      <td>150.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>47.0</td>
      <td>38.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>47.0000</td>
      <td>38.000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24</td>
      <td>307.0</td>
      <td>-46.0</td>
      <td>63.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>11.0</td>
      <td>27.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>25.2000</td>
      <td>27.000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>25</td>
      <td>332.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>36.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 4</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>93.64</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>36.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>26</td>
      <td>345.0</td>
      <td>NaN</td>
      <td>196.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>4.0</td>
      <td>40.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 30</td>
      <td>NaN</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>83.36</td>
      <td>0.000</td>
      <td>63.7856</td>
      <td>40.000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>27</td>
      <td>369.0</td>
      <td>-183.0</td>
      <td>186.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>30.0</td>
      <td>46.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 54</td>
      <td>NaN</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>30.0000</td>
      <td>53.400</td>
    </tr>
    <tr>
      <th>28</th>
      <td>28</td>
      <td>400.0</td>
      <td>85.0</td>
      <td>173.0</td>
      <td>8.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>NaN</td>
      <td>39.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>4.00</td>
      <td>61.928</td>
      <td>19.0000</td>
      <td>18.400</td>
    </tr>
    <tr>
      <th>29</th>
      <td>29</td>
      <td>429.0</td>
      <td>3.0</td>
      <td>87.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>2000-01</td>
      <td>22.0</td>
      <td>28.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 42</td>
      <td>NaN</td>
      <td>20000019</td>
      <td>1610612747</td>
      <td>73.64</td>
      <td>4.00</td>
      <td>0.000</td>
      <td>22.0000</td>
      <td>28.000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30667</th>
      <td>30667</td>
      <td>368.0</td>
      <td>40.0</td>
      <td>250.0</td>
      <td>9.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>33.0</td>
      <td>45.0</td>
      <td>...</td>
      <td>40.361408, -86.186052</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900087</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>33.0000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30668</th>
      <td>30668</td>
      <td>386.0</td>
      <td>-23.0</td>
      <td>222.0</td>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>27.0</td>
      <td>42.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900087</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>68.36</td>
      <td>54.608</td>
      <td>27.0000</td>
      <td>42.000</td>
    </tr>
    <tr>
      <th>30669</th>
      <td>30669</td>
      <td>425.0</td>
      <td>171.0</td>
      <td>53.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>48.0</td>
      <td>37.0</td>
      <td>...</td>
      <td>40.361408, -86.186052</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900087</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>112.7856</td>
      <td>37.000</td>
    </tr>
    <tr>
      <th>30670</th>
      <td>30670</td>
      <td>NaN</td>
      <td>-74.0</td>
      <td>16.0</td>
      <td>9.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>34.0</td>
      <td>27.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>38.64</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>34.0000</td>
      <td>27.000</td>
    </tr>
    <tr>
      <th>30671</th>
      <td>30671</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>51.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 4</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>102.36</td>
      <td>1.000</td>
      <td>51.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>30672</th>
      <td>30672</td>
      <td>29.0</td>
      <td>89.0</td>
      <td>55.0</td>
      <td>8.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>9.0</td>
      <td>30.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 33</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>109.64</td>
      <td>29.36</td>
      <td>1.000</td>
      <td>9.0000</td>
      <td>30.000</td>
    </tr>
    <tr>
      <th>30673</th>
      <td>30673</td>
      <td>NaN</td>
      <td>117.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>14.0</td>
      <td>31.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>63.36</td>
      <td>1.000</td>
      <td>14.0000</td>
      <td>31.000</td>
    </tr>
    <tr>
      <th>30674</th>
      <td>30674</td>
      <td>81.0</td>
      <td>117.0</td>
      <td>216.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>34.0</td>
      <td>44.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>34.0000</td>
      <td>44.000</td>
    </tr>
    <tr>
      <th>30675</th>
      <td>30675</td>
      <td>84.0</td>
      <td>-134.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>34.928</td>
      <td>0.0000</td>
      <td>45.000</td>
    </tr>
    <tr>
      <th>30676</th>
      <td>30676</td>
      <td>98.0</td>
      <td>-141.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 44</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>34.000</td>
    </tr>
    <tr>
      <th>30677</th>
      <td>30677</td>
      <td>101.0</td>
      <td>NaN</td>
      <td>143.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>1.0</td>
      <td>38.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>38.000</td>
    </tr>
    <tr>
      <th>30678</th>
      <td>30678</td>
      <td>181.0</td>
      <td>14.0</td>
      <td>16.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>1999-00</td>
      <td>34.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 4</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>29.36</td>
      <td>1.000</td>
      <td>34.0000</td>
      <td>22.000</td>
    </tr>
    <tr>
      <th>30679</th>
      <td>30679</td>
      <td>212.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>41.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 39</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>71.64</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>41.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>30680</th>
      <td>30680</td>
      <td>213.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>40.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 49</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>121.928</td>
      <td>40.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>30681</th>
      <td>30681</td>
      <td>218.0</td>
      <td>-18.0</td>
      <td>261.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>2.0</td>
      <td>46.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 17</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>68.64</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>2.0000</td>
      <td>46.000</td>
    </tr>
    <tr>
      <th>30682</th>
      <td>30682</td>
      <td>226.0</td>
      <td>NaN</td>
      <td>48.0</td>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>30.0</td>
      <td>28.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 44</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>41.36</td>
      <td>87.608</td>
      <td>30.0000</td>
      <td>28.000</td>
    </tr>
    <tr>
      <th>30683</th>
      <td>30683</td>
      <td>228.0</td>
      <td>1.0</td>
      <td>216.0</td>
      <td>10.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>51.0</td>
      <td>41.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 38</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>51.0000</td>
      <td>41.000</td>
    </tr>
    <tr>
      <th>30684</th>
      <td>30684</td>
      <td>231.0</td>
      <td>-96.0</td>
      <td>89.0</td>
      <td>10.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>16.0</td>
      <td>33.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 37</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>50.64</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>16.0000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30685</th>
      <td>30685</td>
      <td>249.0</td>
      <td>81.0</td>
      <td>250.0</td>
      <td>7.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>1999-00</td>
      <td>31.0</td>
      <td>46.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>44.608</td>
      <td>31.0000</td>
      <td>46.000</td>
    </tr>
    <tr>
      <th>30686</th>
      <td>30686</td>
      <td>268.0</td>
      <td>16.0</td>
      <td>93.0</td>
      <td>5.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>37.0</td>
      <td>29.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 44</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>78.400</td>
    </tr>
    <tr>
      <th>30687</th>
      <td>30687</td>
      <td>284.0</td>
      <td>40.0</td>
      <td>100.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 54</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>18.0000</td>
      <td>30.000</td>
    </tr>
    <tr>
      <th>30688</th>
      <td>30688</td>
      <td>308.0</td>
      <td>-126.0</td>
      <td>61.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>7.0</td>
      <td>33.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>37.64</td>
      <td>NaN</td>
      <td>1.000</td>
      <td>7.0000</td>
      <td>16.400</td>
    </tr>
    <tr>
      <th>30689</th>
      <td>30689</td>
      <td>326.0</td>
      <td>-12.0</td>
      <td>679.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>0.0</td>
      <td>87.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>31.60</td>
      <td>1.000</td>
      <td>0.0000</td>
      <td>45.728</td>
    </tr>
    <tr>
      <th>30690</th>
      <td>30690</td>
      <td>331.0</td>
      <td>-113.0</td>
      <td>100.0</td>
      <td>11.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>37.0</td>
      <td>35.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>23.36</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>35.000</td>
    </tr>
    <tr>
      <th>30691</th>
      <td>30691</td>
      <td>382.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 4</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>4.00</td>
      <td>27.800</td>
      <td>59.7856</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>30692</th>
      <td>30692</td>
      <td>397.0</td>
      <td>1.0</td>
      <td>48.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>5.0</td>
      <td>24.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 1</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>17.20</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>5.0000</td>
      <td>24.000</td>
    </tr>
    <tr>
      <th>30693</th>
      <td>30693</td>
      <td>398.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>5.0</td>
      <td>20.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 49</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>64.36</td>
      <td>1.000</td>
      <td>5.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>30694</th>
      <td>30694</td>
      <td>426.0</td>
      <td>-134.0</td>
      <td>166.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>28.0</td>
      <td>41.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>shot - 3</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>28.0000</td>
      <td>41.000</td>
    </tr>
    <tr>
      <th>30695</th>
      <td>30695</td>
      <td>448.0</td>
      <td>31.0</td>
      <td>267.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>10.0</td>
      <td>46.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 26</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>112.36</td>
      <td>1.000</td>
      <td>10.0000</td>
      <td>46.000</td>
    </tr>
    <tr>
      <th>30696</th>
      <td>30696</td>
      <td>471.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>1999-00</td>
      <td>39.0</td>
      <td>27.0</td>
      <td>...</td>
      <td>42.982923, -71.446094</td>
      <td>shot - 45</td>
      <td>NaN</td>
      <td>49900088</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>39.0000</td>
      <td>27.000</td>
    </tr>
  </tbody>
</table>
<p>30697 rows × 28 columns</p>
</div>



# Removing Unnecessary Columns


```python
dataset.drop(['match_event_id','location_x','location_y','game_season','shot_id_number','lat/lng','type_of_combined_shot','type_of_shot','match_id',], axis=1, inplace=True)
dataset.drop(['shot_basics','team_name','date_of_game','home/away',],axis = 1,inplace = True)
```

# Removing the Unnamed Column


```python
dataset.drop(dataset.columns[dataset.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>remaining_min</th>
      <th>power_of_shot</th>
      <th>knockout_match</th>
      <th>remaining_sec</th>
      <th>distance_of_shot</th>
      <th>is_goal</th>
      <th>area_of_shot</th>
      <th>range_of_shot</th>
      <th>team_id</th>
      <th>remaining_min.1</th>
      <th>power_of_shot.1</th>
      <th>knockout_match.1</th>
      <th>remaining_sec.1</th>
      <th>distance_of_shot.1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>50.608</td>
      <td>54.2000</td>
      <td>38.000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>28.800</td>
      <td>22.0000</td>
      <td>35.000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>36.0</td>
      <td>1.0</td>
      <td>Left Side Center(LC)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>92.64</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>63.7216</td>
      <td>54.400</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>52.0</td>
      <td>42.0</td>
      <td>0.0</td>
      <td>Right Side Center(RC)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>122.608</td>
      <td>52.0000</td>
      <td>42.000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>42.64</td>
      <td>2.00</td>
      <td>0.000</td>
      <td>19.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>34.000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>52.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>112.2000</td>
      <td>89.400</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>68.64</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>5.0000</td>
      <td>22.000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>32.0</td>
      <td>1.0</td>
      <td>Left Side(L)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>12.0000</td>
      <td>32.000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>32.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>52.2000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>56.0</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>24+ ft.</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>3.00</td>
      <td>80.928</td>
      <td>56.0000</td>
      <td>45.000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.0</td>
      <td>1.0</td>
      <td>Right Side Center(RC)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>1.00</td>
      <td>106.608</td>
      <td>64.7856</td>
      <td>16.400</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Left Side(L)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>77.36</td>
      <td>0.000</td>
      <td>9.0000</td>
      <td>32.000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>44.0</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>68.36</td>
      <td>0.000</td>
      <td>44.0000</td>
      <td>29.000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>16.0000</td>
      <td>25.000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>94.64</td>
      <td>70.36</td>
      <td>0.000</td>
      <td>70.7856</td>
      <td>40.000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>1.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>17</th>
      <td>8.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>45.0</td>
      <td>1.0</td>
      <td>Left Side Center(LC)</td>
      <td>NaN</td>
      <td>1610612747</td>
      <td>38.64</td>
      <td>2.00</td>
      <td>0.000</td>
      <td>50.0000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>29.0</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>29.0000</td>
      <td>106.400</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>46.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>70.36</td>
      <td>0.000</td>
      <td>46.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>8.0000</td>
      <td>34.000</td>
    </tr>
    <tr>
      <th>21</th>
      <td>9.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>43.36</td>
      <td>0.000</td>
      <td>4.0000</td>
      <td>38.000</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>31.0</td>
      <td>1.0</td>
      <td>Center(C)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>57.0000</td>
      <td>31.000</td>
    </tr>
    <tr>
      <th>23</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>Left Side Center(LC)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>47.0000</td>
      <td>38.000</td>
    </tr>
    <tr>
      <th>24</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>27.0</td>
      <td>1.0</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>25.2000</td>
      <td>27.000</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>93.64</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>36.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>83.36</td>
      <td>0.000</td>
      <td>63.7856</td>
      <td>40.000</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>24+ ft.</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>30.0000</td>
      <td>53.400</td>
    </tr>
    <tr>
      <th>28</th>
      <td>8.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>Right Side Center(RC)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>4.00</td>
      <td>61.928</td>
      <td>19.0000</td>
      <td>18.400</td>
    </tr>
    <tr>
      <th>29</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>73.64</td>
      <td>4.00</td>
      <td>0.000</td>
      <td>22.0000</td>
      <td>28.000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30667</th>
      <td>9.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>33.0</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>24+ ft.</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>33.0000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30668</th>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>42.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>68.36</td>
      <td>54.608</td>
      <td>27.0000</td>
      <td>42.000</td>
    </tr>
    <tr>
      <th>30669</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>48.0</td>
      <td>37.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>112.7856</td>
      <td>37.000</td>
    </tr>
    <tr>
      <th>30670</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>34.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>38.64</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>34.0000</td>
      <td>27.000</td>
    </tr>
    <tr>
      <th>30671</th>
      <td>8.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>51.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>102.36</td>
      <td>1.000</td>
      <td>51.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>30672</th>
      <td>8.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>Right Side(R)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>109.64</td>
      <td>29.36</td>
      <td>1.000</td>
      <td>9.0000</td>
      <td>30.000</td>
    </tr>
    <tr>
      <th>30673</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>Right Side(R)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>63.36</td>
      <td>1.000</td>
      <td>14.0000</td>
      <td>31.000</td>
    </tr>
    <tr>
      <th>30674</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>34.0</td>
      <td>44.0</td>
      <td>1.0</td>
      <td>Right Side Center(RC)</td>
      <td>24+ ft.</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>34.0000</td>
      <td>44.000</td>
    </tr>
    <tr>
      <th>30675</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>24+ ft.</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>34.928</td>
      <td>0.0000</td>
      <td>45.000</td>
    </tr>
    <tr>
      <th>30676</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>1.0</td>
      <td>Left Side(L)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>34.000</td>
    </tr>
    <tr>
      <th>30677</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>38.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>38.000</td>
    </tr>
    <tr>
      <th>30678</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>29.36</td>
      <td>1.000</td>
      <td>34.0000</td>
      <td>22.000</td>
    </tr>
    <tr>
      <th>30679</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>41.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>71.64</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>41.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>30680</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>121.928</td>
      <td>40.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>30681</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>46.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>24+ ft.</td>
      <td>1610612747</td>
      <td>68.64</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>2.0000</td>
      <td>46.000</td>
    </tr>
    <tr>
      <th>30682</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>41.36</td>
      <td>87.608</td>
      <td>30.0000</td>
      <td>28.000</td>
    </tr>
    <tr>
      <th>30683</th>
      <td>10.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>51.0</td>
      <td>41.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>51.0000</td>
      <td>41.000</td>
    </tr>
    <tr>
      <th>30684</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>16.0</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>50.64</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>16.0000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30685</th>
      <td>7.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>31.0</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>24+ ft.</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>44.608</td>
      <td>31.0000</td>
      <td>46.000</td>
    </tr>
    <tr>
      <th>30686</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>37.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>78.400</td>
    </tr>
    <tr>
      <th>30687</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>18.0000</td>
      <td>30.000</td>
    </tr>
    <tr>
      <th>30688</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>33.0</td>
      <td>1.0</td>
      <td>Left Side(L)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>37.64</td>
      <td>NaN</td>
      <td>1.000</td>
      <td>7.0000</td>
      <td>16.400</td>
    </tr>
    <tr>
      <th>30689</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>87.0</td>
      <td>0.0</td>
      <td>Mid Ground(MG)</td>
      <td>Back Court Shot</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>31.60</td>
      <td>1.000</td>
      <td>0.0000</td>
      <td>45.728</td>
    </tr>
    <tr>
      <th>30690</th>
      <td>11.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>37.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>8-16 ft.</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>23.36</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>35.000</td>
    </tr>
    <tr>
      <th>30691</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>4.00</td>
      <td>27.800</td>
      <td>59.7856</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>30692</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>17.20</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>5.0000</td>
      <td>24.000</td>
    </tr>
    <tr>
      <th>30693</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>64.36</td>
      <td>1.000</td>
      <td>5.0000</td>
      <td>20.000</td>
    </tr>
    <tr>
      <th>30694</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>41.0</td>
      <td>1.0</td>
      <td>Left Side Center(LC)</td>
      <td>16-24 ft.</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>28.0000</td>
      <td>41.000</td>
    </tr>
    <tr>
      <th>30695</th>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>NaN</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>112.36</td>
      <td>1.000</td>
      <td>10.0000</td>
      <td>46.000</td>
    </tr>
    <tr>
      <th>30696</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>39.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>Less Than 8 ft.</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>39.0000</td>
      <td>27.000</td>
    </tr>
  </tbody>
</table>
<p>30697 rows × 14 columns</p>
</div>



# Creating Features

## No.1


```python
dataset['goal scoring'] = np.where(dataset['range_of_shot']=='Less Than 8 ft.', '1', '-1')
dataset
dataset.drop(['range_of_shot'],axis = 1,inplace = True)
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>remaining_min</th>
      <th>power_of_shot</th>
      <th>knockout_match</th>
      <th>remaining_sec</th>
      <th>distance_of_shot</th>
      <th>is_goal</th>
      <th>area_of_shot</th>
      <th>team_id</th>
      <th>remaining_min.1</th>
      <th>power_of_shot.1</th>
      <th>knockout_match.1</th>
      <th>remaining_sec.1</th>
      <th>distance_of_shot.1</th>
      <th>goal scoring</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>50.608</td>
      <td>54.2000</td>
      <td>38.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>28.800</td>
      <td>22.0000</td>
      <td>35.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>36.0</td>
      <td>1.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>92.64</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>63.7216</td>
      <td>54.400</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>52.0</td>
      <td>42.0</td>
      <td>0.0</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>122.608</td>
      <td>52.0000</td>
      <td>42.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>42.64</td>
      <td>2.00</td>
      <td>0.000</td>
      <td>19.0000</td>
      <td>20.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>34.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>52.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>112.2000</td>
      <td>89.400</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>68.64</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>5.0000</td>
      <td>22.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>32.0</td>
      <td>1.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>12.0000</td>
      <td>32.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>32.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>52.2000</td>
      <td>NaN</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>56.0</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>3.00</td>
      <td>80.928</td>
      <td>56.0000</td>
      <td>45.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.0</td>
      <td>1.0</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>1.00</td>
      <td>106.608</td>
      <td>64.7856</td>
      <td>16.400</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>77.36</td>
      <td>0.000</td>
      <td>9.0000</td>
      <td>32.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>44.0</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>68.36</td>
      <td>0.000</td>
      <td>44.0000</td>
      <td>29.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>16.0000</td>
      <td>25.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>94.64</td>
      <td>70.36</td>
      <td>0.000</td>
      <td>70.7856</td>
      <td>40.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>1.0000</td>
      <td>20.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>8.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>45.0</td>
      <td>1.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>38.64</td>
      <td>2.00</td>
      <td>0.000</td>
      <td>50.0000</td>
      <td>NaN</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>29.0</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>29.0000</td>
      <td>106.400</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>46.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>70.36</td>
      <td>0.000</td>
      <td>46.0000</td>
      <td>20.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>8.0000</td>
      <td>34.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>9.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>43.36</td>
      <td>0.000</td>
      <td>4.0000</td>
      <td>38.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>31.0</td>
      <td>1.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>57.0000</td>
      <td>31.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>47.0000</td>
      <td>38.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>27.0</td>
      <td>1.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>25.2000</td>
      <td>27.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>93.64</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>36.0000</td>
      <td>20.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>83.36</td>
      <td>0.000</td>
      <td>63.7856</td>
      <td>40.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>30.0000</td>
      <td>53.400</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>8.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>4.00</td>
      <td>61.928</td>
      <td>19.0000</td>
      <td>18.400</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>73.64</td>
      <td>4.00</td>
      <td>0.000</td>
      <td>22.0000</td>
      <td>28.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30667</th>
      <td>9.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>33.0</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>33.0000</td>
      <td>NaN</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30668</th>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>42.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>68.36</td>
      <td>54.608</td>
      <td>27.0000</td>
      <td>42.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30669</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>48.0</td>
      <td>37.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>112.7856</td>
      <td>37.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30670</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>34.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>38.64</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>34.0000</td>
      <td>27.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30671</th>
      <td>8.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>51.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>102.36</td>
      <td>1.000</td>
      <td>51.0000</td>
      <td>20.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30672</th>
      <td>8.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>109.64</td>
      <td>29.36</td>
      <td>1.000</td>
      <td>9.0000</td>
      <td>30.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30673</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>63.36</td>
      <td>1.000</td>
      <td>14.0000</td>
      <td>31.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30674</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>34.0</td>
      <td>44.0</td>
      <td>1.0</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>34.0000</td>
      <td>44.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30675</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>34.928</td>
      <td>0.0000</td>
      <td>45.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30676</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>1.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>34.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30677</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>38.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>38.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30678</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>29.36</td>
      <td>1.000</td>
      <td>34.0000</td>
      <td>22.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30679</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>41.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>71.64</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>41.0000</td>
      <td>20.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30680</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>121.928</td>
      <td>40.0000</td>
      <td>20.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30681</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>46.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>68.64</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>2.0000</td>
      <td>46.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30682</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>41.36</td>
      <td>87.608</td>
      <td>30.0000</td>
      <td>28.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30683</th>
      <td>10.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>51.0</td>
      <td>41.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>51.0000</td>
      <td>41.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30684</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>16.0</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>50.64</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>16.0000</td>
      <td>NaN</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30685</th>
      <td>7.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>31.0</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>44.608</td>
      <td>31.0000</td>
      <td>46.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30686</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>37.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>78.400</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30687</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>18.0000</td>
      <td>30.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30688</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>33.0</td>
      <td>1.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>37.64</td>
      <td>NaN</td>
      <td>1.000</td>
      <td>7.0000</td>
      <td>16.400</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30689</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>87.0</td>
      <td>0.0</td>
      <td>Mid Ground(MG)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>31.60</td>
      <td>1.000</td>
      <td>0.0000</td>
      <td>45.728</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30690</th>
      <td>11.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>37.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>23.36</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>35.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30691</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>4.00</td>
      <td>27.800</td>
      <td>59.7856</td>
      <td>20.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30692</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>17.20</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>5.0000</td>
      <td>24.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30693</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>64.36</td>
      <td>1.000</td>
      <td>5.0000</td>
      <td>20.000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30694</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>41.0</td>
      <td>1.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>28.0000</td>
      <td>41.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30695</th>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>112.36</td>
      <td>1.000</td>
      <td>10.0000</td>
      <td>46.000</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30696</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>39.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>39.0000</td>
      <td>27.000</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>30697 rows × 14 columns</p>
</div>



## No.2


```python
dataset['goal scoring2'] = np.where(dataset['area_of_shot']=='Right Side(R)', '1', '-1')
dataset
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>remaining_min</th>
      <th>power_of_shot</th>
      <th>knockout_match</th>
      <th>remaining_sec</th>
      <th>distance_of_shot</th>
      <th>is_goal</th>
      <th>area_of_shot</th>
      <th>team_id</th>
      <th>remaining_min.1</th>
      <th>power_of_shot.1</th>
      <th>knockout_match.1</th>
      <th>remaining_sec.1</th>
      <th>distance_of_shot.1</th>
      <th>goal scoring</th>
      <th>goal scoring2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>50.608</td>
      <td>54.2000</td>
      <td>38.000</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>28.800</td>
      <td>22.0000</td>
      <td>35.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>36.0</td>
      <td>1.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>92.64</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>63.7216</td>
      <td>54.400</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>52.0</td>
      <td>42.0</td>
      <td>0.0</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>122.608</td>
      <td>52.0000</td>
      <td>42.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>42.64</td>
      <td>2.00</td>
      <td>0.000</td>
      <td>19.0000</td>
      <td>20.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>32.0</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>34.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>8.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>52.0</td>
      <td>20.0</td>
      <td>1.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>112.2000</td>
      <td>89.400</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>68.64</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>5.0000</td>
      <td>22.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>12.0</td>
      <td>32.0</td>
      <td>1.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>12.0000</td>
      <td>32.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>32.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>52.2000</td>
      <td>NaN</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>56.0</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>3.00</td>
      <td>80.928</td>
      <td>56.0000</td>
      <td>45.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>37.0</td>
      <td>1.0</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>1.00</td>
      <td>106.608</td>
      <td>64.7856</td>
      <td>16.400</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>77.36</td>
      <td>0.000</td>
      <td>9.0000</td>
      <td>32.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>44.0</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>68.36</td>
      <td>0.000</td>
      <td>44.0000</td>
      <td>29.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>25.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>16.0000</td>
      <td>25.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>94.64</td>
      <td>70.36</td>
      <td>0.000</td>
      <td>70.7856</td>
      <td>40.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>1.0000</td>
      <td>20.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>8.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>45.0</td>
      <td>1.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>38.64</td>
      <td>2.00</td>
      <td>0.000</td>
      <td>50.0000</td>
      <td>NaN</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>29.0</td>
      <td>36.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>29.0000</td>
      <td>106.400</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>46.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>70.36</td>
      <td>0.000</td>
      <td>46.0000</td>
      <td>20.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>8.0</td>
      <td>34.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>8.0000</td>
      <td>34.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>9.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>43.36</td>
      <td>0.000</td>
      <td>4.0000</td>
      <td>38.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>7.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>57.0</td>
      <td>31.0</td>
      <td>1.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>57.0000</td>
      <td>31.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>38.0</td>
      <td>1.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>47.0000</td>
      <td>38.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>27.0</td>
      <td>1.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>25.2000</td>
      <td>27.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>93.64</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>36.0000</td>
      <td>20.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>40.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>83.36</td>
      <td>0.000</td>
      <td>63.7856</td>
      <td>40.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>30.0</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>30.0000</td>
      <td>53.400</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>28</th>
      <td>8.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>4.00</td>
      <td>61.928</td>
      <td>19.0000</td>
      <td>18.400</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>22.0</td>
      <td>28.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>73.64</td>
      <td>4.00</td>
      <td>0.000</td>
      <td>22.0000</td>
      <td>28.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30667</th>
      <td>9.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>33.0</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>33.0000</td>
      <td>NaN</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30668</th>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>42.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>68.36</td>
      <td>54.608</td>
      <td>27.0000</td>
      <td>42.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30669</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>48.0</td>
      <td>37.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>112.7856</td>
      <td>37.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30670</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>34.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>38.64</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>34.0000</td>
      <td>27.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30671</th>
      <td>8.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>51.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>102.36</td>
      <td>1.000</td>
      <td>51.0000</td>
      <td>20.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30672</th>
      <td>8.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>9.0</td>
      <td>30.0</td>
      <td>1.0</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>109.64</td>
      <td>29.36</td>
      <td>1.000</td>
      <td>9.0000</td>
      <td>30.000</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30673</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>14.0</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>63.36</td>
      <td>1.000</td>
      <td>14.0000</td>
      <td>31.000</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30674</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>34.0</td>
      <td>44.0</td>
      <td>1.0</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>34.0000</td>
      <td>44.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30675</th>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>45.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>34.928</td>
      <td>0.0000</td>
      <td>45.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30676</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>1.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>34.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30677</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>38.0</td>
      <td>0.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>38.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30678</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>34.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>29.36</td>
      <td>1.000</td>
      <td>34.0000</td>
      <td>22.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30679</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>41.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>71.64</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>41.0000</td>
      <td>20.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30680</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>121.928</td>
      <td>40.0000</td>
      <td>20.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30681</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>46.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>68.64</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>2.0000</td>
      <td>46.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30682</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>41.36</td>
      <td>87.608</td>
      <td>30.0000</td>
      <td>28.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30683</th>
      <td>10.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>51.0</td>
      <td>41.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>51.0000</td>
      <td>41.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30684</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>16.0</td>
      <td>33.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>50.64</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>16.0000</td>
      <td>NaN</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30685</th>
      <td>7.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>31.0</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>44.608</td>
      <td>31.0000</td>
      <td>46.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30686</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>37.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>78.400</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30687</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>18.0000</td>
      <td>30.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30688</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>7.0</td>
      <td>33.0</td>
      <td>1.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>37.64</td>
      <td>NaN</td>
      <td>1.000</td>
      <td>7.0000</td>
      <td>16.400</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30689</th>
      <td>0.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>87.0</td>
      <td>0.0</td>
      <td>Mid Ground(MG)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>31.60</td>
      <td>1.000</td>
      <td>0.0000</td>
      <td>45.728</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30690</th>
      <td>11.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>37.0</td>
      <td>35.0</td>
      <td>0.0</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>23.36</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>35.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30691</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>4.0</td>
      <td>20.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>4.00</td>
      <td>27.800</td>
      <td>59.7856</td>
      <td>20.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30692</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>24.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>17.20</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>5.0000</td>
      <td>24.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30693</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>64.36</td>
      <td>1.000</td>
      <td>5.0000</td>
      <td>20.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30694</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>28.0</td>
      <td>41.0</td>
      <td>1.0</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>28.0000</td>
      <td>41.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30695</th>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>10.0</td>
      <td>46.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>112.36</td>
      <td>1.000</td>
      <td>10.0000</td>
      <td>46.000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30696</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>39.0</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>39.0000</td>
      <td>27.000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>30697 rows × 15 columns</p>
</div>




```python
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(dataset.drop('is_goal',axis=1), dataset['is_goal'], test_size=0.2)

```

# Train-Test split


```python
X_test.isnull().sum()
```




    remaining_min         352
    power_of_shot         281
    knockout_match        308
    remaining_sec         320
    distance_of_shot      345
    area_of_shot          314
    team_id                 0
    remaining_min.1       279
    power_of_shot.1       315
    knockout_match.1      292
    remaining_sec.1       305
    distance_of_shot.1    306
    goal scoring            0
    goal scoring2           0
    dtype: int64




```python
X_train.drop(['area_of_shot'],axis = 1,inplace = True)
X_test.drop(['area_of_shot'],axis = 1,inplace = True)
```


```python
X_test.fillna(X_test.median(), inplace=True)
```


```python
X_train.fillna(X_train.mean(), inplace=True)
```


```python
X_test.isnull().sum()
```




    remaining_min         0
    power_of_shot         0
    knockout_match        0
    remaining_sec         0
    distance_of_shot      0
    team_id               0
    remaining_min.1       0
    power_of_shot.1       0
    knockout_match.1      0
    remaining_sec.1       0
    distance_of_shot.1    0
    goal scoring          0
    goal scoring2         0
    dtype: int64




```python
X_train.isnull().sum()
```




    remaining_min         0
    power_of_shot         0
    knockout_match        0
    remaining_sec         0
    distance_of_shot      0
    team_id               0
    remaining_min.1       0
    power_of_shot.1       0
    knockout_match.1      0
    remaining_sec.1       0
    distance_of_shot.1    0
    goal scoring          0
    goal scoring2         0
    dtype: int64




```python
Y_train.fillna(Y_train.median(), inplace=True)
```


```python
Y_test.fillna(Y_test.median(), inplace=True)
```


```python
Y_train.isnull().sum()
```




    0




```python
Y_test.isnull().sum()
```




    0




```python



X_train.isnull().sum()

X_train.isnull().sum()


#Y_test.fillna(Y_test.median(), inplace=True)
#Y_test.isnull().sum()
```


```python
X_test.fillna(X_test.mean(), inplace=True)
```

# Converting float64 to int64


```python

X_train['goal scoring'] = X_train['goal scoring'].astype(np.int64)
X_train['goal scoring2'] = X_train['goal scoring2'].astype(np.int64)
X_train['remaining_min'] = X_train['remaining_min'].astype(np.int64)
X_train['power_of_shot'] = X_train['power_of_shot'].astype(np.int64)
X_train['knockout_match'] = X_train['knockout_match'].astype(np.int64)
X_train['remaining_sec'] = X_train['remaining_sec'].astype(np.int64)
X_train['distance_of_shot'] = X_train['distance_of_shot'].astype(np.int64)
X_train['remaining_min.1'] = X_train['remaining_min.1'].astype(np.int64)
X_train['power_of_shot.1'] = X_train['power_of_shot.1'].astype(np.int64)
X_train['knockout_match.1'] = X_train['knockout_match.1'].astype(np.int64)
X_train['remaining_sec.1'] = X_train['remaining_sec.1'].astype(np.int64)
X_train['distance_of_shot.1'] = X_train['distance_of_shot.1'].astype(np.int64)
```


```python
X_test['goal scoring'] = X_train['goal scoring'].astype(np.int64)
X_test['goal scoring2'] = X_train['goal scoring2'].astype(np.int64)
```


```python
X_train.fillna(X_train.mean(), inplace=True)

```


```python

```




    remaining_min            0
    power_of_shot            0
    knockout_match           0
    remaining_sec            0
    distance_of_shot         0
    team_id                  0
    remaining_min.1          0
    power_of_shot.1          0
    knockout_match.1         0
    remaining_sec.1          0
    distance_of_shot.1       0
    goal scoring          6140
    goal scoring2         6140
    dtype: int64




```python
X_test.fillna(value=0, inplace=True)
```


```python
X_test.isnull().sum()
```




    remaining_min         0
    power_of_shot         0
    knockout_match        0
    remaining_sec         0
    distance_of_shot      0
    team_id               0
    remaining_min.1       0
    power_of_shot.1       0
    knockout_match.1      0
    remaining_sec.1       0
    distance_of_shot.1    0
    goal scoring          0
    goal scoring2         0
    dtype: int64



# Correlation

# Model Estimation

## KNN


```python

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, Y_train)

predictions = knn.predict(X_test)
```


```python
X_test.dtypes
```




    remaining_min         float64
    power_of_shot         float64
    knockout_match        float64
    remaining_sec         float64
    distance_of_shot      float64
    team_id                 int64
    remaining_min.1       float64
    power_of_shot.1       float64
    knockout_match.1      float64
    remaining_sec.1       float64
    distance_of_shot.1    float64
    goal scoring          float64
    goal scoring2         float64
    dtype: object




```python
!pip install xgboost
import xgboost as xgb
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.5, learning_rate = 0.05,max_depth = 6, alpha = 5, n_estimators = 100)
xg_reg.fit(X_train,Y_train)
preds2 = xg_reg.predict(X_test)

print("On Test")
print("Mean Absolute Error:",mean_absolute_error(Y_test,preds2))
print("RMSE:",np.sqrt(mean_squared_error(Y_test, preds2)))


```

    Requirement already satisfied: xgboost in d:\anacondanew\lib\site-packages (0.90)
    Requirement already satisfied: numpy in d:\anacondanew\lib\site-packages (from xgboost) (1.14.3)
    Requirement already satisfied: scipy in d:\anacondanew\lib\site-packages (from xgboost) (1.1.0)
    

    distributed 1.21.8 requires msgpack, which is not installed.
    You are using pip version 10.0.1, however version 19.1.1 is available.
    You should consider upgrading via the 'python -m pip install --upgrade pip' command.
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-36-f1b1869d63f7> in <module>()
          2 import xgboost as xgb
          3 xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.5, learning_rate = 0.05,max_depth = 6, alpha = 5, n_estimators = 100)
    ----> 4 xg_reg.fit(X_train,Y_train)
          5 preds2 = xg_reg.predict(X_test)
          6 
    

    D:\Anacondanew\lib\site-packages\xgboost\sklearn.py in fit(self, X, y, sample_weight, eval_set, eval_metric, early_stopping_rounds, verbose, xgb_model, sample_weight_eval_set, callbacks)
        358                                    missing=self.missing, nthread=self.n_jobs)
        359         else:
    --> 360             trainDmatrix = DMatrix(X, label=y, missing=self.missing, nthread=self.n_jobs)
        361 
        362         evals_result = {}
    

    D:\Anacondanew\lib\site-packages\xgboost\core.py in __init__(self, data, label, missing, weight, silent, feature_names, feature_types, nthread)
        378         data, feature_names, feature_types = _maybe_pandas_data(data,
        379                                                                 feature_names,
    --> 380                                                                 feature_types)
        381 
        382         data, feature_names, feature_types = _maybe_dt_data(data,
    

    D:\Anacondanew\lib\site-packages\xgboost\core.py in _maybe_pandas_data(data, feature_names, feature_types)
        237         msg = """DataFrame.dtypes for data must be int, float or bool.
        238                 Did not expect the data types in fields """
    --> 239         raise ValueError(msg + ', '.join(bad_fields))
        240 
        241     if feature_names is None:
    

    ValueError: DataFrame.dtypes for data must be int, float or bool.
                    Did not expect the data types in fields goal scoring, goal scoring2



```python
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=500, n_jobs=-1)
rf_model.fit(X_train, Y_train)
```




    RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                           max_depth=None, max_features='auto', max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, n_estimators=500,
                           n_jobs=-1, oob_score=False, random_state=None, verbose=0,
                           warm_start=False)




```python
predictions11 = rf_model.predict(X_test)
```


```python
predictions11
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           1., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 1., 0., 0.,
           1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
           0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 1.,
           0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0.,
           0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 1., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           1., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0.,
           1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 0.,
           0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
           0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1.,
           0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0.,
           0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
           1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
           0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 1.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.,
           1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0.,
           0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           1., 0., 1.])




```python
Y_test.isnull().sum()
Y_test.fillna(Y_test.median(), inplace=True)
```

## Accuracy


```python
from sklearn.metrics import accuracy_score 
print("Accuracy:",accuracy_score(Y_test, predictions11))
```

    Accuracy: 0.6372964169381108
    

## MAE


```python
errors = abs(predictions - Y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2))
```

    Mean Absolute Error: 0.4
    

## Naive Bayes


```python

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, Y_train)
predictions1 = gnb.predict(X_test)

```

## Accuracy


```python
from sklearn.metrics import accuracy_score
print("Accuracy:",accuracy_score(Y_test, predictions))
predictions1
```

    Accuracy: 0.6013029315960912
    




    array([0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
           0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1., 1., 1., 0.,
           1., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 1., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
           1., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0.,
           1., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 1., 0.,
           0., 1., 0., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1.,
           1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0.,
           0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 1.,
           0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 1., 0.,
           1., 0., 1., 1., 1., 1., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1.,
           0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1.,
           0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0.,
           1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0.,
           1., 1., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0.,
           0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1.,
           1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
           0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0.,
           1., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1.,
           0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1.,
           1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
           1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1.,
           0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1.,
           1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
           1., 0., 1., 0., 1., 1., 1., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0.,
           1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0.,
           1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 1., 1., 0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0.,
           0., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0.,
           0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0.,
           1., 0., 0., 1., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1.,
           0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 1., 0.,
           1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 1., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 1.,
           1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1.,
           0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1.,
           0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1.,
           1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           1., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0.,
           1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           1., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1.,
           1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0.,
           1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 0., 1.,
           0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 1., 0., 0., 1., 1., 0., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 1.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 1.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1.,
           1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0.,
           1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1.,
           0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 1.,
           0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0.,
           1., 0., 1., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.,
           1., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0.,
           1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1.,
           0., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 1., 1.,
           0., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0.,
           1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1.,
           1., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 1.,
           0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0.,
           1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 1.,
           0., 0., 0., 1., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0.,
           0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0.,
           1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1.,
           1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
           0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1.,
           0., 1., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 1., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1.,
           0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1.,
           1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1., 1.,
           0., 1., 1., 1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0.,
           1., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0.,
           0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1.,
           0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 1.,
           0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
           1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1.,
           0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0.,
           1., 0., 1., 1., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0.,
           1., 1., 0., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1.,
           0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0.,
           0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
           0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
           1., 0., 0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           1., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
           1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 1., 0.,
           0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0.,
           1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
           1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1.,
           1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0.,
           1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 1.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.,
           0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 1.,
           0., 1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1., 1.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0.,
           0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1.,
           1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 1., 1.,
           0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0.,
           0., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
           1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 1., 0., 0., 1., 0., 1., 1., 0., 0., 1., 1., 0., 1., 1., 0.,
           0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0.,
           1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
           1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
           0., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 1., 1.,
           1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1.,
           1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
           1., 1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 1., 0.])



## MAE


```python
errors1 = abs(predictions1 - Y_test)
print('Mean Absolute Error:', round(np.mean(errors1), 2))
```

    Mean Absolute Error: 0.38
    

# Data with blank is_goal


```python
newdata = dataset.drop(dataset[dataset.is_goal==1].index, axis=0, inplace=False)
newdata = dataset.drop(dataset[dataset.is_goal==0].index, axis=0, inplace=False)

```


```python
bool_series = pd.isnull(newdata["is_goal"]) 
newdata[bool_series]  
#newdata.drop(['area_of_shot'],axis = 1,inplace=True)
newdata1 = newdata[bool_series] 
newdata1

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>remaining_min</th>
      <th>power_of_shot</th>
      <th>knockout_match</th>
      <th>remaining_sec</th>
      <th>distance_of_shot</th>
      <th>is_goal</th>
      <th>area_of_shot</th>
      <th>team_id</th>
      <th>remaining_min.1</th>
      <th>power_of_shot.1</th>
      <th>knockout_match.1</th>
      <th>remaining_sec.1</th>
      <th>distance_of_shot.1</th>
      <th>goal scoring</th>
      <th>goal scoring2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>50.608</td>
      <td>54.2000</td>
      <td>38.0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>68.64</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>5.0000</td>
      <td>22.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>1.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>46.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>70.36</td>
      <td>0.000</td>
      <td>46.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>9.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>43.36</td>
      <td>0.000</td>
      <td>4.0000</td>
      <td>38.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>11.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>26.0</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>26.0000</td>
      <td>37.0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>58.0</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>121.608</td>
      <td>58.0000</td>
      <td>40.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>33.0</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>33.0000</td>
      <td>21.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>58.0</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>104.36</td>
      <td>102.608</td>
      <td>58.0000</td>
      <td>84.4</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>108.64</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>9.0000</td>
      <td>59.4</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>5.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>33.0</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>2.00</td>
      <td>0.000</td>
      <td>33.0000</td>
      <td>36.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>44</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>29.0000</td>
      <td>22.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>0.000</td>
      <td>19.0000</td>
      <td>37.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>54</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>58.0</td>
      <td>48.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>4.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>48.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>59</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>48.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>42.36</td>
      <td>0.000</td>
      <td>10.0000</td>
      <td>NaN</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>108.928</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>65</th>
      <td>10.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>4.00</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>29.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>66</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>32.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>6.0000</td>
      <td>32.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>70</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>39.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>77.800</td>
      <td>15.0000</td>
      <td>39.0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>45.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.2000</td>
      <td>45.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>75</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>42.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>41.60</td>
      <td>0.000</td>
      <td>2.0000</td>
      <td>NaN</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>11.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>49.0</td>
      <td>27.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>34.64</td>
      <td>4.00</td>
      <td>0.000</td>
      <td>28.2000</td>
      <td>13.4</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>84</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>48.0</td>
      <td>27.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>48.0000</td>
      <td>27.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>85</th>
      <td>7.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>26.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>26.0000</td>
      <td>33.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>86</th>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>36.0000</td>
      <td>38.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>34.64</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>47.0000</td>
      <td>35.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>94</th>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>4.00</td>
      <td>2.00</td>
      <td>0.000</td>
      <td>114.2000</td>
      <td>94.4</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>114.36</td>
      <td>39.608</td>
      <td>15.0000</td>
      <td>38.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>103</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>102.928</td>
      <td>16.0000</td>
      <td>72.4</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>112</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>41.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>0.000</td>
      <td>45.7216</td>
      <td>50.4</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30567</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>32.0</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>113.64</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>32.0000</td>
      <td>35.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30569</th>
      <td>11.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>49.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>117.36</td>
      <td>1.000</td>
      <td>49.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30580</th>
      <td>8.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>93.2000</td>
      <td>36.0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30583</th>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>15.0000</td>
      <td>44.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30590</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>41.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>4.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>41.0000</td>
      <td>29.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30593</th>
      <td>11.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>36.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>36.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30613</th>
      <td>9.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>132.7856</td>
      <td>37.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30616</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>31.0</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>31.0000</td>
      <td>40.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30617</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>21.0000</td>
      <td>25.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30625</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>29.0</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>29.0000</td>
      <td>37.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30629</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>55.0</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>2.00</td>
      <td>78.608</td>
      <td>55.0000</td>
      <td>36.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30630</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>2.0000</td>
      <td>40.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30631</th>
      <td>10.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>36.0</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>21.60</td>
      <td>1.000</td>
      <td>36.0000</td>
      <td>57.4</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30633</th>
      <td>9.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>23.36</td>
      <td>1.000</td>
      <td>27.0000</td>
      <td>14.4</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30635</th>
      <td>7.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>23.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>21.0000</td>
      <td>52.4</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30636</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>39.0</td>
      <td>23.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>106.64</td>
      <td>25.36</td>
      <td>1.000</td>
      <td>39.0000</td>
      <td>23.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30638</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>56.0</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>56.0000</td>
      <td>36.0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30646</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>42.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>1.00</td>
      <td>5.00</td>
      <td>1.000</td>
      <td>87.7856</td>
      <td>42.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30648</th>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>51.0</td>
      <td>23.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>5.00</td>
      <td>1.000</td>
      <td>35.2000</td>
      <td>23.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30655</th>
      <td>11.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>17.0</td>
      <td>34.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>85.2000</td>
      <td>34.0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30659</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>NaN</td>
      <td>73.608</td>
      <td>18.0000</td>
      <td>36.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30664</th>
      <td>6.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>NaN</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>26.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30668</th>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>42.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>68.36</td>
      <td>54.608</td>
      <td>27.0000</td>
      <td>42.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30679</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>41.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>71.64</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>41.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30680</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>121.928</td>
      <td>40.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30681</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>46.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>68.64</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>2.0000</td>
      <td>46.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30682</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>41.36</td>
      <td>87.608</td>
      <td>30.0000</td>
      <td>28.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30686</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>37.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>78.4</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30687</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>18.0000</td>
      <td>30.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30693</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>64.36</td>
      <td>1.000</td>
      <td>5.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>6268 rows × 15 columns</p>
</div>




```python
newdata1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>remaining_min</th>
      <th>power_of_shot</th>
      <th>knockout_match</th>
      <th>remaining_sec</th>
      <th>distance_of_shot</th>
      <th>is_goal</th>
      <th>area_of_shot</th>
      <th>team_id</th>
      <th>remaining_min.1</th>
      <th>power_of_shot.1</th>
      <th>knockout_match.1</th>
      <th>remaining_sec.1</th>
      <th>distance_of_shot.1</th>
      <th>goal scoring</th>
      <th>goal scoring2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>27.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>50.608</td>
      <td>54.2000</td>
      <td>38.0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>68.64</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>5.0000</td>
      <td>22.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>1.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>46.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>70.36</td>
      <td>0.000</td>
      <td>46.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>9.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>43.36</td>
      <td>0.000</td>
      <td>4.0000</td>
      <td>38.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>11.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>26.0</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>26.0000</td>
      <td>37.0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>10.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>58.0</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>1.00</td>
      <td>121.608</td>
      <td>58.0000</td>
      <td>40.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>7.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>33.0</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>33.0000</td>
      <td>21.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>58.0</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>104.36</td>
      <td>102.608</td>
      <td>58.0000</td>
      <td>84.4</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>9.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>108.64</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>9.0000</td>
      <td>59.4</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>5.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>33.0</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>2.00</td>
      <td>0.000</td>
      <td>33.0000</td>
      <td>36.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>44</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>0.000</td>
      <td>29.0000</td>
      <td>22.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>19.0</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>4.00</td>
      <td>0.000</td>
      <td>19.0000</td>
      <td>37.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>54</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>58.0</td>
      <td>48.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>4.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>48.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>59</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>10.0</td>
      <td>48.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>42.36</td>
      <td>0.000</td>
      <td>10.0000</td>
      <td>NaN</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>108.928</td>
      <td>NaN</td>
      <td>25.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>65</th>
      <td>10.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>13.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>4.00</td>
      <td>0.000</td>
      <td>NaN</td>
      <td>29.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>66</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>32.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>6.0000</td>
      <td>32.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>70</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>39.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>77.800</td>
      <td>15.0000</td>
      <td>39.0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>3.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>11.0</td>
      <td>45.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29.2000</td>
      <td>45.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>75</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>42.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>41.60</td>
      <td>0.000</td>
      <td>2.0000</td>
      <td>NaN</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>11.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>49.0</td>
      <td>27.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>34.64</td>
      <td>4.00</td>
      <td>0.000</td>
      <td>28.2000</td>
      <td>13.4</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>84</th>
      <td>9.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>48.0</td>
      <td>27.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>48.0000</td>
      <td>27.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>85</th>
      <td>7.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>26.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>26.0000</td>
      <td>33.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>86</th>
      <td>6.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>36.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>36.0000</td>
      <td>38.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>47.0</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>34.64</td>
      <td>1.00</td>
      <td>0.000</td>
      <td>47.0000</td>
      <td>35.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>94</th>
      <td>4.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>25.0</td>
      <td>33.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>4.00</td>
      <td>2.00</td>
      <td>0.000</td>
      <td>114.2000</td>
      <td>94.4</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>15.0</td>
      <td>38.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>114.36</td>
      <td>39.608</td>
      <td>15.0000</td>
      <td>38.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>103</th>
      <td>4.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>16.0</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>4.00</td>
      <td>3.00</td>
      <td>102.928</td>
      <td>16.0000</td>
      <td>72.4</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>112</th>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>43.0</td>
      <td>41.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>4.00</td>
      <td>0.000</td>
      <td>45.7216</td>
      <td>50.4</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30567</th>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>32.0</td>
      <td>35.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>113.64</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>32.0000</td>
      <td>35.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30569</th>
      <td>11.0</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>49.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>117.36</td>
      <td>1.000</td>
      <td>49.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30580</th>
      <td>8.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>8.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>93.2000</td>
      <td>36.0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30583</th>
      <td>2.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>44.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>2.00</td>
      <td>4.00</td>
      <td>1.000</td>
      <td>15.0000</td>
      <td>44.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30590</th>
      <td>4.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>41.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>4.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>41.0000</td>
      <td>29.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30593</th>
      <td>11.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>36.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>36.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30613</th>
      <td>9.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>12.0</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>Right Side Center(RC)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>132.7856</td>
      <td>37.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30616</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>31.0</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>31.0000</td>
      <td>40.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30617</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>1.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>21.0000</td>
      <td>25.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30625</th>
      <td>5.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>29.0</td>
      <td>37.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>1.00</td>
      <td>1.000</td>
      <td>29.0000</td>
      <td>37.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30629</th>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>55.0</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>2.00</td>
      <td>78.608</td>
      <td>55.0000</td>
      <td>36.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30630</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>40.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>2.0000</td>
      <td>40.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30631</th>
      <td>10.0</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>36.0</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>10.00</td>
      <td>21.60</td>
      <td>1.000</td>
      <td>36.0000</td>
      <td>57.4</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30633</th>
      <td>9.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>9.00</td>
      <td>23.36</td>
      <td>1.000</td>
      <td>27.0000</td>
      <td>14.4</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30635</th>
      <td>7.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>21.0</td>
      <td>23.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>21.0000</td>
      <td>52.4</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30636</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>39.0</td>
      <td>23.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>106.64</td>
      <td>25.36</td>
      <td>1.000</td>
      <td>39.0000</td>
      <td>23.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30638</th>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>56.0</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>56.0000</td>
      <td>36.0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30646</th>
      <td>1.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>58.0</td>
      <td>42.0</td>
      <td>NaN</td>
      <td>Left Side Center(LC)</td>
      <td>1610612747</td>
      <td>1.00</td>
      <td>5.00</td>
      <td>1.000</td>
      <td>87.7856</td>
      <td>42.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30648</th>
      <td>0.0</td>
      <td>5.0</td>
      <td>1.0</td>
      <td>51.0</td>
      <td>23.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>NaN</td>
      <td>5.00</td>
      <td>1.000</td>
      <td>35.2000</td>
      <td>23.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30655</th>
      <td>11.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>17.0</td>
      <td>34.0</td>
      <td>NaN</td>
      <td>Right Side(R)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>85.2000</td>
      <td>34.0</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30659</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>36.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>NaN</td>
      <td>73.608</td>
      <td>18.0000</td>
      <td>36.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30664</th>
      <td>6.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>26.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>NaN</td>
      <td>1.000</td>
      <td>1.0000</td>
      <td>26.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30668</th>
      <td>7.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>27.0</td>
      <td>42.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>7.00</td>
      <td>68.36</td>
      <td>54.608</td>
      <td>27.0000</td>
      <td>42.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30679</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>41.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>71.64</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>41.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30680</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>40.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>0.00</td>
      <td>2.00</td>
      <td>121.928</td>
      <td>40.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30681</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>46.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>68.64</td>
      <td>2.00</td>
      <td>1.000</td>
      <td>2.0000</td>
      <td>46.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30682</th>
      <td>11.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>30.0</td>
      <td>28.0</td>
      <td>NaN</td>
      <td>Left Side(L)</td>
      <td>1610612747</td>
      <td>11.00</td>
      <td>41.36</td>
      <td>87.608</td>
      <td>30.0000</td>
      <td>28.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30686</th>
      <td>5.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>37.0</td>
      <td>29.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>5.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>37.0000</td>
      <td>78.4</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30687</th>
      <td>3.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>18.0</td>
      <td>30.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>3.00</td>
      <td>3.00</td>
      <td>1.000</td>
      <td>18.0000</td>
      <td>30.0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30693</th>
      <td>6.0</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>5.0</td>
      <td>20.0</td>
      <td>NaN</td>
      <td>Center(C)</td>
      <td>1610612747</td>
      <td>6.00</td>
      <td>64.36</td>
      <td>1.000</td>
      <td>5.0000</td>
      <td>20.0</td>
      <td>1</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>6268 rows × 15 columns</p>
</div>



# Preparing the new test set


```python

newdata1.fillna(X_test.mean(), inplace=True)
newdata1.isnull().sum()
#
```

    D:\Anacondanew\lib\site-packages\pandas\core\generic.py:5430: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      self._update_inplace(new_data)
    




    remaining_min            0
    power_of_shot            0
    knockout_match           0
    remaining_sec            0
    distance_of_shot         0
    is_goal               6268
    area_of_shot           320
    team_id                  0
    remaining_min.1          0
    power_of_shot.1          0
    knockout_match.1         0
    remaining_sec.1          0
    distance_of_shot.1       0
    goal scoring             0
    goal scoring2            0
    dtype: int64




```python
newdata1.drop(['is_goal'],axis = 1,inplace = True)
             
```

    D:\Anacondanew\lib\site-packages\pandas\core\frame.py:3694: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)
    


```python
newdata1.drop(['area_of_shot'],axis = 1,inplace = True)
```

    D:\Anacondanew\lib\site-packages\pandas\core\frame.py:3694: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
      errors=errors)
    


```python
newdata1
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>remaining_min</th>
      <th>power_of_shot</th>
      <th>knockout_match</th>
      <th>remaining_sec</th>
      <th>distance_of_shot</th>
      <th>team_id</th>
      <th>remaining_min.1</th>
      <th>power_of_shot.1</th>
      <th>knockout_match.1</th>
      <th>remaining_sec.1</th>
      <th>distance_of_shot.1</th>
      <th>goal scoring</th>
      <th>goal scoring2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>27.000000</td>
      <td>38.0</td>
      <td>1610612747</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>50.608000</td>
      <td>54.200000</td>
      <td>38.000000</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8.00000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>22.0</td>
      <td>1610612747</td>
      <td>68.640000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>5.000000</td>
      <td>22.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>20.0</td>
      <td>1610612747</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>20.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>10.00000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>46.000000</td>
      <td>20.0</td>
      <td>1610612747</td>
      <td>10.000000</td>
      <td>70.360000</td>
      <td>0.000000</td>
      <td>46.000000</td>
      <td>20.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>9.00000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>38.0</td>
      <td>1610612747</td>
      <td>9.000000</td>
      <td>43.360000</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>38.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>11.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>26.000000</td>
      <td>37.0</td>
      <td>1610612747</td>
      <td>11.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>26.000000</td>
      <td>37.000000</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>10.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>58.000000</td>
      <td>40.0</td>
      <td>1610612747</td>
      <td>10.000000</td>
      <td>1.000000</td>
      <td>121.608000</td>
      <td>58.000000</td>
      <td>40.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>7.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>33.000000</td>
      <td>21.0</td>
      <td>1610612747</td>
      <td>17.527455</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>33.000000</td>
      <td>21.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>35</th>
      <td>5.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>58.000000</td>
      <td>21.0</td>
      <td>1610612747</td>
      <td>5.000000</td>
      <td>104.360000</td>
      <td>102.608000</td>
      <td>58.000000</td>
      <td>84.400000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>36</th>
      <td>4.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>20.0</td>
      <td>1610612747</td>
      <td>108.640000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>9.000000</td>
      <td>59.400000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>5.00000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>33.000000</td>
      <td>36.0</td>
      <td>1610612747</td>
      <td>5.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>33.000000</td>
      <td>36.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>44</th>
      <td>4.00000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>28.299186</td>
      <td>22.0</td>
      <td>1610612747</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>29.000000</td>
      <td>22.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>49</th>
      <td>3.00000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>19.000000</td>
      <td>37.0</td>
      <td>1610612747</td>
      <td>3.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>19.000000</td>
      <td>37.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>54</th>
      <td>4.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>58.000000</td>
      <td>48.0</td>
      <td>1610612747</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>38.909594</td>
      <td>48.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>59</th>
      <td>3.00000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>48.0</td>
      <td>1610612747</td>
      <td>17.527455</td>
      <td>42.360000</td>
      <td>0.000000</td>
      <td>10.000000</td>
      <td>38.506796</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>61</th>
      <td>0.00000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>47.000000</td>
      <td>25.0</td>
      <td>1610612747</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>108.928000</td>
      <td>38.909594</td>
      <td>25.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>65</th>
      <td>10.00000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>13.000000</td>
      <td>29.0</td>
      <td>1610612747</td>
      <td>10.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>38.909594</td>
      <td>29.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>66</th>
      <td>9.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>32.0</td>
      <td>1610612747</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>6.000000</td>
      <td>32.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>70</th>
      <td>5.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>15.000000</td>
      <td>39.0</td>
      <td>1610612747</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>77.800000</td>
      <td>15.000000</td>
      <td>39.000000</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>3.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>11.000000</td>
      <td>45.0</td>
      <td>1610612747</td>
      <td>3.000000</td>
      <td>15.421244</td>
      <td>16.223872</td>
      <td>29.200000</td>
      <td>45.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>75</th>
      <td>11.00000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>42.0</td>
      <td>1610612747</td>
      <td>11.000000</td>
      <td>41.600000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>38.506796</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>11.00000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>49.000000</td>
      <td>27.0</td>
      <td>1610612747</td>
      <td>34.640000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>28.200000</td>
      <td>13.400000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>84</th>
      <td>9.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>48.000000</td>
      <td>27.0</td>
      <td>1610612747</td>
      <td>9.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>48.000000</td>
      <td>27.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>85</th>
      <td>7.00000</td>
      <td>2.567101</td>
      <td>0.000000</td>
      <td>26.000000</td>
      <td>33.0</td>
      <td>1610612747</td>
      <td>7.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>26.000000</td>
      <td>33.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>86</th>
      <td>6.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>36.000000</td>
      <td>38.0</td>
      <td>1610612747</td>
      <td>6.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>36.000000</td>
      <td>38.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>91</th>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>47.000000</td>
      <td>35.0</td>
      <td>1610612747</td>
      <td>34.640000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>47.000000</td>
      <td>35.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>94</th>
      <td>4.00000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>25.000000</td>
      <td>33.0</td>
      <td>1610612747</td>
      <td>4.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>114.200000</td>
      <td>94.400000</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>3.00000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>15.000000</td>
      <td>38.0</td>
      <td>1610612747</td>
      <td>3.000000</td>
      <td>114.360000</td>
      <td>39.608000</td>
      <td>15.000000</td>
      <td>38.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>103</th>
      <td>4.00000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>16.000000</td>
      <td>37.0</td>
      <td>1610612747</td>
      <td>4.000000</td>
      <td>3.000000</td>
      <td>102.928000</td>
      <td>16.000000</td>
      <td>72.400000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>112</th>
      <td>0.00000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>43.000000</td>
      <td>41.0</td>
      <td>1610612747</td>
      <td>0.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>45.721600</td>
      <td>50.400000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>30567</th>
      <td>4.85228</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>32.000000</td>
      <td>35.0</td>
      <td>1610612747</td>
      <td>113.640000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>32.000000</td>
      <td>35.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30569</th>
      <td>11.00000</td>
      <td>2.000000</td>
      <td>0.138436</td>
      <td>49.000000</td>
      <td>20.0</td>
      <td>1610612747</td>
      <td>11.000000</td>
      <td>117.360000</td>
      <td>1.000000</td>
      <td>49.000000</td>
      <td>20.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30580</th>
      <td>8.00000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>36.0</td>
      <td>1610612747</td>
      <td>8.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>93.200000</td>
      <td>36.000000</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30583</th>
      <td>2.00000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>28.299186</td>
      <td>44.0</td>
      <td>1610612747</td>
      <td>2.000000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>15.000000</td>
      <td>44.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30590</th>
      <td>4.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>41.000000</td>
      <td>29.0</td>
      <td>1610612747</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>41.000000</td>
      <td>29.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30593</th>
      <td>11.00000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>36.000000</td>
      <td>20.0</td>
      <td>1610612747</td>
      <td>11.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>36.000000</td>
      <td>20.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30613</th>
      <td>9.00000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>12.000000</td>
      <td>37.0</td>
      <td>1610612747</td>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>132.785600</td>
      <td>37.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30616</th>
      <td>4.85228</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>31.000000</td>
      <td>40.0</td>
      <td>1610612747</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>31.000000</td>
      <td>40.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30617</th>
      <td>1.00000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>25.0</td>
      <td>1610612747</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>25.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30625</th>
      <td>5.00000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>29.000000</td>
      <td>37.0</td>
      <td>1610612747</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>29.000000</td>
      <td>37.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30629</th>
      <td>3.00000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>55.000000</td>
      <td>36.0</td>
      <td>1610612747</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>78.608000</td>
      <td>55.000000</td>
      <td>36.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30630</th>
      <td>11.00000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>40.0</td>
      <td>1610612747</td>
      <td>11.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>40.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30631</th>
      <td>10.00000</td>
      <td>2.567101</td>
      <td>1.000000</td>
      <td>36.000000</td>
      <td>30.0</td>
      <td>1610612747</td>
      <td>10.000000</td>
      <td>21.600000</td>
      <td>1.000000</td>
      <td>36.000000</td>
      <td>57.400000</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30633</th>
      <td>9.00000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>27.000000</td>
      <td>25.0</td>
      <td>1610612747</td>
      <td>9.000000</td>
      <td>23.360000</td>
      <td>1.000000</td>
      <td>27.000000</td>
      <td>14.400000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30635</th>
      <td>7.00000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>23.0</td>
      <td>1610612747</td>
      <td>7.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>21.000000</td>
      <td>52.400000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30636</th>
      <td>3.00000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>39.000000</td>
      <td>23.0</td>
      <td>1610612747</td>
      <td>106.640000</td>
      <td>25.360000</td>
      <td>1.000000</td>
      <td>39.000000</td>
      <td>23.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30638</th>
      <td>1.00000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>56.000000</td>
      <td>36.0</td>
      <td>1610612747</td>
      <td>17.527455</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>56.000000</td>
      <td>36.000000</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30646</th>
      <td>1.00000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>58.000000</td>
      <td>42.0</td>
      <td>1610612747</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>87.785600</td>
      <td>42.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30648</th>
      <td>0.00000</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>51.000000</td>
      <td>23.0</td>
      <td>1610612747</td>
      <td>17.527455</td>
      <td>5.000000</td>
      <td>1.000000</td>
      <td>35.200000</td>
      <td>23.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30655</th>
      <td>11.00000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>17.000000</td>
      <td>34.0</td>
      <td>1610612747</td>
      <td>11.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>85.200000</td>
      <td>34.000000</td>
      <td>-1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>30659</th>
      <td>11.00000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>28.299186</td>
      <td>36.0</td>
      <td>1610612747</td>
      <td>11.000000</td>
      <td>15.421244</td>
      <td>73.608000</td>
      <td>18.000000</td>
      <td>36.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30664</th>
      <td>6.00000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>26.0</td>
      <td>1610612747</td>
      <td>6.000000</td>
      <td>15.421244</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>26.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30668</th>
      <td>7.00000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>27.000000</td>
      <td>42.0</td>
      <td>1610612747</td>
      <td>7.000000</td>
      <td>68.360000</td>
      <td>54.608000</td>
      <td>27.000000</td>
      <td>42.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30679</th>
      <td>0.00000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>41.000000</td>
      <td>20.0</td>
      <td>1610612747</td>
      <td>71.640000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>41.000000</td>
      <td>20.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30680</th>
      <td>0.00000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>40.000000</td>
      <td>20.0</td>
      <td>1610612747</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>121.928000</td>
      <td>40.000000</td>
      <td>20.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30681</th>
      <td>0.00000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>46.0</td>
      <td>1610612747</td>
      <td>68.640000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>46.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30682</th>
      <td>11.00000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>30.000000</td>
      <td>28.0</td>
      <td>1610612747</td>
      <td>11.000000</td>
      <td>41.360000</td>
      <td>87.608000</td>
      <td>30.000000</td>
      <td>28.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30686</th>
      <td>5.00000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>37.000000</td>
      <td>29.0</td>
      <td>1610612747</td>
      <td>5.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>37.000000</td>
      <td>78.400000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30687</th>
      <td>3.00000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>18.000000</td>
      <td>30.0</td>
      <td>1610612747</td>
      <td>3.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>18.000000</td>
      <td>30.000000</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>30693</th>
      <td>6.00000</td>
      <td>4.000000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>20.0</td>
      <td>1610612747</td>
      <td>6.000000</td>
      <td>64.360000</td>
      <td>1.000000</td>
      <td>5.000000</td>
      <td>20.000000</td>
      <td>1</td>
      <td>-1</td>
    </tr>
  </tbody>
</table>
<p>6268 rows × 13 columns</p>
</div>



# Final output

## using knn model


```python
predictions = knn.predict(newdata1)
predictions
```




    array([0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.,
           0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
           1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1.,
           1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 1., 0.,
           1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
           0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 1., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0.,
           1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0.,
           1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.,
           0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0.,
           0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0.,
           1., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0., 1., 0., 1.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
           1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0., 0., 1.,
           1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
           1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0.,
           1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 1.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
           0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.])



## using naive bayes model


```python
y_hats = gnb.predict(newdata1)
y_hats
```




    array([0., 1., 1., 1., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           0., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 1., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 1., 1., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 1., 0., 1., 1., 0., 1., 0., 1., 1., 0., 1., 1.,
           1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 0.,
           1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           0., 1., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 1., 1., 1., 0., 0.,
           1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
           0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           1., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
           0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.,
           1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 1., 0., 0., 1., 1., 0., 1., 1., 0., 0., 1., 1., 1.,
           1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 1., 0.,
           0., 1., 1., 1., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 1.,
           0., 1., 0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0.,
           1., 1., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1., 0.,
           1., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1.,
           1., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0.,
           1., 1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1.,
           1., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1.,
           1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 1.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1.,
           0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 0., 1., 1.,
           0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0.,
           1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 1.,
           0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 0.,
           1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 1., 1., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 1., 0., 0., 1., 0.,
           1., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0.,
           1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 1., 0., 0., 1., 1., 0.,
           0., 1., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0.,
           1., 1., 1., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 1., 0.,
           0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 1., 1.,
           0., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0.,
           1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0.,
           0., 0., 1., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 1., 1., 0., 1., 1., 1., 0., 1., 0., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           1., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           1., 0., 0., 1., 1., 0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 1.,
           0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0.,
           1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 1.,
           1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
           0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
           0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 1.,
           0., 0., 0., 0., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 1.,
           0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 1., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 1., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1.,
           0., 0., 1., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 1., 1., 0., 0., 1., 1., 0., 0., 1., 0., 1., 1., 1., 0., 0., 1.,
           0., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           0., 0., 0., 1., 1., 1., 1., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 1., 1., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 1., 0., 1., 1., 1., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1.,
           1., 0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 1., 1., 1., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0.,
           0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0.,
           0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 1., 1., 0., 0.,
           1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1.,
           1., 0., 0., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0.,
           0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           1., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 1., 0.,
           0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0., 0., 1.,
           0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 1., 0., 1., 0.,
           1., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 1.,
           0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 1.,
           1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 1., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0.,
           0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1.,
           1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 1., 1., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 0., 0., 1.,
           1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0.,
           1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1.,
           0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 0., 1.,
           0., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0.,
           1., 1., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 1., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1.,
           0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 0., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0.,
           1., 1., 0., 0., 1., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 1.,
           1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0., 1., 0.,
           0., 0., 0., 1., 1., 0., 0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 0.,
           1., 0., 0., 0., 1., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1.,
           0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 1., 0., 0., 0., 1., 1., 1.,
           1., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0.,
           1., 1., 0., 1., 0., 0., 0., 1., 1., 1., 1., 0., 1., 0., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 0., 1., 1.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1.,
           1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 1., 0., 1., 1.,
           1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 1., 0., 0., 0., 1., 1., 0.,
           1., 0., 0., 1., 1., 1., 0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 1.,
           1., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 1., 1., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0.,
           1., 1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0.,
           1., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0., 0.,
           1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 1., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 1., 1., 0., 0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1.,
           0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
           0., 0., 1., 1., 0., 1., 0., 1., 0., 0., 1., 1., 0., 1., 0., 1., 1.,
           0., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1., 1., 0., 0., 0., 1., 0.,
           1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1., 1., 1., 0.,
           1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0.,
           1., 0., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1.,
           0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1.,
           1., 0., 1., 1., 0., 1., 1., 0., 1., 0., 1., 1., 0., 0., 0., 1., 1.,
           0., 1., 0., 0., 0., 1., 0., 1., 1., 0., 0., 1., 1., 1., 1., 1., 0.,
           0., 1., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 0., 1.,
           0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 1., 0., 1., 0., 1., 0., 1., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 1., 0., 1., 1., 0., 0.,
           0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 1., 1., 1., 0., 1.,
           0., 1., 1., 1., 0., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 1., 0.,
           1., 0., 1., 1., 0., 1., 1., 0., 0., 1., 0., 1., 0., 1., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 1., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 1.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 1., 0., 0., 0., 0.,
           1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 1.,
           1., 1., 0., 1., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 1., 1., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           1., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 1.,
           0., 0., 1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 1., 1., 0., 1.,
           0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 1., 1., 0., 0., 1., 0., 0.,
           1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.,
           1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           1., 0., 0., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 0.,
           0., 0., 1., 0., 0., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 1.,
           0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 1., 1., 0., 0., 1., 0., 1., 0., 0., 1., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 1., 0., 1., 0., 1., 1., 1., 0., 0., 1., 0., 1., 0., 1., 1., 1.,
           1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 1., 1., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           1., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 1., 1., 1., 0., 0.,
           1., 0., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1.])




```python
predictions11 = rf_model.predict(newdata1)
predictions11
```




    array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])


