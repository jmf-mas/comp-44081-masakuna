# Technical Test for Applied Scientist (Data Scientist)

__Opportunity No. 44081__

## Overview

This test is used to evaluate your technical ability and proficiency in writing code and modelling business problems. We will review not only your output, but also the process that you used to arrive at your solution. 

> [!NOTE]
> The problem presented here is representative of the __typical__ problem we solve. In the interest of fairness and out of respect for your time, a relatively simple problem has been chosen. In practice, the problems we solve are more complex.


> [!CAUTION]
> If you choose to use an AI tool to prepare your submission, you must disclose what tool and how you used it. Submissions that fail to disclose their use of AI will be disqualified. 


## Challenge

For this exercise, you will use data from the City of Toronto's Open Data Portal. The dataset is related to [ferry tickets](https://open.toronto.ca/dataset/toronto-island-ferry-ticket-counts/) for the Toronto Island Park service.

We have built a simple forecasting model that uses this data to predict the number of redemptions (i.e., people getting on the ferry) at any given time. It does not perform as well as we would like and it does not have any way to account for uncertainty in the forecast.

Your task is to:

1. Improve the first forecasting model for redemptions. Use the Python code provided as a starting point.  (30% of points).
2. Create another forecasting model for the number of sales (i.e., people buying tickets). You may do this in Python or R. (40% of points)

*How you go about the task* is important and we are paying attention to process. (20% of points) You will see some template code has already been started in the repository. You should use this code and built off it with proper development workflows - as if you are collaborating with a colleague.

You are free to make assumptions about the business problem and needs - document these assumptions clearly. 

Finally, you should prepare a short summary of what you did, why it is better than what we provided, and how you have approached the business problem (10% of points). You should provide an accessible summary of your work in bullet points and plain language, as well as a more detailed description in standard prose. Word limits apply (see below).

## Expected Outcomes

- An improved forecasting model for redemptions that can be used to forecast redemption volume on a daily basis.
- A forecasting model for sales.
- Accessible description of the work (bullet points) for a non-technical client,  not exceeding 200 words.
- A more detailed description of the work with technical specifics that does not exceed 500 words.

> [!NOTE]
> Reproducibility is a core focus of our team and all our outputs are held to high standards in this respect. We may run your code to verify your solution. Furthermore, you must only use free and open source tools in your submission to promote reproducibility.  


## Submission

Provide a link to the completed modelling exercise on a GitHub account. You can choose to keep the repository private or public. If it is private, you must share your repository with the [gom-ta GitHub account](https://github.com/gom-ta).

##

Candidate<br>
**Jordan Felicien MASAKUNA**

Date<br>
**12 June 2025**

### Accessible description of the work (bullet points) for a non-technical client

- This code uses forecasting tools, to predict future Sales and Redemption Counts in a time-based data. It organizes and normalizes the past information, and identifies recurring yearly and weekly patterns. Once learned, the model generates future forecasts.
- It is assumed the following:
  - data points are accurate and complete, and that the observed target patterns are representative. 
  - the historical seasonal patterns (weekly, monthly, quarterly) persist. 
  - both target variables align with logistic growth. 
  - there is a single dominant seasonality (e.g., weekly or monthly) and that the series can be made stationary with differencing to model linear relationships and constant variance of residuals.

- Comparing two forecasting approaches of the base model, the first version (which was given) assumes a purely seasonal, non-negative signal, which is less realistic for typical time series as it ignores underlying trends and can lead to underestimation bias, especially after data normalization, making it suitable only for strictly seasonal and non-negative data. 
- In contrast, the second version (our improvement) is more realistic, assuming seasonality around a stable average baseline, thus better capturing real-world fluctuations and resulting in lower bias and more balanced forecasts that align well with normalized data, making it more suitable for general forecasting tasks.

### A more detailed description of the work with technical specifics

**Dataset preparation.**
- After analyzing the statistical characteristics of the target variables, I found that they have large values and high standard deviations. This can pose a problem, as large values may disproportionately influence the loss function, potentially leading to numerical instability, loss function sensitivity and erratic updates during training.
- To mitigate this issue, we normalize the target values within a fixed range. To determine the most suitable range, we conducted experiments with several scaling options and selected the one that yielded the most stable and accurate results. 
- I applied Min-Max normalization to scale the target values into the range [0.5, 1].

**Forecasting models.**
- I improved the baseline model that was provided (I reduced MAPE from at least 0.85 to at most 0.15 for different splits).
- I implemented another forecasting model using the Prophet framework.

**First version of  the base model.** 
- It directly utilizes the ```res.seasonal``` component from ```seasonal_decompose``` and clips any negative values to zero. This implies an assumption that the seasonal contribution itself cannot be negative. 
- It performs no explicit adjustment by the overall mean of the target. It explicitly applies ```max(0, x)``` to the seasonal component to remove negative values. 
- The prediction for a given day of the year is simply its non-negative seasonal value. 
- High risk of systematic underestimation bias. Clipping negative seasonal variations and the absence of a mean adjustment can significantly distort the predictions .
- Outputs are inherently greater than or equal to zero due to the clipping.

*Improved version of  the base model.* 
- It uses a centered version of ```res.seasonal``` (i.e., ```seasonal - seasonal.mean()```). This assumes the seasonal effect is a deviation around a baseline rather than always being non-negative.
- It adjusts predictions by adding the overall mean (base level) of the target from the training data. This aims to restore the predictions to the true scale of the data.
- It does not apply clipping during the seasonal component's calculation, assuming that centering the seasonal component and adding the base mean will generally avoid unrealistic negative values in the final prediction.
- The final prediction is calculated as ```mean(target) + centered_seasonality``` for the corresponding day of the year.
- Lower bias risk. Centering the seasonal component and re-introducing the base level (mean) helps in restoring the original scale and provides more balanced and aligned forecasts, leading to more realistic estimations.
- Outputs may include negative values if the ```centered_seasonality``` and ```mean(target)``` combine to a negative sum. 

**Prophet model.**
- Model configuration. A Prophet object is instantiated with several key parameters:
   - growth='logistic': This is specifically chosen to model the target variable's trend with explicit saturation limits, aligning with the assumption that the target (e.g., scaled sales count 'sc') operates within defined upper (cap=1.0) and lower (floor=0.5) bounds.
   - yearly_seasonality=True, weekly_seasonality=True: Configured to capture annual and weekly recurring patterns, which are typical in sales time series.
   - daily_seasonality=False: Deliberately excluded, as daily granularity is usually sufficient for sales counts without needing sub-daily patterns.
   - seasonality_mode='additive': Specifies that seasonal effects are linearly added to the trend component.


### Environment

You will need to prepare your environment to run this code. The dataset should be placed in a folder named */data* at the project's root.

Create a Conda environment

```python
conda create --name comp_44081 python=3.8
```

Activate your environment

```python
conda activate comp_44081
```

Libs installation

```python
pip install -r requirements.txt
```

Add the created environment into Jupyter Notebook

```python
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=comp_44081
```
