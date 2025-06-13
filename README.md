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


## Environment

You will need to install several packages to run the existing code

```python
pip install pandas seaborn matplotlib statsmodels scikit-learn
```
