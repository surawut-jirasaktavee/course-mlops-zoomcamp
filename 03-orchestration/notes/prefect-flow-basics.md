# Prefect Flow Basic
---

**Sample usecase** from `EVIDENTLY AI` blog about model accuracy
The model accuracy will decrease overtime.

![](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/03-orchestration/images/model_depay.png)

People call it **model drift, decay, or staleness**.

Things change, and model performance degrades over time. The ultimate measure is the model quality metric. It can be accuracy, mean error rate, or some downstream business KPI, such as click-through rate.

![model decay](https://github.com/surawut-jirasaktavee/course-mlops-zoomcamp/blob/main/03-orchestration/images/model_decay1.png)

To avoid **model decay** we can regularly re-train the model and then can publish new model to production to keep model performance as it should be.

ref: [Data and concept drift](https://evidentlyai.com/blog/machine-learning-monitoring-data-and-concept-drift)


