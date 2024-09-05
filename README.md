
# Driving Behavior Classifier

This project aims to classify driving behavior into three categories: normal, aggressive, or slow. The classification is based on key metrics gathered from a vehicle’s On-Board Diagnostics (OBD) system, including:

	•	Acceleration
	•	Speed
	•	Instantaneous Engine Power (calculated based on fuel consumption)
	•	Engine RPM

### Approach

The process begins with a sample of “normal” driving data, which serves as a baseline. Test driving data is then collected, and all key metrics are Z-score normalized. This standardization helps detect deviations from the normal driving pattern and classify each driving sample accordingly.

### Machine Learning Models

Once the driving behavior is classified, two machine learning models are trained to predict driving styles:

	1.	Logistic Regression
	2.	Neural Network

### Evaluation and Visualization

Both models are evaluated on the classified dataset. Their performances are visualized through confusion matrices, as well as other relevant plots, showing each model’s accuracy and capability in classifying driving styles. The results provide a clear comparison of how well each model predicts normal, aggressive, and slow driving behaviors.
## Lessons Learned

### 1. Creating a Labeled Dataset

One of the most valuable lessons I learned during this project is the importance of establishing a labeled dataset before training any machine learning models. 

Since there wasn’t an existing dataset for driving behavior, I had to label the data myself. I did this by creating a script in python (main.py); however, this process of manual classification allowed me to have a deeper understanding of the driving patterns and behaviors that the models would later predict. 

Without properly labeled data, even the most advanced machine learning algorithms would struggle to produce meaningful results. This is why a rather long portion of my time spent on this project was dedicated to developing a script that would accurately represent the data.

### 2. Importance of Z-Score Normalization

Another key takeaway was the significance of Z-score normalization, especially when dealing with data that comes in different units and magnitudes. 

The metrics from the OBD system, such as speed, acceleration, and engine RPM, are measured on very different scales. Without normalization, these differences would cause models to overemphasize certain metrics while underweighting others, leading to biased classifications. 

Z-score normalization helps by converting all features into the same scale, making the model more robust and capable of generalizing across different types of driving behavior. 

This step was crucial in ensuring that all metrics were treated equally (before assigning weights) in the classification process.

### 3. Logistic Regression vs. Neural Networks

Throughout the development of this project, I gained insight into the differences between Logistic Regression and Neural Networks for multi-class classification. 

Logistic Regression is a (relatively) simple but effective model that works well for linearly separable data. In this program, it provided interpretable results and performed well in classifying driving behavior into one of the three available classes. 

On the other hand, Neural Networks, while more complex, demonstrated greater flexibility and the ability to capture non-linear relationships between the metrics. The downside of Neural Networks is their need for more computational resources and tuning to avoid overfitting. 

Logistic Regression was easier to implement and interpret but lacked the same predictive power in more complex scenarios. Both models have their strengths, and choosing one of them over the other depends on the complexity of the problem at hand and the amount of data available.



## Sample Results
### Normal Data Sample (top) & Driving Data Sample (bottom)
![normalDataPlot](https://github.com/user-attachments/assets/c5ae42bc-39c0-43f1-ac62-7431b5edaf38)
![drivingDataPlot](https://github.com/user-attachments/assets/0611d5c2-43ad-4e0c-8e78-a888ab9d6d22)

### Neural Network Results (top) & Logisitc Regression Results (bottom)
![neuralNetworkConfMatrix](https://github.com/user-attachments/assets/7db5b57b-4ebf-4f07-b679-cf72925b772a)
![neuralNetworkClassifications](https://github.com/user-attachments/assets/7bdff138-095c-479e-a3c3-dc96fe93448b)
![logRegConfMatrix](https://github.com/user-attachments/assets/41dc715a-1f43-4cb7-9aa4-e2ba4d8dc770)
![logRegClassifications](https://github.com/user-attachments/assets/e3bb34db-79c1-40fa-957d-5a552e439c3a)

## Authors

- [@Boussi7](https://www.github.com/Boussi7)

