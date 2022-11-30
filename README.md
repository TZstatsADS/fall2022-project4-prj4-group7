# Project 4: Machine Learning Fairness

### [Project Description](doc/project4_desc.md)

Term: Fall 2022

+ Team #7
+ Project title: Study of Select Machine Learning Fairness
+ Team members
	+ Cheng, Louis yc3733@columbia.edu
	+ Kolluri, Sameer ssk2258@columbia.edu
	+ Lee, Sangmin sl4876@columbia.edu
	+ Wang, Fu fw2376@columbia.edu
	+ Wu, Judy dw2936@columbia.edu

+ Project summary: We explored the implementation of two methods of machine learning fairness by applying these methods to the COMPAS dataset. One method was a in-processing method: Maximizing fairness under accuracy constraints (referred as A3) and the other was a pre-processing approach: Handling Conditional Discrimination (A6). Our goal was to optimize accuracy and calibration (getting close to an equal percentage of African-Americans and Caucasians being labeled correctly) in determining whether an individual would become a recidivist - a re-offender. A3 did this by maximizing the fairness of the model subject to constraints on accuracy determined by the unconstrained model. A6 preprocess the dataset by two different approaches, Local Massaging (LM) and Local Preferential Sampling (LPS). LM identifies the instances that are close to the decision boundary and changes the values of their labels to the opposite. LPS deletes the ‘wrong’ instances that are close to the decision boundary and duplicates the instances that are ‘right’ and close to the boundary. Although the accuracy in the models implemented has decreased from the baseline models, the goal of achieving fair predictions was fulfilled.
	
**Contribution statement**: [default] All team members contributed to the data processing steps. Louis, Sangmin, Judy and Sameer worked on Algorithm A3. Fu worked on Algorithm A6. All team members approve our work presented in this GitHub repository including this contributions statement. 

Following [suggestions](http://nicercode.github.io/blog/2013-04-05-projects/) by [RICH FITZJOHN](http://nicercode.github.io/about/#Team) (@richfitz). This folder is orgarnized as follows.

```
proj/
├── lib/
├── data/
├── doc/
├── figs/
└── output/
```

Please see each subfolder for a README file.
