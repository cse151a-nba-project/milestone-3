# CSE 151A Milestone 3: First Model

Repo link: https://github.com/cse151a-nba-project/milestone-3/
Data link: https://github.com/cse151a-nba-project/data/

Group Member List: 

- Aaron Li all042\@ucsd.edu
- Bryant Jin brjin\@ucsd.edu
- Daniel Ji daji\@ucsd.edu
- David Wang dyw001\@ucsd.edu
- Dhruv Kanetkar dkanetkar\@ucsd.edu
- Eric Ye e2ye\@ucsd.edu
- Kevin Lu k8lu\@ucsd.edu
- Kevin Shen k3shen\@ucsd.edu
- Max Weng maweng\@ucsd.edu
- Roshan Sood rosood\@ucsd.edu

*Planned* Abstract, for reference: 

Although sports analytics captured national attention only in 2011 with the release of Moneyball, research in the field is nearly a century old. Until relatively recently, this research was largely done by hand; however, the heavily quantitative nature of sports analytics makes it an attractive target for machine learning. This paper explores the application of advanced machine learning models to predict team performance in National Basketball Association (NBA) regular season and playoff games. Several models were trained on a rich dataset spanning 73 years, which includes individual player metrics, opponent-based performance, and team composition. The core of our analysis lies in combining individual player metrics, opponent-based game performances, and team chemistry, extending beyond traditional stat line analysis by incorporating nuanced factors. We employ various machine learning techniques, including neural networks and gradient boosting machines, to generate predictive models for player performance and compare their performance with both each other and traditional predictive models. Our analysis suggests that gradient boosting machines and neural networks significantly outperform other models. Neural networks demonstrate significant effectiveness in handling complex, non-linear data interrelations, while gradient boosting machines excel in identifying intricate predictor interactions. Our findings emphasize the immense potential of sophisticated machine learning techniques in sports analytics and mark a growing shift towards computer-aided and computer-based approaches in sports analytics.

# 1. Finish Major Preprocessing

Nearly all of our preprocessing was done in the previous milestone. In this milestone, we went ahead and took advanced player data from teams and try to predict their win percentage. For each team from 1990 to 2023, we identified the top 8 players in minutes played per game, and extracted the stats: PER, Win shares per 48, Usage Percentage, BPM, and VORP, which we found to have the highest correlation with win percentage. We arranged this data so that we had 40 input features, 5 stats for each of the top 8 players and we could use this to predict win loss percentage using a regression model.

# 2. Train First Model

We decided to use a simple linear regression model as our first model as a proof of concept. We wanted to keep it simple at first so that we can easily pivot if needed later. Additionally, since Neural Networks are essentially combinations of linear regressions and non linearities, a simple linear regression would give us a good baseline model performance to test future models against.

# 3. Evaluate your model compare training vs test error

Model Training MSE: 21.829770676150357
Model Training MAE (Mean Absolute Error): 3.6668298317069015
Model Training R^2 (Coeff of Deter.): 0.9101089445076146

Model Testing MSE: 24.089338248039507
Model Testing MAE (Mean Absolute Error): 3.8786683795508496
Model Testing R^2 (Coeff of Deter.): 0.8878729896447337

As illustrated above, our testing mestrics are slightly worse than the training metrics, but not significantly worse. This indicates that there is no significant overfitting occuring. The MSE of our model is still quite high, suggesting that our model cannot adequately predict the WL% to the desired accuracy. However, the R^2 scores of 0.91 and 0.88 are very promising, indicating that a very high percentage of the variation in WL% has been captured by the linear regressor, indicating that with a more complex model we may be able to further improve the performance of our model.

# 4. Where does your model fit in the fitting graph.

We still have quite a bit of error (our MSE is > 20 for both testing and training) while a good value would be around 1-5 (an approximation, we're not exactly sure how good we can get it, but we believe it can be improved). To interpret this MSE, an error of 9 would translate to an average difference of 3% between predicted winning percentage and actual, which means 2-3 more / less games won.  Based on the test and training errors themselves, we can't differentiate if we're underfitting or having a good fit, because they're quite similar, with the test error only being slightly greater than the training error (~24.1 vs ~21.8). We believe the model is underfitting and has a high error that can be improved upon using more complex models.

# 5. Future plans

Based on the analysis of your current model's performance with linear regression, which indicates underfitting given the high mean squared error (MSE) rates for both testing and training, we are considering the following Polynomial Regression and Deep Neural Network (DNN) Regression as the next two models to potentially improve the predictive accuracy of our model:

We want to experiment with polynomial regression next because it can model relationships between variables that are not linear, capturing more complex patterns in the dataset. Given the nature of sports statistics and team performance metrics, the relationship between the input features (advanced player statistics) and the output (team win percentages) might not be linear but could have a polynomial relationship. Polynomial regression can provide a better fit to the dataset by adding degrees of freedom through higher-order terms, potentially reducing the underfitting observed with the linear regression model.

Planned Strategy:
We plan to experiment with different polynomial degrees to find the optimal level of complexity that improves the model's performance without overfitting. This involves transforming our input features into polynomial features and then applying linear regression to these transformed features.
Cross-validation will be used to evaluate the model's performance and prevent overfitting by selecting the degree of the polynomial that provides the best generalization to unseen data.

Deep Neural Network (DNN) Regression
Because Deep Neural Networks are capable of learning very complex patterns in large datasets, we believe it is suitable for our problem where traditional linear models fail to capture the underlying relationships. DNN regression can model intricate interactions between player statistics and their impact on team success rates through its multiple hidden layers and non-linear activation functions. This flexibility allows DNNs to potentially offer significant improvements over simpler models by accurately capturing the non-linear and complex relationships in our dataset.

Planned Strategy:
We aim to design a DNN with several hidden layers and neurons, experimenting with different architectures to find the most effective configuration for our dataset. Regularization techniques (like dropout and L2 regularization) and optimization algorithms will be also used to enhance learning and prevent overfitting.
Feature scaling and normalization will be crucial preprocessing steps
(that we've done in milestone 2) to ensure that the DNN model trains efficiently.
We will also use a portion of the dataset for validation during training to monitor for overfitting and adjust the model's parameters accordingly.

# 7. Conclusion section: What is the conclusion of your 1st model? What can be done to possibly improve it?

Our initial approach to predicting team win percentages based on advanced player statistics utilized a linear regression model. The analysis of this model's performance revealed a higher than desired mean squared error (MSE) for both training and testing datasets, indicating that our model is not fitting the data as well as we hoped. With training and testing MSE values greater than 20, where a more acceptable range would likely be around 1-5, it's clear that our model is underperforming. The similarity of the test and training errors suggests that our model is underfitting, rather than perfectly fitting or overfitting the data.

Possible Improvements
To address the underfitting and improve our model's performance, we propose several strategies:

Feature Engineering and Selection:
Re-evaluate the input features used in the model. Advanced player statistics are complex and can have nonlinear relationships with team success. Exploring additional or different metrics, as well as interactions between features, might uncover more relevant predictors.
Polynomial features can be introduced to model nonlinear relationships without immediately moving to more complex models like DNNs.

Model Complexity:
Increase the complexity of our predictive model to better capture the nuances in the data. This involves transitioning from a simple linear regression to models capable of learning nonlinear patterns, such as Polynomial Regression and Deep Neural Networks (DNNs), as discussed earlier.

Cross-Validation and Hyperparameter Tuning:
Use cross-validation techniques to more accurately evaluate model performance and prevent overfitting. This approach will be particularly important when experimenting with polynomial regression degrees and DNN architectures.
Hyperparameter tuning can optimize the model's performance, especially for DNNs, where parameters such as the number of layers, number of neurons, learning rate, and regularization terms play significant roles.

Regularization Techniques:
Apply regularization techniques (like Lasso, Ridge, or ElasticNet for polynomial regression, and dropout or batch normalization for DNNs) to reduce the risk of overfitting by penalizing overly complex models.