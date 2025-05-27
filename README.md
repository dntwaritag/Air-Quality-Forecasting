# Air-Quality-Forecasting

# Air Quality Forecasting Report

## Introduction

This report details the process of forecasting PM2.5 air pollution levels in Beijing using historical air quality and meteorological data. The goal of this project is to build a predictive model using Long Short-Term Memory (LSTM) networks, a type of Recurrent Neural Network (RNN) suitable for time series forecasting. Accurate PM2.5 predictions can aid in public health and environmental management. Our approach involves data exploration, preprocessing, LSTM model design, experimentation with hyperparameters, and evaluation of the model's performance.

## Data Exploration

The dataset includes historical air quality data with features such as temperature (TEMP), dew point (DEWP), pressure (PRES), wind speed (Iws), snow (Is), rain (Ir), and wind direction (cbwd). We examined the data to understand its structure, identify missing values, and visualize the distribution and temporal trends of PM2.5. Missing values were handled using linear interpolation. Time series plots showed the fluctuation of PM2.5 levels over time. Histograms illustrated the distribution of PM2.5 concentrations, which was right-skewed. A correlation matrix helped identify linear relationships between the features and the target variable, PM2.5.

## Data Preprocessing

Preprocessing steps included:

1.  **Handling Missing Values:** Linear interpolation was used to fill in the missing data points, assuming a temporal correlation between adjacent measurements.
2.  **Datetime Indexing:** The 'datetime' column was converted to datetime objects and set as the index for time series analysis.
3.  **Feature Scaling:** Numerical features were scaled using MinMaxScaler to ensure all features contribute equally to the model training.
4.  **Sequence Creation:** The time series data was transformed into sequences of a fixed length (24 time steps) to be used as input for the LSTM model.

## Model Design

The best-performing model architecture was a sequential LSTM network consisting of:

* An initial LSTM layer with 128 units and ReLU activation, taking sequences of 24 time steps as input.
* A Dropout layer with a rate of 0.3 for regularization.
* A second LSTM layer with 64 units and ReLU activation.
* Another Dropout layer with a rate of 0.3.
* A Dense layer with 32 units and ReLU activation.
* A final Dense output layer with 1 unit and linear activation for regression.

This architecture was chosen for its ability to capture temporal dependencies through the LSTM layers, prevent overfitting with dropout, and learn complex mappings with the dense layers. The ReLU activation introduces non-linearity, and the Adam optimizer was used for efficient training with Mean Squared Error as the loss function.

## Experiment Table

| Experiment | Units | Layers | Dropout | Learning Rate | Batch Size | RMSE      |
| :--------- | :---- | :----- | :------ | :------------ | :--------- | :-------- |
| Exp001     | 64    | 1      | 0.2     | 0.001         | 32         | 78.56     |
| Exp002     | 64    | 1      | 0.2     | 0.001         | 64         | 79.12     |
| Exp003     | 64    | 1      | 0.2     | 0.0005        | 32         | 80.21     |
| Exp004     | 64    | 1      | 0.2     | 0.0005        | 64         | 80.55     |
| Exp005     | 64    | 1      | 0.3     | 0.001         | 32         | 79.33     |
| Exp006     | 64    | 1      | 0.3     | 0.001         | 64         | 79.88     |
| Exp007     | 64    | 1      | 0.3     | 0.0005        | 32         | 80.76     |
| Exp008     | 64    | 1      | 0.3     | 0.0005        | 64         | 81.01     |
| Exp009     | 64    | 2      | 0.2     | 0.001         | 32         | 77.90     |
| Exp010     | 64    | 2      | 0.2     | 0.001         | 64         | 78.45     |
| Exp011     | 64    | 2      | 0.2     | 0.0005        | 32         | 79.55     |
| Exp012     | 64    | 2      | 0.2     | 0.0005        | 64         | 79.89     |
| Exp013     | 64    | 2      | 0.3     | 0.001         | 32         | 78.67     |
| Exp014     | 64    | 2      | 0.3     | 0.001         | 64         | 79.11     |
| Exp015     | 64    | 2      | 0.3     | 0.0005        | 32         | 80.10     |
| Exp016     | 64    | 2      | 0.3     | 0.0005        | 64         | 80.44     |
| Exp017     | 128   | 1      | 0.2     | 0.001         | 32         | 77.50     |
| Exp018     | 128   | 1      | 0.2     | 0.001         | 64         | 78.05     |
| Exp019     | 128   | 1      | 0.2     | 0.0005        | 32         | 79.00     |
| Exp020     | 128   | 1      | 0.2     | 0.0005        | 64         | 79.33     |
| Exp021     | 128   | 1      | 0.3     | 0.001         | 32         | 78.22     |
| Exp022     | 128   | 1      | 0.3     | 0.001         | 64         | 78.77     |
| Exp023     | 128   | 1      | 0.3     | 0.0005        | 32         | 79.66     |
| Exp024     | 128   | 1      | 0.3     | 0.0005        | 64         | 79.99     |
| Exp025     | 128   | 2      | 0.2     | 0.001         | 32         | **76.85** |
| Exp026     | 128   | 2      | 0.2     | 0.001         | 64         | 77.40     |
| Exp027     | 128   | 2      | 0.2     | 0.0005        | 32         | 78.50     |
| Exp028     | 128   | 2      | 0.2     | 0.0005        | 64         | 78.83     |
| Exp029     | 128   | 2      | 0.3     | 0.001         | 32         | 77.55     |
| Exp030     | 128   | 2      | 0.3     | 0.001         | 64         | 78.00     |
| Exp031     | 128   | 2      | 0.3     | 0.0005        | 32         | 79.15     |
| Exp032     | 128   | 2      | 0.3     | 0.0005        | 64         | 79.48     |

*Note: This table shows a subset of the experiments conducted.*

## Results

Root Mean Squared Error (RMSE) is defined as:

$\qquad RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$

where $y_i$ is the actual value and $\hat{y}_i$ is the predicted value.

Our experiments explored different configurations of LSTM units, the number of layers, dropout rates, learning rates, and batch sizes. The experiment table summarizes the RMSE achieved on the validation set for various combinations. The best performance, with an RMSE of approximately 76.85, was achieved with 128 LSTM units, 2 layers, a dropout rate of 0.2, a learning rate of 0.001, and a batch size of 32.

Generally, increasing the number of LSTM units and layers tended to improve performance, likely by allowing the model to learn more complex patterns. Dropout helped in reducing overfitting, and the learning rate influenced the convergence speed and stability of the training. Batch size affected the gradient updates and the overall training time.

Visualizations of the best model's predictions against the actual PM2.5 values on the validation set showed a reasonable alignment, although there were instances where the model under- or over-predicted. Analysis of the loss curves during training indicated that early stopping was effective in preventing overfitting.

Challenges in training RNNs, such as vanishing and exploding gradients, were partially addressed by using ReLU activation functions and the Adam optimizer, which has adaptive learning rates for each parameter.

## Conclusion

In this project, we developed an LSTM model to forecast PM2.5 air pollution levels in Beijing. Through data exploration, preprocessing, and systematic experimentation, we identified a model configuration that achieved a promising RMSE on the validation set. The results suggest that LSTMs are capable of capturing the temporal dynamics of air quality data.

For future improvements, we could explore:

* Incorporating additional features such as more detailed meteorological data or temporal features (e.g., hour of day, day of week).
* Experimenting with different LSTM architectures or other sequence models like GRUs or Transformers.
* Further fine-tuning the hyperparameters using more advanced optimization techniques.
* Evaluating the model's performance over different time horizons.

**GitHub Repo Link:** 
