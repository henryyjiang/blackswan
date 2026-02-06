This repository contains demo notebooks for the equations used in the model. 

### How to Run
1. Log into QuantConnect and create a new project.
2. In the left hand sidebar, click the upload icon and import the .ipynb files.
3. Select a QuantConnect kernel, and then you can run the cells.
4. You can observe the behavior of all the features implemented that the model takes in; model details and performance are currently private as the model is actively being tested for live deployment.
5. Behaviors documented in mandelbrotian_features.ipynb:
- Hurst exponent (with DFA)
- Hill's tail index (and other tail fatness estimators)
- RV/IV ratio
- Simple multifractal spectrum width
- 
6. Behaviors documented in technical_features.ipynb:
- RSI
- Bollinger Band %B
- ATR
- Volume Ratio
- VWAP Deviation
- Put/Call Ratio
- IV Skew
- Term Structure Slope
