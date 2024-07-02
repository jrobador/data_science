from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_absolute_error
import numpy as np

def r2_train_score(net,X,y):
    y_pred = net.predict(X)
    return r2_score(y,y_pred)

def r2_test_score(net,X,y):
    y_pred = net.predict(X)
    return r2_score(y,y_pred)

def mape_train_score(net, x, y):
    y_pred = net.predict(x)
    mape = - mean_absolute_percentage_error(y, y_pred)
    return mape

def mape_test_score(net, x, y):
    y_pred = net.predict(x)
    mape = - mean_absolute_percentage_error(y, y_pred)
    return mape

def mae_train_score(net, x, y):
    y_pred = net.predict(x)
    mse = mean_absolute_error(y, y_pred)
    return mse

def mae_test_score(net, x, y):
    y_pred = net.predict(x)
    mse = mean_absolute_error(y, y_pred)
    return mse


def mape_train_mme(net, x, y):
    x_values = []
    y_values = []

    for sample in x:
        data_dict, y = sample
        x_values.append(data_dict['x'])
        y_values.append(y)

    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    y_pred = net.module_.predict_y_from_x(x_values).cpu().numpy()
    mape = mean_absolute_percentage_error(y_values,y_pred)
    return mape
