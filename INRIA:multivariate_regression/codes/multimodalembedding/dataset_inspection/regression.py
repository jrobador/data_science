from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

def regression_task (f_gamma_train, g_beta_train, f_gamma_test, g_beta_test):
    krd = KernelRidge(alpha=1.0)
    krd.fit(f_gamma_train, g_beta_train)
    g_beta_pred = krd.predict(f_gamma_test)
    mse = mean_squared_error(g_beta_pred, g_beta_test)
    r2 = r2_score(g_beta_pred, g_beta_test)
    print("Kernel Ridge Squared Error:", mse)
    print("Kernel Ridge R-squared:", r2)

    kernel = DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel,
        random_state=0).fit(f_gamma_train, g_beta_train)
    g_beta_pred = gpr.predict(f_gamma_test)
    mse = mean_squared_error(g_beta_pred, g_beta_test)
    r2 = r2_score(g_beta_pred, g_beta_test)
    print("GaussianProcessRegressor Squared Error:", mse)
    print("GaussianProcessRegressor R-squared:", r2)