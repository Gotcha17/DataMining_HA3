# Part 1 ------------------------------------------------------------------

# Generating samples of size n = 100 of all random variables
n <- 100
set.seed(1)
u <- c(rnorm(n, mean = 0, sd = 1))
set.seed(100)
x1 <- c(rnorm(n, mean = 0, sd = 1))
set.seed(300)
x2 <- c(rnorm(n, mean = 0, sd = 1))
set.seed(900)
x3 <- c(rnorm(n, mean = 0, sd = 1))
# Calculating y with already generated r.v.
y <- 1+0.5*x1^2+u                                      # Calculating y with the given model
Y <- as.matrix(y)                                      # Transforming the dependent variable to a matrix
X <- as.matrix(cbind(1,x1,x2,x3))                      # Combining all independent variables to a matrix
X_sq <-  X^2                                           # Needed as we have x1^2+x2^2+x3^2                                         # Since X_sq is modified this is used for final est.
lambda <- 2                                            # This lambda valur is set for testing


# Part 2 ------------------------------------------------------------------

# Defining the Ridge Regression function
ridge_reg <- function(beta, Y, X, lambda){
  t(Y - X %*% beta)%*%(Y - X %*% beta) + lambda*t(beta)%*%beta
}

## Standardizing the predictors i.e. x_ij=x_ij/(mean(x_ij-mean(x)_j)^2)^0.5
# Xscale <- drop(rep(1/n, n) %*% X_sq^2)^0.5
# X_sq <- X_sq/rep(Xscale, rep(n, dim(X)[2]))
## Since data is of the same magnitude, it is not needed to apply the standardization.

Ridge_RSS <- optim(par=c(0,0,0,0), fn=ridge_reg, Y=Y, X=X_sq, lambda=lambda) 
beta_est <- Ridge_RSS$par
names(beta_est) <- c("Intercept", "x1", "x2", "x3")
beta_est

# Checking the result with Analytical approach
I_p <- diag(dim(X)[2])
beta_analytical<-as.vector(solve(t(X_sq)%*%X_sq+I_p*lambda)%*%(t(X_sq)%*%Y))
names(beta_analytical) <- c("Intercept", "x1", "x2", "x3")
beta_analytical

# Results are similar, therefore estimated results should be correct


# Part 3 ------------------------------------------------------------------

# Cross validation
# input "k" as numeric, "data" as dataframe
CV_Ridge <- function(k, data, lambda){
  data_range <- length(data[,1])                       # Setting range of the whole sample
  k_fold_range <- data_range/k                         # Setting the range of the k-folds
  data <- data[sample(nrow(data)), ]                   # Shuffling data-set
  rownames(data) <- NULL                               # Disabling rownames
  
  # Regressing by leaving out the k-th fold out of the data
  MSE <- c()                                           # Initializing vector for storing result the MSEs
  for (l in 0:(k-1)){                                  # For-loop over all ks
    test <- ((l*k_fold_range+1):((l+1)*k_fold_range))  # Test set range is set
    Y_train <- as.matrix(data[-test,1])                # Train set of ys is set
    rownames(Y) <- NULL                                # Disabling rownames
    X_train <- as.matrix(data[-test,2:(dim(X)[2]+1)])  # Train set of xs is set
    rownames(X) <- NULL                                # Disabling rownames
    # Running ridge regression function
    Ridge_RSS <- optim(par=c(0,0,0,0), fn=ridge_reg, Y=Y_train, X=X_train, lambda=lambda) 
    beta_est <- Ridge_RSS$par                          # Exctracting estimates
    # Predicting values for k-th fold values
    X_test <- as.matrix(data[test,2:(dim(X)[2]+1)])    # Test set of xs is set
    Y_test <- as.matrix(data[test,1])                  # Test set of ys is set
    y_hat <- X_test%*%beta_est                         # Predicting y value
    # Calculating MSEs for k-th fold predicted values
    MSE[l+1] <- mean((Y_test-y_hat)^2)
  }
  CV_MSE <- mean(MSE)
  return(CV_MSE)
}

# Finding lambda with least MSE from CV with K=10 K-fold 
grid <- 10^seq(10,-2,length.out = 1000)                # Setting parameters for lambda
linear_reg_data <- data.frame(Y, X_sq)                 # Creating data frame with from matrices Y and X
MSE <- c()
for (i in 1:length(grid)){
  MSE[i] <- CV_Ridge(10, linear_reg_data, grid[i])
}
opt_lambda_index <- which.min(MSE)
opt_lambda <- grid[opt_lambda_index]


# Part 4 ------------------------------------------------------------------

# As we can see, optimal lambda is not very large and therefore regressors are not greatly pushed towards zero
opt_lambda

## Here we can see which values for the estimators are calculated. Values for beta2 and beta3 are relatively
# near the zero in comparison to beta0 and beta1. That was expected as the true model does not contain 
# this regressors at all, but they are still not zero, as it is inpossible to do so due to the existend form
# of the penalty part in the Ridge regression. Beta0 is estimated at 1.10 and beta1 is estimated at 0.44,
# since in the true model beta0 is 1 and beta1 is 0.5, this results are close to the true model.
Ridge_RSS <- optim(par=c(0,0,0,0), fn=ridge_reg, Y=Y, X=X_sq, lambda=opt_lambda)$par
names(Ridge_RSS) <- c("Intercept", "x1", "x2", "x3")
Ridge_RSS
