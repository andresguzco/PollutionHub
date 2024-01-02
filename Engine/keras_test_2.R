library(tensorflow)
library(keras)

iN <- 1e5
x_train <- as_tensor(sort(runif(iN)) - 0.5, shape = iN)
y_train <- 2 + as_tensor(cos(4 * pi * x_train), shape = iN)

xx <- rnorm(iN, mean = 5)
yy <- xx * y_train + rnorm(iN, sd = 0.01)
yy_train <- cbind(yy, xx)

model <- keras_model_sequential(input_shape = c(1, 1)) %>%
  layer_flatten() %>%
  layer_dense(6, activation = "sigmoid") %>%
  layer_dense(3, activation = "sigmoid") %>%
  layer_dense(2)

loss_fn <- function(y, ypred) { 
  e <- y[ , 1] - y[ , 2] * ypred[ , 1]
  return(as_tensor(sum(e*e)))
}

model %>% compile(
  optimizer = optimizer_adam(learning_rate = .001),
  loss = loss_fn
)

model %>% fit(x_train, yy_train, batch_size = 1e4, epochs = 250)
model %>% evaluate(x_train,  yy_train, batch_size = iN, verbose = 2)
y_hat <- as_tensor(predict(model, x_train), shape = c(iN, 2))
print(loss_fn(yy_train, y_hat))

y_hat <- y_hat[ , 1];
aid <- c(as.array(y_hat), as.array(y_train))
my_ymin <- min(aid); my_ymax <- max(aid)
plot(x_train, y_train, type = "l", col = "black", ylim = c(my_ymin, my_ymax))
lines(x_train, y_hat, type = "l", col = "red")
