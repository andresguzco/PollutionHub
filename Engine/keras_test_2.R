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
  # layer_dense(4, activation = "sigmoid") %>%
  # layer_dropout(0.2) %>%
  layer_dense(2)


loss_fn <- function(y, ypred) { 
  e <- y[ , 1] - y[ , 2] * ypred[ , 1]
  # e = y - xx * ypred
  # if (y$shape != ypred$shape) stop(paste0("ERROR loss_fn: shape y = (", 
  #                                         paste(y$shape, sep = "", collapse = ","),
  #                                         ") and shape ypred = (",
  #                                         paste(ypred$shape, sep = "", collapse = ","),
  #                                         ")"))
  # e = y - ypred
  return(as_tensor(sum(e*e)))
}
# print(paste0("DGP loss: ",as.double(loss_fn(yy_train, cbind(y_train, y_train)))))
# print(paste0("Lousy loss: ",as.double(loss_fn(yy_train, cbind(y_train, y_train) + 5))))


model %>% compile(
  optimizer = optimizer_adam(learning_rate = .001),
  loss = loss_fn
)


# a = model$get_weights()
# a[[1]] = matrix(c(-8, 0, 0, 0, 0, 0), nrow = 1, ncol = ncol(a[[1]]))
# for (j in 1:6) a[[2]][j] = c(-1, 0, 0, 0, 0, 0)[j]
# a[[3]] = matrix(0, nrow = nrow(a[[3]]), ncol = ncol(a[[3]]))
# for (j in 1:6) a[[3]][j,1] = 1
# a[[5]] = matrix(0, nrow = nrow(a[[5]]), ncol = ncol(a[[5]]))
# a[[5]][1,1] = 1
# for (i in c(4,6)) for (j in 1:length(a[[i]])) a[[i]][j] = 0
# model$set_weights(a)
# y_hat = as_tensor(predict(model, x_train), shape = c(iN, 2))
# y_hat = y_hat[ , 1];
# aid = c(as.array(y_hat), as.array(y_train))
# my_ymin = min(aid); my_ymax = max(aid)
# plot(x_train, y_train, type = "l", col = "black", ylim = c(my_ymin, my_ymax))
# lines(x_train, y_hat, type = "l", col = "red")



# stop("stopping here")
model %>% fit(x_train, yy_train, batch_size = 1e4, epochs = 250)
model %>% evaluate(x_train,  yy_train, batch_size = iN, verbose = 2)
y_hat <- as_tensor(predict(model, x_train), shape = c(iN, 2))
print(loss_fn(yy_train, y_hat))
# print(loss_fn(
#   as_tensor(yy, dtype = tf$float32, shape = c(iN,1)),
#   as_tensor(y_hat, dtype = tf$float32, shape = c(iN,1))
# ))

y_hat <- y_hat[ , 1]
aid <- c(as.array(y_hat), as.array(y_train))
my_ymin <- min(aid); my_ymax <- max(aid)
plot(x_train, y_train, type = "l", col = "black", ylim = c(my_ymin, my_ymax))
lines(x_train, y_hat, type = "l", col = "red")
