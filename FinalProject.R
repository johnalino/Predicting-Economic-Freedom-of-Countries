FLAGS <- flags(
  flag_numeric("nodes1", 128),
  flag_numeric("nodes2", 128),
  flag_numeric("batch_size", 100),
  flag_string("activation1", "relu"),
  flag_string("activation2", "relu"),
  flag_numeric("learning_rate", 0.01),
  flag_numeric("epochs", 30),
  flag_numeric("dropout1", 0.05),
  flag_numeric("dropout2", 0.05)
)

model = keras_model_sequential()
model %>%
  layer_dense(units=FLAGS$nodes1, activation=FLAGS$activation1, input_shape=dim(efw_train2)[2]) %>%
  layer_dropout(FLAGS$dropout1) %>%
  layer_dense(units=1)

model %>% compile(
  optimizer = optimizer_adam(lr=FLAGS$learning_rate),
  loss = 'mse',
  metrics = c('mse')
)

model %>% fit(
  as.matrix(efw_train2), train2Label,
  epochs=FLAGS$epochs,
  batch_size=FLAGS$batch_size,
  validation_data=list(as.matrix(efw_val), valLabel)
)