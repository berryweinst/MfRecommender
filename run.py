from model import *


X_train, X_test, users, items = create_user_item_map('../ml-1m/ratings.dat', '', percent=0.8, sort_ratings=True)


# 1a - SGD
# model = MFModel(X_train, X_test, users, items, hidden_dim=40, lr=0.01, gamma=0.1, reg_u=0.1, reg_i=0.1, epochs=22, optimizer='sgd')
# print("Starting training SGD...")
# model.print_hyper_params()
# model.train_model()
#
#
#
# train_rmse = [t for e, t, v in model.scbd]
# test_rmse = [v for e, t, v in model.scbd]
# epochs = [e for e, t, v in model.scbd]
# plt.figure(figsize=((16,4)))
# plt.plot(epochs, train_rmse, label="Train")
# plt.plot(epochs, test_rmse, label="Test")
# plt.legend()
# plt.xticks(epochs, fontsize=8)
# plt.xlabel("Epochs")
# plt.ylabel("RMSE")
# plt.grid(axis="y")


# 1b - ALS
model = MFModel(X_train, X_test, users, items, hidden_dim=40, lr=0.001, gamma=0.1, reg_u=0.1, reg_i=0.1, epochs=20, optimizer='als')
print("Starting training ALS...")
model.print_hyper_params()
model.train_model()


train_rmse = [t for e, t, v in model.scbd]
test_rmse = [v for e, t, v in model.scbd]
epochs = [e for e, t, v in model.scbd]
plt.figure(figsize=((16,4)))
plt.plot(epochs, train_rmse)
plt.plot(epochs, test_rmse)
plt.xticks(epochs, fontsize=8)
plt.xlabel("Epochs")
plt.ylabel("RMSE")
plt.grid(axis="y")