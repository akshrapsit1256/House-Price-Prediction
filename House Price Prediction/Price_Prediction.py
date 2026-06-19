import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor



malbourne_data=pd.read_csv("Intro to ML/melb_data.csv")
print(malbourne_data.columns)

# Drop missing Values
malbourne_data=malbourne_data.dropna(axis=0)


y=malbourne_data.Price

malbourne_features=['Rooms','Bathroom','Landsize','Lattitude','Longtitude']
X=malbourne_data[malbourne_features]
print(X.describe())
print(X.head())

malbourne_model=DecisionTreeRegressor(random_state=1)
malbourne_model.fit(X,y)

print("Making prediction for the following 5 houses: ")
print(X.head())
print("The Predictions are: ")
print(malbourne_model.predict(X.head()))


# Mean Absolute error
print("Normal Error: ")
predicted_home_prices=malbourne_model.predict(X)
print(mean_absolute_error(y,predicted_home_prices))



train_X, val_X, train_y, val_y=train_test_split(X,y,random_state=0)

malbourne_model=DecisionTreeRegressor()
malbourne_model.fit(train_X,train_y)

val_predictions=malbourne_model.predict(val_X)

print("After comaring data with unseen data Error is : ")
print(mean_absolute_error(val_y,val_predictions))


def get_mae(max_leaf_node,train_X,train_y,val_X,val_y):
    model=DecisionTreeRegressor(max_leaf_nodes=max_leaf_node,random_state=0)
    model.fit(train_X,train_y)
    preds_val=model.predict(val_X)
    mae=mean_absolute_error(val_y,preds_val)
    return mae

for max_leaf_node in [5,50,500,5000]:
    my_mae=get_mae(max_leaf_node,train_X,train_y,val_X,val_y)
    print(f"Max leaf nodes: {max_leaf_node}\t\tMean Absolute Error: {my_mae}")

forest_model=RandomForestRegressor(random_state=1)
forest_model.fit(train_X,train_y)
melb_preds=forest_model.predict(val_X)
print(mean_absolute_error(val_y,melb_preds))    