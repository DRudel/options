from sklearn.datasets import load_diabetes
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor, export_text

data = load_diabetes()
X = data['data']
y = data['target']

simple_model = DecisionTreeRegressor(max_depth=4)
prototype = DecisionTreeRegressor(max_depth=4)
simple_ada = AdaBoostRegressor(prototype, n_estimators=1)
simple_gbr = GradientBoostingRegressor(max_depth=4, n_estimators=1, criterion='mse')

simple_model.fit(X, y)
simple_ada.fit(X, y)
simple_gbr.fit(X, y)

ada_one = simple_ada.estimators_[0]
gbr_one = simple_gbr.estimators_[0][0]

print(export_text(simple_model))
# print(export_text(ada_one))
print(export_text(gbr_one))
