import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

# Save iris model
iris = load_iris()
nb = GaussianNB()
nb.fit(iris.data, iris.target)
joblib.dump(nb, 'naive_bayes_model.pkl')
print("naive_bayes_model.pkl created!")

# Save diabetes model (empty — app.py refits it on load)
nb2 = GaussianNB()
joblib.dump(nb2, 'diabetes_model.pkl')
print("diabetes_model.pkl created!")

print("Done! Both files saved.")