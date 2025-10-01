import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tkinter as tk
from tkinter import messagebox, ttk

-------------------------------

1. Load & Preprocess Data

-------------------------------

data = pd.read_csv("train.csv")

if "index" in data.columns:
data = data.drop("index", axis=1)

data = data.ffill()  # handle missing values

label_encoders = {}
for col in data.select_dtypes(include=["object"]).columns:
le = LabelEncoder()
data[col] = le.fit_transform(data[col])
label_encoders[col] = le

X = data.drop("Depression", axis=1)
y = data["Depression"]

-------------------------------

2. Train-Test Split & Model

-------------------------------

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

-------------------------------

3. Tkinter GUI

-------------------------------

def predict_depression():
try:
values = []
for col, widget in zip(X.columns, entries):
val = widget.get().strip()

# If the column was label encoded  
        if col in label_encoders:  
            le = label_encoders[col]  
            if val in le.classes_:  
                val = le.transform([val])[0]  
            else:  
                messagebox.showerror("Error", f"Invalid input '{val}' for {col}. Allowed: {list(le.classes_)}")  
                return  
        else:  
            val = float(val)  # numeric column  

        values.append(val)  

    person = pd.DataFrame([values], columns=X.columns)  
    pred = model.predict(person)[0]  

    if pred == 1:  
        messagebox.showinfo("Result", "🔴 Person is likely Depressed")  
    else:  
        messagebox.showinfo("Result", "🟢 Person is Not Depressed")  
except Exception as e:  
    messagebox.showerror("Error", str(e))

Build GUI window

root = tk.Tk()
root.title("Depression Predictor")
root.geometry("400x600")

entries = []
for col in X.columns:
tk.Label(root, text=col).pack()

if col in label_encoders:  
    # Dropdown for categorical values  
    values = list(label_encoders[col].classes_)  
    combo = ttk.Combobox(root, values=values, state="readonly")  
    combo.pack()  
    entries.append(combo)  
else:  
    # Numeric input  
    entry = tk.Entry(root)  
    entry.pack()  
    entries.append(entry)

tk.Button(root, text="Predict", command=predict_depression, bg="blue", fg="white").pack(pady=20)

root.mainloop() 

