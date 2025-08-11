# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import xgboost as xgb
import streamlit as st

# %%
df = pd.read_csv(r"C:\Users\rosel\Downloads\New folder\Classifying Network Traffic\DDos.pcap_ISCX.csv", low_memory=False)

# %%
df.head()

# %%
df.columns = df.columns.str.strip()

# %%
df['Label'].value_counts()

# %%
def is_running_in_streamlit():
    try:
        import streamlit.runtime.scriptrunner.script_run_context as st_context
        return st_context.get_script_run_ctx() is not None
    except:
        return False

in_streamlit = is_running_in_streamlit()

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: to avoid plot stacking
plt.clf()
plt.cla()
plt.close('all')

# Compute correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", linewidths=0.5, ax=ax)
ax.set_title("Feature Correlation Heatmap")

# Detect Streamlit runtime
def is_running_in_streamlit():
    try:
        import streamlit.runtime.scriptrunner.script_run_context as st_context
        return st_context.get_script_run_ctx() is not None
    except:
        return False

in_streamlit = is_running_in_streamlit()

# Show plot in correct environment
if in_streamlit:
    import streamlit as st
    st.pyplot(fig)
else:
    plt.show()

# %%
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Encode target
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])

# %%
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]
print("Highly correlated features to drop:", to_drop)

# %%
df.drop(columns=to_drop, inplace=True)

# %%
# Split features/target
X = df.drop('Label', axis=1)
y = df['Label']

# %%
# Check for NaNs
print("NaNs in X:", X.isna().sum().sum())

# Check for Infs
print("Infs in X:", np.isinf(X).sum().sum())

# Optional: Check extreme large values
print("Max value in X:", X.max().max())

# %%
# Replace inf/-inf with NaN
X.replace([np.inf, -np.inf], np.nan, inplace=True)

# %%
# Drop rows with NaNs
X.dropna(inplace=True)

# %%
y = y[X.index]  # So labels match after dropping rows

# %%
# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %%
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# %%
from sklearn.metrics import classification_report, confusion_matrix

model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_), eval_metric='mlogloss')
model.fit(X_train, y_train)

# %%
# Predict
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# %%
df = df.loc[X.index]
y = y.loc[X.index]

# %%
# Get probability for the 'DDoS' class (assuming it's class 1)
probs = model.predict_proba(X_scaled)[:, 1]

# Add as 'anomaly_score' (higher means more likely to be DDoS)
df['anomaly_score'] = probs

# Predict labels
df['anomaly'] = model.predict(X_scaled)

# Show results
print(df[['anomaly_score', 'anomaly']].head())

# %%
import os

fig, ax = plt.subplots()
sns.scatterplot(data=df, x='Flow Duration', y='anomaly_score', hue='anomaly')
ax.set_title("Network Traffic Anomaly Detection")
if "streamlit" in os.environ.get("PYTHONPATH", "").lower():
    st.pyplot(fig)
else:
    plt.show()

# %%
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Confusion Matrix
cm = confusion_matrix(df['Label'], df['anomaly'])
print(cm)

# %%
# ROC Curve
fig, ax = plt.subplots()
fpr, tpr, _ = roc_curve(df['Label'], df['anomaly_score'])
ax.plot(fpr, tpr, label=f"AUC = {auc(fpr, tpr):.2f}")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
ax.legend()
if in_streamlit:
    st.pyplot(fig)
else:
    plt.show()

# %%
from sklearn.impute import SimpleImputer

# Fill NaNs with the mean of each column
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# %%
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# %%
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

wcss = []  # Within-Cluster Sum of Squares

for k in range(1, 11):  # Try K values from 1 to 10
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)  # Store WCSS for each k

# Plot Elbow Curve
fig, ax = plt.subplots()
ax.plot(range(1, 11), wcss, marker='o')
ax.set_xlabel('Number of Clusters (k)')
ax.set_ylabel('WCSS')
ax.set_title('Elbow Method for Optimal k')
if in_streamlit:
    st.pyplot(fig)
else:
    plt.show()

# %%
optimal_k = 4

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)  # Assign cluster labels

df['Cluster'] = clusters  # Add cluster labels to your original dataset

# %%
fig, ax = plt.subplots(figsize=(12, 8))  # Create figure and axes

# Optional: Clear previous figures
plt.close('all')

# Plot the clusters with transparency and custom size
sns.scatterplot(
    x=df['Flow Bytes/s'],
    y=df['Total Fwd Packets'],
    hue=df['Cluster'],
    palette='viridis',
    alpha=0.6,  # 🔸 transparency
    s=40        # 🔸 marker size
)

# Plot centroids
centroids = kmeans.cluster_centers_
plt.scatter(
    centroids[:, 0],
    centroids[:, 1],
    s=300,         # 🔸 bigger size
    c='red',
    marker='X',
    edgecolors='black',
    label='Centroids'
)

# Enhancements
ax.set_title('K-Means Clustering of Network Traffic with Centroids')
ax.set_xlabel('Flow Bytes/s')
ax.set_ylabel('Total Fwd Packets')

ax.grid(True, linestyle='--', alpha=0.5)  # 🧭 grid for clarity
ax.set_xscale('log') 

ax.legend()
fig.tight_layout()
if in_streamlit:
    st.pyplot(fig)
else:
    plt.show()

# %%
from sklearn.model_selection import GridSearchCV

params = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200]
}

grid = GridSearchCV(xgb.XGBClassifier(objective='multi:softmax', num_class=len(le.classes_)), param_grid=params, scoring='accuracy', cv=3)
grid.fit(X_train, y_train)

# %%
print("Best Parameters:", grid.best_params_)

# %%
from sklearn.metrics import ConfusionMatrixDisplay

# Optional: to avoid plot stacking
plt.clf()
plt.cla()
plt.close('all')

disp = ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred, 
    display_labels=le.classes_, 
    xticks_rotation=45
)
disp.plot()
if in_streamlit:
    st.pyplot(fig)
else:
    plt.show()

# %%
from xgboost import plot_importance
plot_importance(model, max_num_features=10)
if in_streamlit:
    st.pyplot(fig)
else:
    plt.show()

# %%
# Get actual feature names
feature_names = X.columns

# Create a mapping
feature_mapping = {f"f{i}": name for i, name in enumerate(feature_names)}
print(feature_mapping)

# %%
import pandas as pd

# Get importance scores
importance = model.get_booster().get_score(importance_type='weight')

# Map them to actual names
named_importance = {feature_mapping[k]: v for k, v in importance.items()}

# Convert to DataFrame
importance_df = pd.DataFrame.from_dict(named_importance, orient='index', columns=['Importance'])
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Show top 10
print(importance_df.head(10))

# %%
top_features = importance_df.head(10).index.tolist()

X_reduced = X[top_features]

# Then redo train-test split and model training
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)

# %%
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# %%
from tensorflow.keras import layers, Sequential, Input

input_dim = X.shape[1]

# Define encoder
encoder = Sequential([
    Input(shape=(input_dim,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
])

# Define decoder
decoder = Sequential([
    Input(shape=(8,)),
    layers.Dense(16, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(input_dim, activation='sigmoid'),
])

# %%
autoencoder = Sequential([encoder, decoder])
autoencoder.compile(optimizer='adam', loss='mse')

# %%
import shap

# Sample data for SHAP background and explanation
X_background = X_scaled[np.random.choice(X_scaled.shape[0], 100, replace=False)]
X_test_sample = X_scaled[:100]

explainer = shap.KernelExplainer(autoencoder.predict, X_background)
shap_values = explainer.shap_values(X_test_sample)

shap.summary_plot(shap_values, X_test_sample)

# %%
import joblib
joblib.dump(autoencoder, "Classifying_Network_Traffic.keras")

# %%
import streamlit as st
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("Classifying_Network_Traffic.keras")

model.save("Classifying_Network_Traffic.keras")

# %%
st.title("Network Traffic Anomaly Detector")

# Add input fields based on top features
destination_port = st.number_input("Destination Port")
init_win_fwd = st.number_input("Init Win Bytes Forward")
flow_duration = st.number_input("Flow Duration")
flow_iat_min = st.number_input("Flow IAT Min")
init_win_bwd = st.number_input("Init Win Bytes Backward")
flow_bytes_per_sec = st.number_input("Flow Bytes/s")

# %%
# Prediction
if st.button("Predict"):
    features = np.array([[destination_port, init_win_fwd, flow_duration, flow_iat_min,
                          init_win_bwd, flow_bytes_per_sec]])
    
    reconstructed = model.predict(features)
    mse = np.mean(np.power(features - reconstructed, 2))

    threshold = 0.01
    if mse > threshold:
        st.write("🔍 Prediction:", "❗ Anomaly Detected")
    else:
        st.write("🔍 Prediction:", "✅ Benign")

# %%
del model
import gc
gc.collect()

# %%



