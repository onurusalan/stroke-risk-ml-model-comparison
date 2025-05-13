# Import required libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
import warnings
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

# Create analysis directory
if not os.path.exists('analysis'):
    os.makedirs('analysis')

# Read the dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# General information about the dataset
print("Dataset size:", df.shape)
print("\nGeneral information about the dataset:")
print(df.info())
print("\nFirst 5 rows of the dataset:")
print(df.head())
print("\nStatistical summary of the dataset:")
print(df.describe())
print("\n")

# Check and visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Visualization of Missing Values')
plt.tight_layout()
plt.savefig('analysis/missing_values.png')
plt.close()

print("Checking missing values:")
print(df.isnull().sum())
print("\n")

# Fill missing values in BMI with median
df['bmi'] = df['bmi'].fillna(df['bmi'].median())

# Encode categorical variables
le = LabelEncoder()
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for column in categorical_columns:
    df[column] = le.fit_transform(df[column])

# Visualize correlation matrix
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('analysis/correlation_matrix.png')
plt.close()

# Separate features and target variable
X = df.drop(['id', 'stroke'], axis=1)
y = df['stroke']

# Split training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Balance training data with SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Define classification models and hyperparameters
models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, class_weight='balanced'),
        'params': {
            'C': [0.1, 1, 10],
            'max_iter': [1000]
        }
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'max_depth': [3, 5, 7],
            'min_samples_split': [2, 5, 10]
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7]
        }
    },
    'SVM': {
        'model': SVC(random_state=42, probability=True, class_weight='balanced'),
        'params': {
            'C': [1],
            'kernel': ['linear']
        }
    },
    'k-NN': {
        'model': KNeighborsClassifier(),
        'params': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    }
}

# Dictionaries to store results
results = {}
best_models = {}
cv_scores = {}

# Register font for English character support
pdfmetrics.registerFont(TTFont('DejaVuSans', 'analysis/DejaVuSans.ttf'))

# Training and evaluation for each model
for name, model_info in models.items():
    print(f"\n{name} Model:")
    print("-" * 50)
    
    # Hyperparameter optimization with GridSearchCV
    grid_search = GridSearchCV(
        model_info['model'],
        model_info['params'],
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    # Create pipeline (with StandardScaler)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', grid_search)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Save best model
    best_models[name] = pipeline
    
    # Cross-validation scores
    cv_score = cross_val_score(pipeline, X_train, y_train, cv=5)
    cv_scores[name] = cv_score.mean()
    
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    # Accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Cross-validation score: {cv_score.mean():.4f} (+/- {cv_score.std() * 2:.4f})")
    print(f"Test accuracy: {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.ylabel('True Values')
    plt.xlabel('Predicted Values')
    plt.tight_layout()
    plt.savefig(f'analysis/confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f'analysis/roc_curve_{name.lower().replace(" ", "_")}.png')
    plt.close()
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Visualize results
plt.figure(figsize=(12, 6))
x = np.arange(len(results))
width = 0.35

plt.bar(x - width/2, results.values(), width, label='Test Accuracy')
plt.bar(x + width/2, cv_scores.values(), width, label='Cross-validation Score')

plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.title('Model Performance Comparison')
plt.xticks(x, results.keys(), rotation=45)
plt.legend()
plt.tight_layout()
plt.savefig('analysis/model_comparison.png')
plt.close()

# Determine best model
best_model = max(results.items(), key=lambda x: x[1])
print(f"\nBest performing model: {best_model[0]} (Accuracy: {best_model[1]:.4f})")

# Visualize feature importance (for Random Forest)
if 'Random Forest' in best_models:
    rf_model = best_models['Random Forest'].named_steps['classifier'].best_estimator_
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('analysis/feature_importance.png')
    plt.close()

# Save results to a text file
with open('analysis/model_results.txt', 'w') as f:
    f.write("Model Performance Results\n")
    f.write("=" * 50 + "\n\n")
    
    for name in results.keys():
        f.write(f"{name} Model:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Test Accuracy: {results[name]:.4f}\n")
        f.write(f"Cross-validation Score: {cv_scores[name]:.4f}\n")
        f.write(f"Best parameters: {best_models[name].named_steps['classifier'].best_params_}\n\n")
    
    f.write("\nBest Model:\n")
    f.write("-" * 30 + "\n")
    f.write(f"{best_model[0]} (Accuracy: {best_model[1]:.4f})\n")

# Save results to a new text file for comparison
with open('analysis/nex_model_results.txt', 'w') as f:
    f.write("All Models Comparison Results\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"{'Model':<25}{'Test Accuracy':<18}{'CV Score':<18}{'Best Parameters'}\n")
    f.write("-" * 60 + "\n")
    for name in results.keys():
        params = best_models[name].named_steps['classifier'].best_params_
        f.write(f"{name:<25}{results[name]:<18.4f}{cv_scores[name]:<18.4f}{str(params)}\n")
    f.write("\nBest Model: ")
    f.write(f"{best_model[0]} (Test Accuracy: {best_model[1]:.4f})\n")

# Create PDF report
def create_pdf_report():
    doc = SimpleDocTemplate("analysis/mergedanalysis.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    styles['Normal'].fontName = 'DejaVuSans'
    styles['Heading2'].fontName = 'DejaVuSans'
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontName='DejaVuSans',
        fontSize=24,
        spaceAfter=30
    )
    story.append(Paragraph("Stroke Prediction Model Analysis Report", title_style))
    story.append(Spacer(1, 12))
    
    # Dataset information
    story.append(Paragraph("Dataset Information", styles['Heading2']))
    story.append(Paragraph(f"Dataset size: {df.shape}", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Missing values visualization
    story.append(Paragraph("Missing Values Analysis", styles['Heading2']))
    img = Image('analysis/missing_values.png', width=6*inch, height=4*inch)
    story.append(img)
    story.append(Spacer(1, 12))
    
    # Correlation matrix
    story.append(Paragraph("Correlation Matrix", styles['Heading2']))
    img = Image('analysis/correlation_matrix.png', width=6*inch, height=4*inch)
    story.append(img)
    story.append(Spacer(1, 12))
    
    # Model performance comparison
    story.append(Paragraph("Model Performance Comparison", styles['Heading2']))
    img = Image('analysis/model_comparison.png', width=6*inch, height=4*inch)
    story.append(img)
    story.append(Spacer(1, 12))
    
    # Results for each model
    for name in results.keys():
        story.append(Paragraph(f"{name} Model Results", styles['Heading2']))
        
        # Confusion Matrix
        img = Image(f'analysis/confusion_matrix_{name.lower().replace(" ", "_")}.png', 
                   width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 12))
        
        # ROC curve
        img = Image(f'analysis/roc_curve_{name.lower().replace(" ", "_")}.png', 
                   width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 12))
        
        # Model metrics
        metrics_text = f"""
        Test Accuracy: {results[name]:.4f}
        Cross-validation Score: {cv_scores[name]:.4f}
        Best parameters: {best_models[name].named_steps['classifier'].best_params_}
        """
        story.append(Paragraph(metrics_text, styles['Normal']))
        story.append(Spacer(1, 12))
    
    # Feature importance (for Random Forest)
    if 'Random Forest' in best_models:
        story.append(Paragraph("Feature Importance (Random Forest)", styles['Heading2']))
        img = Image('analysis/feature_importance.png', width=6*inch, height=4*inch)
        story.append(img)
        story.append(Spacer(1, 12))
    
    # Best model result
    story.append(Paragraph("Best Model", styles['Heading2']))
    best_model_text = f"""
    Best performing model: {best_model[0]}
    Accuracy: {best_model[1]:.4f}
    """
    story.append(Paragraph(best_model_text, styles['Normal']))
    
    # Build PDF
    doc.build(story)

# Create PDF report
create_pdf_report() 