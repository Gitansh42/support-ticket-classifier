import os
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = "support_ticket.csv"
RESULTS_DIR = "results"

# Create results directory if it does not exist
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATA
# ============================================================

print("=" * 70)
print("AUTOMATED SUPPORT TICKET CLASSIFICATION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")

print("\nDataset shape:")
print(df.shape)

print("\nFirst 5 records:")
print(df.head())


# ============================================================
# 3. DATA VALIDATION
# ============================================================

required_columns = [
    "subject",
    "body",
    "category"
]

for column in required_columns:

    if column not in df.columns:
        raise ValueError(
            f"Required column '{column}' is missing."
        )


print("\nCategory distribution:")
print(df["category"].value_counts())


print("\nMissing values:")
print(df[required_columns].isnull().sum())


print(
    "\nDuplicate rows:",
    df.duplicated().sum()
)


# ============================================================
# 4. DATA CLEANING
# ============================================================

# Remove duplicate records
df = df.drop_duplicates()


# Remove records with missing required values
df = df.dropna(
    subset=required_columns
)


# Keep only the four required categories
allowed_categories = {
    "Billing",
    "Technical",
    "HR",
    "General"
}

df = df[
    df["category"].isin(
        allowed_categories
    )
]


# ============================================================
# 5. TEXT CLEANING FUNCTION
# ============================================================

def clean_text(text):

    # Convert to string
    text = str(text)

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove punctuation and numbers
    text = re.sub(
        r"[^a-zA-Z\s]",
        " ",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# 6. COMBINE SUBJECT + BODY
# ============================================================

df["text"] = (
    df["subject"].fillna("")
    + " "
    + df["body"].fillna("")
)


# Apply cleaning
df["clean_text"] = df["text"].apply(
    clean_text
)


print("\nExample cleaned text:")

print(
    df[
        ["text", "clean_text"]
    ].head(3)
)


# ============================================================
# 7. FEATURES AND TARGET
# ============================================================

X = df["clean_text"]

y = df["category"]


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:")
print(len(X_train))

print("\nTesting samples:")
print(len(X_test))


# ============================================================
# 9. BUILD MACHINE LEARNING PIPELINE
# ============================================================

model = Pipeline([

    # TF-IDF converts text into numerical features
    (
        "tfidf",

        TfidfVectorizer(

            # Ignore common English words
            stop_words="english",

            # Use single words + two-word combinations
            ngram_range=(1, 2),

            # Give more importance to terms that are useful
            sublinear_tf=True
        )
    ),

    # Logistic Regression classifier
    (
        "classifier",

        LogisticRegression(

            max_iter=2000,

            # Helps if categories become imbalanced
            class_weight="balanced",

            random_state=42
        )
    )
])


# ============================================================
# 10. TRAIN MODEL
# ============================================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Model training completed.")


# ============================================================
# 11. MAKE TEST PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test
)


# ============================================================
# 12. ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n")
print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)

print(
    f"\nAccuracy: {accuracy:.2%}"
)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,

    y_pred,

    zero_division=0
)


print("\nClassification Report:")
print(report)


# Save classification report
with open(
    os.path.join(
        RESULTS_DIR,
        "classification_report.txt"
    ),
    "w"
) as file:

    file.write(
        f"Accuracy: {accuracy:.2%}\n\n"
    )

    file.write(
        report
    )


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

labels = [
    "Billing",
    "Technical",
    "HR",
    "General"
]


cm = confusion_matrix(

    y_test,

    y_pred,

    labels=labels
)


print("\nConfusion Matrix:")
print(cm)


# Create visualization
disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=labels
)


disp.plot()


plt.title(
    "Support Ticket Classification\nConfusion Matrix"
)

plt.tight_layout()


# Save image
confusion_matrix_path = os.path.join(
    RESULTS_DIR,
    "confusion_matrix.png"
)


plt.savefig(
    confusion_matrix_path,
    dpi=300
)


plt.show()


# ============================================================
# 15. TICKET PREDICTION FUNCTION
# ============================================================

def predict_ticket(
    subject,
    body,
    threshold=0.60
):

    # Handle missing values
    subject = (
        ""
        if pd.isna(subject)
        else str(subject)
    )

    body = (
        ""
        if pd.isna(body)
        else str(body)
    )


    # Combine subject + body
    text = (
        subject
        + " "
        + body
    )


    # Clean text
    text = clean_text(text)


    # Handle empty ticket
    if not text:

        return {
            "predicted_category": "Unknown",
            "final_category": "Manual Review",
            "confidence": 0.0,
            "probabilities": {}
        }


    # Get probability for every category
    probabilities = model.predict_proba(
        [text]
    )[0]


    # Find highest probability
    best_index = probabilities.argmax()


    predicted_category = (
        model.classes_[best_index]
    )


    confidence = (
        probabilities[best_index]
    )


    # Confidence-based routing
    if confidence >= threshold:

        final_category = predicted_category

    else:

        final_category = "Manual Review"


    return {

        "predicted_category":
            predicted_category,

        "final_category":
            final_category,

        "confidence":
            confidence,

        "probabilities":
            dict(
                zip(
                    model.classes_,
                    probabilities
                )
            )
    }


# ============================================================
# 16. FIVE NEW UNSEEN SAMPLE TICKETS
# ============================================================

new_tickets = [

    {
        "subject":
            "Charged twice for one order",

        "body":
            (
                "I placed one order yesterday "
                "but my bank account shows two "
                "identical charges."
            )
    },


    {
        "subject":
            "VPN connection problem",

        "body":
            (
                "I cannot connect to the company "
                "VPN from my laptop."
            )
    },


    {
        "subject":
            "Salary missing",

        "body":
            (
                "My salary for this month has not "
                "been deposited into my bank account."
            )
    },


    {
        "subject":
            "Office working hours",

        "body":
            (
                "Could you please tell me what time "
                "the office opens and closes?"
            )
    },


    {
        "subject":
            "Employee portal login issue",

        "body":
            (
                "I am using the correct password but "
                "the employee portal keeps rejecting "
                "my login."
            )
    }

]


# ============================================================
# 17. PREDICT FIVE NEW TICKETS
# ============================================================

print("\n")
print("=" * 70)
print("NEW UNSEEN TICKET PREDICTIONS")
print("=" * 70)


prediction_results = []


for number, ticket in enumerate(
    new_tickets,
    start=1
):

    result = predict_ticket(

        ticket["subject"],

        ticket["body"]
    )


    print(
        f"\nTicket {number}"
    )

    print(
        "Subject:",
        ticket["subject"]
    )

    print(
        "Body:",
        ticket["body"]
    )

    print(
        "Predicted Category:",
        result[
            "predicted_category"
        ]
    )

    print(
        "Final Routing:",
        result[
            "final_category"
        ]
    )

    print(
        "Confidence:",
        f"{result['confidence']:.2%}"
    )


    print(
        "\nCategory probabilities:"
    )


    for category, probability in sorted(

        result[
            "probabilities"
        ].items(),

        key=lambda x: x[1],

        reverse=True
    ):

        print(
            f"  {category}: "
            f"{probability:.2%}"
        )


    prediction_results.append({

        "Ticket":
            number,

        "Subject":
            ticket["subject"],

        "Prediction":
            result[
                "predicted_category"
            ],

        "Final Routing":
            result[
                "final_category"
            ],

        "Confidence":
            f"{result['confidence']:.2%}"
    })


# ============================================================
# 18. SAVE FIVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame(
    prediction_results
)


prediction_df.to_csv(

    os.path.join(
        RESULTS_DIR,
        "new_ticket_predictions.csv"
    ),

    index=False
)


# ============================================================
# 19. LIVE TICKET TEST
# ============================================================

print("\n")
print("=" * 70)
print("LIVE TICKET CLASSIFICATION")
print("=" * 70)


subject = input(
    "\nEnter ticket subject: "
)


body = input(
    "Enter ticket body: "
)


result = predict_ticket(

    subject,

    body
)


print("\nResult")

print(
    "Predicted Category:",
    result[
        "predicted_category"
    ]
)

print(
    "Final Routing:",
    result[
        "final_category"
    ]
)

print(
    "Confidence:",
    f"{result['confidence']:.2%}"
)


# ============================================================
# 20. END
# ============================================================

print("\n")
print("=" * 70)
print("CLASSIFICATION COMPLETE")
print("=" * 70)

print(
    "\nResults saved in:",
    RESULTS_DIR
)