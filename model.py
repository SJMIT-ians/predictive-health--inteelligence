from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class PatientFeatures:
    age: float
    sex: int  # 0 = female, 1 = male
    bmi: float
    systolic_bp: float
    diastolic_bp: float
    cholesterol: float
    smoker: int  # 0 = no, 1 = yes
    family_history: int  # 0 = no, 1 = yes
    exercise_level: int  # 0 = low, 1 = medium, 2 = high

    def to_vector(self) -> List[float]:
        return [
            self.age,
            self.sex,
            self.bmi,
            self.systolic_bp,
            self.diastolic_bp,
            self.cholesterol,
            self.smoker,
            self.family_history,
            self.exercise_level,
        ]


def _generate_synthetic_data(n_samples: int = 2000, random_state: int = 42):
    rng = np.random.default_rng(random_state)

    age = rng.integers(18, 90, n_samples)
    sex = rng.integers(0, 2, n_samples)
    bmi = rng.normal(27, 5, n_samples).clip(16, 45)
    systolic = rng.normal(125, 18, n_samples).clip(90, 210)
    diastolic = rng.normal(80, 12, n_samples).clip(50, 130)
    chol = rng.normal(190, 35, n_samples).clip(120, 320)
    smoker = rng.integers(0, 2, n_samples)
    family = rng.integers(0, 2, n_samples)
    exercise = rng.integers(0, 3, n_samples)

    # Create a synthetic risk score: higher for classic cardiovascular risk factors
    risk_score = (
        0.03 * (age - 40)
        + 0.06 * (bmi - 25)
        + 0.02 * (systolic - 120)
        + 0.015 * (chol - 180) / 10
        + 0.5 * smoker
        + 0.4 * family
        - 0.25 * exercise
        + 0.2 * sex  # slightly higher risk for males on average
    )

    # Convert continuous risk into a binary label
    prob = 1 / (1 + np.exp(-risk_score / 4))
    y = rng.binomial(1, prob)

    X = np.column_stack(
        [
            age,
            sex,
            bmi,
            systolic,
            diastolic,
            chol,
            smoker,
            family,
            exercise,
        ]
    )

    return X, y


def _train_model() -> Pipeline:
    X, y = _generate_synthetic_data()
    clf = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=200)),
        ]
    )
    clf.fit(X, y)
    return clf


_MODEL: Pipeline | None = None


def get_model() -> Pipeline:
    global _MODEL
    if _MODEL is None:
        _MODEL = _train_model()
    return _MODEL


def predict_risk(features: PatientFeatures) -> dict:
    model = get_model()
    X = np.array([features.to_vector()])
    proba = model.predict_proba(X)[0, 1]

    if proba < 0.25:
        category = "Low"
        recommendation = "Maintain your current lifestyle, keep regular check-ups, and continue healthy habits."
    elif proba < 0.55:
        category = "Moderate"
        recommendation = (
            "Consider improving diet and exercise habits, monitor blood pressure, and consult a clinician for "
            "personalized advice."
        )
    else:
        category = "High"
        recommendation = (
            "Schedule a medical evaluation soon. You may benefit from detailed cardiovascular assessment and "
            "targeted risk reduction."
        )

    return {
        "risk_probability": float(proba),
        "risk_category": category,
        "recommendation": recommendation,
    }

