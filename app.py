import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="퇴사 여부 예측 시스템", layout="centered")


@st.cache_data
def load_data():
    df = pd.read_csv("dataset/HR_comma_sep.csv")

    # 열 이름 공백 제거
    df.rename(columns={"Departments ": "Departments"}, inplace=True)

    # 원-핫 인코딩
    df = pd.get_dummies(df, columns=["Departments", "salary"], drop_first=True)

    return df


def train_model(df):
    # 과제 지시문 기준 선택 특성
    feature_cols = [
        "satisfaction_level",
        "number_project",
        "time_spend_company"
    ]
    target_col = "left"

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)

    feature_importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    return model, scaler, acc, report, feature_importance_df


# 데이터 로드
df = load_data()

# 모델 학습
model, scaler, acc, report, feature_importance_df = train_model(df)

# ---------------- UI ----------------
st.title("퇴사 여부 예측 시스템")
st.write("직원 정보를 입력하면 퇴사 가능성을 예측합니다.")

st.subheader("직원 정보 입력")

satisfaction_level = st.slider(
    "직원 만족도 (satisfaction_level)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

number_project = st.number_input(
    "참여한 프로젝트 수 (number_project)",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

time_spend_company = st.number_input(
    "근무 연수 (time_spend_company)",
    min_value=1,
    max_value=20,
    value=3,
    step=1
)

if st.button("예측하기"):
    input_data = np.array([
        [satisfaction_level, number_project, time_spend_company]
    ])

    input_data_scaled = scaler.transform(input_data)

    prediction = model.predict(input_data_scaled)[0]
    prediction_proba = model.predict_proba(input_data_scaled)[0]

    stay_prob = prediction_proba[0]
    leave_prob = prediction_proba[1]

    st.subheader("예측 결과")

    if prediction == 1:
        st.error(f"예측 결과: 퇴사")
    else:
        st.success(f"예측 결과: 잔류")

    st.write(f"잔류 확률: {stay_prob:.2%}")
    st.write(f"퇴사 확률: {leave_prob:.2%}")

st.markdown("---")

st.subheader("모델 성능")
st.write(f"정확도(Accuracy): {acc:.4f}")

st.subheader("특성 중요도")
st.bar_chart(feature_importance_df.set_index("feature"))

with st.expander("분류 보고서 보기"):
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df)