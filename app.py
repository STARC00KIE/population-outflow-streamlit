# app.py — Streamlit 개요 탭 뼈대 (청년 유출 프로젝트)
# -----------------------------------------------------
# 이 파일은 '개요' 탭을 중심으로 한 포트폴리오형 대시보드 입니다.
# - 사이드바: 데이터 업로드(선택), 연도/지역 필터(샘플)
# - 탭: [개요] [데이터/EDA] [모델/중요도] [지역별 리포트] [정책 시나리오] [문서/링크]
# 실제 데이터 컬럼명에 맞춰 TODO 부분만 채우면 바로 확장 가능합니다.

import io
import textwrap
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import json
import joblib

from matplotlib import rcParams
import matplotlib.pyplot as plt

import streamlit as st
from streamlit_folium import st_folium

import folium
from folium.plugins import TimeSliderChoropleth

from typing import List, Tuple, Optional, Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import shap

# 한글 폰트 지정 (Windows: 맑은 고딕)
rcParams['font.family'] = 'Malgun Gothic'
# 마이너스 깨짐 방지
rcParams['axes.unicode_minus'] = False

# ---------------------------------
# 0) 페이지 설정
# ---------------------------------
st.set_page_config(
    page_title="청년 유출 포트폴리오 대시보드",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------
# 1) 전역 메타/설정 (프로젝트 고정값)
# ---------------------------------
PROJECT_META = {
    "title": "지역사회 활력 회복을 위한 청년 인구 유출 요인 분석",
    "subtitle": "2020–2023년 시군구 단위 분석과 정책 시나리오",
    "period": "2025.07–2025.09",
    "unit": "시군구 229개",
    "target": "청년 순이동률(순유출/순유입)",
    "models": ["ElasticNet", "XGBoost"],
    "core_vars": ["CMR(조혼인율)", "INDUSTRIAL_AREA(공업지역 비중)", "SELF_FINANCE(재정자립도)"],
    "team": [
        {"name": "김종현", "role": "총괄/일정 계획 관리"},
        {"name": "조재홍", "role": "자료 리서치, 시각화"},
        {"name": "강호현", "role": "QGIS/Tableau 시각화, 회귀식 설계"},
        {"name": "오수성", "role": "데이터 모델링, Streamlit 대시보드 구축"},
    ],
    "duration": "2025-07 ~ 2025-09",
}

# ---------------------------------
# 2) 유틸
# ---------------------------------

### 샘플 데이터 & 로더
@st.cache_data
def make_demo(n_regions: int = 40, years: List[int] = [2020, 2021, 2022, 2023]) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for y in years:
        for i in range(n_regions):
            sgg = f"가상시군구{i+1:03d}"
            cmr = rng.normal(5.0, 1.2)
            ind = abs(rng.normal(20, 6))  # 산업지역 비중 가상값
            sf = rng.uniform(20, 80)      # 재정자립도 가상값
            youth_net = rng.normal(0.0, 3.5) - 0.12 * (ind/10) + 0.25 * (cmr-5) + 0.05 * (sf-50)
            rows.append({
                "YEAR": y,
                "SGG_NAME": sgg,
                "CMR": cmr,
                "INDUSTRIAL_AREA": ind,
                "SELF_FINANCE": sf,
                "YOUTH_NET_RATE": youth_net,
            })
    return pd.DataFrame(rows)

@st.cache_data
def load_data(file: io.BytesIO | None) -> pd.DataFrame:
    if file is None:
        try:
            # streamlit/data/streamlit_data.csv 사용
            df = pd.read_csv("./streamlit/data/streamlit_data.csv")
        except FileNotFoundError:
            df = make_demo()
            st.caption("데모 데이터를 사용 중입니다. (streamlit_data.csv 없음)")
    else:
        try:    
            df = pd.read_csv(file)
        except UnicodeDecodeError:
            df = pd.read_csv(file, encoding="cp949")
    return df


### geojson 유틸
@st.cache_data
def load_geojson(path: str) -> dict:
    """GeoJSON 파일을 dict 형태로 로드"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def draw_choropleth(df, geojson_obj, year: int, value_col: str,
                    key_col: str = "BJCD", center=[36.5, 127.8], zoom=7):
    """
    단일 연도별 Choropleth 지도 생성 후 Streamlit에 표시

    Parameters:
        df (pd.DataFrame): 연도별 데이터 (SGG_CODE, YEAR, 값 포함)
        geojson_obj (dict): 로드된 GeoJSON 객체
        year (int): 선택된 연도
        value_col (str): 지도에 표시할 변수명
        key_col (str): GeoJSON과 매핑할 키
    """
    df_year = df[df["YEAR"] == year]

    m = folium.Map(location=center, zoom_start=zoom, tiles="cartodbpositron")

    folium.Choropleth(
        geo_data=geojson_obj,
        data=df_year,
        columns=[key_col, value_col],
        key_on=f"feature.properties.{key_col}",
        fill_color="YlOrRd",
        fill_opacity=0.8,
        line_opacity=0.2,
        nan_fill_color="white",
        legend_name=value_col,
    ).add_to(m)

    st_folium(m, width=900, height=650)


### modeling 유틸
def extract_final_model(obj):
    """GridSearchCV면 best_estimator_ 추출, 아니면 원본 반환."""
    if hasattr(obj, "best_estimator_") and obj.best_estimator_ is not None:
        return obj.best_estimator_
    return obj

def load_models_from_paths(paths):
    """로컬 경로 리스트에서 모델 읽기."""
    loaded, errors = [], []
    for p in paths:
        try:
            obj = joblib.load(p)
            model = extract_final_model(obj)
            display_name = p.split("/")[-1]
            feat_from_model = getattr(model, "feature_names_", None)
            loaded.append((display_name, model, feat_from_model))
        except Exception as e:
            errors.append(f"{p}: {e}")
    return loaded, errors

def align_features(X_all: pd.DataFrame, feat_from_model):
    """
    모델에 feature_names_가 있으면 교집합 정렬.
    return: (X_use, feat_names, missing)
    """
    if feat_from_model:
        missing = [c for c in feat_from_model if c not in X_all.columns]
        used_cols = [c for c in feat_from_model if c in X_all.columns]
        if not used_cols:
            return pd.DataFrame(index=X_all.index), [], missing
        return X_all[used_cols], used_cols, missing
    return X_all, list(X_all.columns), []

def evaluate_model(model, X: pd.DataFrame, y: pd.Series):
    """RMSE/MAE/R² 계산."""
    y_pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }

def get_importance_series(model, feat_names):
    """트리/선형/파이프라인에서 변수중요도 추출."""
    try:
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)
        if hasattr(model, "coef_"):
            coef = np.ravel(model.coef_)
            return pd.Series(np.abs(coef), index=feat_names).sort_values(ascending=False)
        if hasattr(model, "steps"):
            last_est = model.steps[-1][1]
            if hasattr(last_est, "feature_importances_"):
                return pd.Series(last_est.feature_importances_, index=feat_names).sort_values(ascending=False)
            if hasattr(last_est, "coef_"):
                coef = np.ravel(last_est.coef_)
                return pd.Series(np.abs(coef), index=feat_names).sort_values(ascending=False)
    except Exception:
        return None
    return None

def choose_shap_explainer(model, X_use: pd.DataFrame):
    """
    모델 유형에 맞춰 적절한 SHAP Explainer를 생성해 반환.
    return: (explainer, method_str, base_model_for_pred)
    """

    m = model
    m_str = str(type(m)).lower()

    # 파이프라인이면 마지막 추정기
    if hasattr(model, "steps"):
        m = model.steps[-1][1]
        m_str = str(type(m)).lower()

    if any(k in m_str for k in ["xgboost", "lightgbm", "randomforest", "gradientboosting"]):
        explainer = shap.TreeExplainer(m)
        return explainer, "tree", m
    elif any(k in m_str for k in ["linear", "elasticnet", "lasso", "ridge", "sgdregressor"]):
        explainer = shap.LinearExplainer(m, X_use, feature_perturbation="interventional")
        return explainer, "linear", m
    else:
        background = shap.sample(X_use, min(200, len(X_use)), random_state=42)
        explainer = shap.KernelExplainer(m.predict, background)
        return explainer, "kernel", m

def mean_abs_shap_top(sv: np.ndarray, X_sample: pd.DataFrame, top_k: int = 15):
    """평균 |SHAP| 상위표."""
    mean_abs = np.abs(sv).mean(axis=0)
    return pd.Series(mean_abs, index=X_sample.columns).sort_values(ascending=False).head(top_k).rename("mean|SHAP|").to_frame()

### 지역별 리포트 유틸
def find_col_ci(df: pd.DataFrame, *candidates: str) -> str | None:
    """대소문자 무시로 df 컬럼명 매칭하여 첫 일치 컬럼 반환."""
    lc = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand is None:
            continue
        if cand.lower() in lc:
            return lc[cand.lower()]
    return None

def pct_rank(series: pd.Series, value: float) -> float:
    """분위(%): series에서 value가 아래에 있는 비율(0~100)."""
    s = series.dropna()
    if pd.notnull(value) and len(s):
        return float((s < value).mean() * 100.0)
    return np.nan

def ensure_bjcd_string(df: pd.DataFrame, col: str = "BJCD") -> pd.DataFrame:
    """BJCD → 문자열 보장."""
    df = df.copy()
    df[col] = df[col].astype(str)
    return df

def make_val_map(df: pd.DataFrame, code_col: str, value_cols, name_col: str):
    """
    BJCD -> {여러 값들..., name} 딕셔너리 생성.
    value_cols: str 또는 [str, ...]
    """
    if isinstance(value_cols, str):
        cols = [value_cols]
    else:
        cols = list(value_cols)

    # 실제 존재하는 컬럼만 사용
    cols = [c for c in cols if c in df.columns]
    use_cols = [code_col, name_col] + cols
    if not cols:
        # 값 컬럼이 하나도 없으면 이름만 담아둔다
        use_cols = [code_col, name_col]

    return (
        df[use_cols]
        .dropna(subset=[code_col])
        .set_index(code_col)[[c for c in use_cols if c != code_col]]
        .to_dict(orient="index")
    )

def extract_bjcd_from_popup(popup_text: str) -> str | None:
    """folium popup 텍스트에서 'CODE: 12345' 패턴으로 BJCD 추출."""
    import re
    if not popup_text:
        return None
    m = re.search(r"(?:코드|CODE)\s*[:：]\s*(\d+)", str(popup_text))
    return m.group(1) if m else None

def feature_bounds(feat):
    """GeoJSON feature의 경계 [[south, west], [north, east]] 반환."""
    try:
        coords = []
        geom = feat.get("geometry", {})
        t = geom.get("type", "")
        if t == "Polygon":
            for ring in geom["coordinates"]:
                coords.extend(ring)   # [lon, lat]
        elif t == "MultiPolygon":
            for poly in geom["coordinates"]:
                for ring in poly:
                    coords.extend(ring)
        if not coords:
            return None
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        return [[min(lats), min(lons)], [max(lats), max(lons)]]
    except Exception:
        return None

def popup_html_for_code(code_, val_map: dict, popup_cols: list, label_map: dict, name_key: str):
    """
    선택 코드의 팝업 HTML 생성.
    - code_: BJCD 문자열
    - val_map: {BJCD: {<col>: value, name_key: 지역명, ...}}
    - popup_cols: 팝업에 보여줄 컬럼 리스트
    - label_map: 화면에 보여줄 라벨 매핑 {col_name: "라벨"}
    - name_key: 지역명 키(예: 'SGG_NAME')
    """
    rec = val_map.get(str(code_))
    if not rec:
        return f"CODE: {code_}<br>(데이터 없음)"

    lines = [f"<b>{rec.get(name_key, '')}</b>", f"CODE: {code_}"]
    for col in popup_cols:
        v = rec.get(col)
        label = label_map.get(col, col)
        if isinstance(v, (int, float, np.number)) and pd.notnull(v):
            lines.append(f"{label}: {v:,.3f}")
        elif v is not None and str(v) != "nan":
            lines.append(f"{label}: {v}")
    return "<br>".join(lines)

# ---------------------------------
# 3-1) 개요 탭 렌더러
# ---------------------------------
def render_overview_tab(df: pd.DataFrame, meta: Dict):
    # 헤더 영역
    c1, c2 = st.columns([0.75, 0.25])
    with c1:
        st.markdown(f"# {meta['title']}")
        st.markdown(f"**{meta['subtitle']}**")
        st.markdown("""
            **목표**  
            - 지역별 청년 순이동률(YOUTH_NET_MOVE_RATE)을 데이터로 설명하고, 정책 레버(CMR·SELF_FINANCE·INDUSTRIAL_AREA 등)를 조정했을 때
            예측이 어떻게 바뀌는지 **What-if 시나리오**로 탐색합니다.

            **왜 필요한가?**  
            - 청년 유출/유입의 **핵심 요인**을 파악해 근거 기반 의사결정을 돕습니다.  
            - **어느 지역**이 기회/위험 구간인지, **어떤 정책 레버**가 효과적인지 비교합니다.  
            - 시나리오를 통해 **정책 우선순위**와 **예상 효과 규모**를 가늠합니다.
            """)
    with c2:
        st.markdown("### 프로젝트 정보")
        st.markdown(f"- 기간: **{meta['period']}**")
        st.markdown(f"- 단위: **{meta['unit']}**")
        st.markdown(f"- 독립변수: **{meta['target']}**")

    st.divider()

    # 핵심 질문
    with st.container():
        st.subheader("핵심 질문 (Research Questions)")
        st.markdown("- 어떤 요인들이 청년 유출/유입을 설명하는가?")
        st.markdown("- 요인들의 지역, 유형별 차이는 무엇인가?")
        st.markdown("- 정책적 개선 방안은 무엇인가?")

    st.divider()

    # 데이터 스냅샷 & 요약 메트릭
    with st.container():
        st.subheader("데이터 개요")
        total_rows = len(df)
        n_years = df["YEAR"].nunique() if "YEAR" in df.columns else None
        n_regions = df["SGG_NAME"].nunique() if "SGG_NAME" in df.columns else None

        m1, m2, m3 = st.columns(3)
        m1.metric("총 행 수", f"{total_rows:,}")
        m2.metric("연도 수", n_years if n_years is not None else "—")
        m3.metric("지역 수", n_regions if n_regions is not None else "—")

        st.dataframe(df.head(10), use_container_width=True)

    st.divider()

    # 방법론 요약 (파이프라인)
    with st.container():
        st.subheader("방법론 요약")
        st.markdown(
            "➡️ **정제 → 전처리 → EDA → 변수선택 → 모델링 → 정책 시나리오**"
        )
        # 간단한 Graphviz 다이어그램 (선택)
        try:
            st.graphviz_chart(
                """
                digraph G {
                  rankdir=LR;
                  A[label="정제"]; B[label="전처리"]; C[label="EDA"]; D[label="변수선택"]; E[label="모델링"]; F[label="정책 시나리오"];
                  A -> B -> C -> D -> E -> F;
                }
                """
            )
        except Exception:
            st.info("Graphviz 렌더가 불가한 환경입니다. 텍스트 파이프라인을 확인하세요.")

        st.markdown("**사용 모델**: " + ", ".join(meta["models"]))
        st.caption("ElasticNet(해석력), XGBoost(비선형·상호작용·예측력)")

    st.divider()

    # 핵심 결과 요약 (하이라이트 카드)
    with st.container():
        st.subheader("핵심 결과 요약")
        st.markdown("- 상위 핵심 변수: **" + ", ".join(meta["core_vars"]) + "**")
        c1, c2, c3 = st.columns(3)
        c1.info("조혼인율 증가 ↗️ → 순이동률 개선 경향")
        c2.info("공업지역 면적 증대 ↗️ → 순이동률에 복합적 영향")
        c3.info("재정자립도 향상 ↗️ → 정책 추진 기반 강화")

    st.divider()

    # 팀/기간
    with st.container():
        st.subheader("팀 역할 & 기간")
        t1, t2 = st.columns([0.6, 0.4])
        with t1:
            for m in meta["team"]:
                st.markdown(f"- **{m['name']}** — {m['role']}")
        with t2:
            st.markdown(f"**진행 기간**: {meta['duration']}")

    st.divider()


# ---------------------------------
# 3-2)EDA 탭 렌더러
# ---------------------------------
def render_eda_tab(df: pd.DataFrame):
    st.header("탐색적 데이터 분석(EDA)")

    # 0) 선택 필터 (연도)
    filtered = df.copy()

    # 1) 데이터 구조
    st.subheader("데이터 구조 요약")
    n_rows, n_cols = filtered.shape
    m1, m2 = st.columns(2)
    m1.metric("행(Row)", f"{n_rows:,}")
    m2.metric("열(Col)", f"{n_cols:,}")

    st.divider()
    
    # 2) 기술통계
    st.subheader("기술통계")
    num_cols = filtered.select_dtypes(include=["number"]).columns.tolist()
    if num_cols:
        st.dataframe(filtered[num_cols].describe().T)
    else:
        st.info("수치형 컬럼이 없습니다.")

    st.divider()

    # 3) 타깃 및 주요 변수 분포
    st.subheader("분포 탐색")
    target_col = "YOUTH_NET_RATE" if "YOUTH_NET_RATE" in filtered.columns else None
    default_feats = [c for c in ["CMR", "INDUSTRIAL_AREA", "SELF_FINANCE"] if c in filtered.columns]
    pick_cols = st.multiselect(
        "분포 확인할 컬럼 선택", 
        options=[c for c in num_cols if c != target_col],
        default=default_feats
    )

    # 타깃 분포
    if target_col:
        st.markdown(f"**타깃 분포: {target_col}**")
        fig, ax = plt.subplots()
        ax.hist(filtered[target_col].dropna(), bins=30)
        ax.set_xlabel(target_col)
        ax.set_ylabel("count")
        st.pyplot(fig)

    # 선택 컬럼 분포 그리드(2열)
    if pick_cols:
        cols = st.columns(2)
        for i, col in enumerate(pick_cols):
            fig, ax = plt.subplots()
            ax.hist(filtered[col].dropna(), bins=30)
            ax.set_xlabel(col)
            ax.set_ylabel("count")
            cols[i % 2].pyplot(fig)

    st.divider()

    # 4) 상관분석
    num_cols_all = [c for c in filtered.columns 
                    if pd.api.types.is_numeric_dtype(filtered[c])]
    candidates = [c for c in num_cols_all if c != target_col]

    st.subheader("상관관계 분석")

    # 4-1) 분석 대상 변수 선택 (없으면 전체 숫자형)
    sel_vars = st.multiselect(
        "상관분석에 포함할 변수 선택",
        options=candidates,
        default=["CMR", "INDUSTRIAL_AREA", "SELF_FINANCE"],  # 기본: 전부
        help="원하는 변수만 골라 히트맵/타깃 상관 막대를 봅니다."
    )

    # 4-2) (선택) 타깃 컬럼 선택 UI (target_col이 없다면 대체)
    if not target_col or target_col not in num_cols_all:
        tgt_opt = ["(선택 안 함)"] + num_cols_all
        tgt_pick = st.selectbox("타깃 변수 선택 (선택 시 타깃 상관 막대 표시)", tgt_opt, index=0)
        target = None if tgt_pick == "(선택 안 함)" else tgt_pick
    else:
        target = target_col  # 기존에 정한 타깃 사용

    # 4-3) 히트맵
    if len(sel_vars) >= 2:
        corr = filtered[sel_vars].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(
            corr,
            annot=True,       # 셀 안에 숫자 표시
            fmt=".2f",        # 소수점 2자리까지
            cmap="Blues",
            ax=ax
        )
        ax.set_title("Correlation Heatmap (Selected Variables)")
        st.pyplot(fig)
    else:
        st.info("상관 히트맵을 보려면 2개 이상 변수를 선택하세요.")

    # 4-4) 타깃과의 상관 bar
    if target and target in filtered.columns and len(sel_vars) >= 1:
        corr_t = filtered[sel_vars + [target]].corr()[target].drop(labels=[target])
        use_abs = st.checkbox("절대값 기준 정렬", value=False)
        corr_t = corr_t.reindex(corr_t.abs().sort_values(ascending=False).index) if use_abs \
                else corr_t.sort_values(ascending=False)

        st.markdown(f"**타깃({target})과의 피어슨 상관**")
        fig2, ax2 = plt.subplots(figsize=(6, max(2, 0.3*len(corr_t))))
        corr_t.plot(kind="barh", ax=ax2)
        ax2.invert_yaxis()
        ax2.set_xlabel("corr with target")
        st.pyplot(fig2)

    st.divider()

    # 5) 지리 EDA
    st.subheader("지도 기반 지역 비교 히트맵")
    
    # 데이터 로드 (예시 CSV)
    df = pd.read_csv("./streamlit/data/streamlit_data.csv")  # YEAR, SGG_CODE, 값들 포함
    geojson_obj = load_geojson("./streamlit/data/SGG_GEOJSON.geojson")
    
    # GeoJSON 샘플 코드 길이 확인
    gj_sample_code = str(geojson_obj["features"][0]["properties"]["BJCD"])
    gj_len = len(gj_sample_code)

    # df의 BJCD → 문자열화 + zero-fill
    df["BJCD"] = (
        df["BJCD"].astype(str)
    )

    # 숫자형 변수 후보
    num_cols = [c for c in df.columns if df[c].dtype in ("int64", "float64") and c not in ["YEAR"]]
    map_var = st.selectbox("지도에 표시할 변수", num_cols)

    # 연도 선택
    years = sorted(df["YEAR"].unique().tolist())
    sel_year = st.selectbox("연도 선택", years, index=len(years) - 1)

    # 지도 출력
    draw_choropleth(df, geojson_obj, sel_year, map_var)

    
    # 6) 그룹 비교 (선택 컬럼 존재 시)
    group_keys = [c for c in ["region_type", "urban_class"] if c in filtered.columns]
    if target_col and group_keys:
        st.subheader("그룹별 분포 비교 (Boxplot)")
        gkey = st.selectbox("그룹 컬럼 선택", group_keys)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=filtered, x=gkey, y=target_col, ax=ax)
        ax.set_xlabel(gkey)
        ax.set_ylabel(target_col)
        st.pyplot(fig)

# ---------------------------------
# 3-3)모델/중요도 탭 렌더러
# ---------------------------------
def render_model_tab(df: pd.DataFrame,
                     target_col: str,
                     exclude_cols: set = {"SGG_NAME", "SGG_CODE", "YEAR", "BJCD"},
                     model_paths: list = None):

    if model_paths is None:
        model_paths = [
            "./streamlit/model/ElasticNet(GridSearch)_baseline.pkl",
            "./streamlit/model/XGBoost_ensemble.pkl",
        ]

    st.subheader("모델 결과 · 비교 · 변수중요도 · SHAP 요약")

    # ------- 공통 준비 -------
    if not target_col:
        st.warning("타깃 컬럼을 먼저 선택해주세요.")
        st.stop()
    if target_col not in df.columns:
        st.error(f"타깃 컬럼 `{target_col}` 이(가) 데이터에 없습니다.")
        st.stop()

    cand_feats = [c for c in df.select_dtypes(include=["number"]).columns
                  if c not in exclude_cols and c != target_col]
    if not cand_feats:
        st.error("학습 가능한 숫자형 피처가 없습니다.")
        st.stop()

    X_all = df[cand_feats].copy()
    y_all = df[target_col].copy()

    # ------- 모델 로딩 -------
    loaded, errors = load_models_from_paths(model_paths)
    if errors:
        st.warning("로드 실패:\n" + "\n".join([f"- {e}" for e in errors]))
    if not loaded:
        st.stop()

    # ------- 일괄 평가 -------
    rows = []
    aligned_feature_sets = {}
    for disp_name, model, feat_from_model in loaded:
        X_use, feat_names, missing = align_features(X_all, feat_from_model)
        if not feat_names:
            st.error(f"{disp_name}: 공통 피처 없음 → 평가 불가")
            continue
        if missing:
            st.info(f"⚠️ {disp_name} 누락 피처: {missing}")

        try:
            m = evaluate_model(model, X_use, y_all)
            rows.append({
                "Model": disp_name,
                "Rows": len(X_use),
                "Features": len(feat_names),
                "RMSE": m["rmse"], "MAE": m["mae"], "R²": m["r2"]
            })
            aligned_feature_sets[disp_name] = (X_use, feat_names)
        except Exception as e:
            st.info(f"{disp_name}: 예측/평가 실패 — {e}")

    if not rows:
        st.stop()

    res_df = pd.DataFrame(rows).sort_values(by="RMSE").reset_index(drop=True)
    st.subheader("📈 평가 지표")
    st.dataframe(res_df, use_container_width=True)

    # ------- 상세 분석 대상 선택 -------
    st.divider()
    st.subheader("상세 분석 모델 선택")
    sel_model_name = st.selectbox("모델 선택", [r["Model"] for r in rows])
    sel_tuple = next((t for t in loaded if t[0] == sel_model_name), None)
    if sel_tuple is None:
        st.stop()
    disp_name, sel_model, _ = sel_tuple
    X_use, feat_names = aligned_feature_sets[disp_name][:2]

    # ------- 변수중요도 -------
    st.subheader("🎯 변수중요도")
    importance = get_importance_series(sel_model, feat_names)
    if importance is not None and len(importance) > 0:
        top_k = st.slider("상위 N개 표시", 5, min(30, len(importance)), min(15, len(importance)))
        fig_imp, ax_imp = plt.subplots(figsize=(6, 5))
        importance.head(top_k)[::-1].plot(kind="barh", ax=ax_imp)
        ax_imp.set_title(f"Top Feature Importances — {disp_name}")
        ax_imp.set_xlabel("Importance / |coef|")
        ax_imp.set_ylabel("Feature")
        st.pyplot(fig_imp)
    else:
        st.info("이 모델에서는 변수중요도를 계산할 수 없습니다.")

    # ------- SHAP 요약 + 행별 로컬 중요도 -------
    st.subheader("🧠 SHAP 요약")

    try:

        # 1) Explainer 준비
        explainer, method, _ = choose_shap_explainer(sel_model, X_use)

        # 2) 요약용 SHAP (샘플링)
        sample_n = st.slider("SHAP 계산 샘플 수(요약용)", 100, min(2000, len(X_use)), min(500, len(X_use)))
        X_sample = X_use.sample(n=sample_n, random_state=42) if len(X_use) > sample_n else X_use

        sv_summary = explainer.shap_values(X_sample)
        # 다차원 방어
        if isinstance(sv_summary, list):
            sv_summary = np.array(sv_summary)
        if getattr(sv_summary, "ndim", 2) == 3:
            sv_summary = sv_summary[0]

        st.write(f"**SHAP Summary — {disp_name} (method: {method})**")
        fig_shap = plt.figure(figsize=(7, 5))
        shap.summary_plot(sv_summary, X_sample, show=False)
        st.pyplot(fig_shap)

        st.write("**평균 |SHAP| 상위 15개 (요약)**")
        st.dataframe(mean_abs_shap_top(sv_summary, X_sample, top_k=15), use_container_width=True)

        st.divider()

        # 3) 행 선택 → 로컬 SHAP
        st.subheader("🔎 시군구별 로컬 중요도")

        # 행 식별 컬럼 자동 선택
        if "SGG_NAME" in df.columns:
            id_col = "SGG_NAME"
        elif "SGG_CODE" in df.columns:
            id_col = "SGG_CODE"
        else:
            id_col = None  # 인덱스로 고름

        # 선택 목록
        if id_col:
            candidates = df.loc[X_use.index, id_col].astype(str).tolist()
            default_idx = 0
            sel_label = st.selectbox(f"행(시군구) 선택 — 기준: {id_col}", candidates, index=default_idx)
            # 선택 라벨 → 원본 인덱스 찾기 (동명이 많으면 첫 번째)
            match_idx = df.loc[X_use.index][df.loc[X_use.index, id_col].astype(str) == str(sel_label)].index
            if len(match_idx) == 0:
                st.info("선택한 시군구를 찾을 수 없습니다.")
                st.stop()
            row_idx = match_idx[0]
        else:
            # 인덱스로 선택
            idx_list = list(map(str, X_use.index.tolist()))
            sel_idx_str = st.selectbox("행(인덱스) 선택", idx_list, index=0)
            row_idx = int(sel_idx_str) if sel_idx_str.isdigit() else X_use.index[0]

        # 선택 행의 X(1×features)
        X_row = X_use.loc[[row_idx]]

        # 4) 로컬 SHAP 계산 (1행)
        sv_row = explainer.shap_values(X_row)
        if isinstance(sv_row, list):
            sv_row = np.array(sv_row)
        if getattr(sv_row, "ndim", 3) == 3:
            sv_row = sv_row[0]               # (1, n_features)
        sv_row = np.ravel(sv_row)            # (n_features,)

        # 5) 로컬 중요도 테이블 & 바차트 (Top-k)
        k_local = st.slider("로컬 중요도 — 상위 N개", 5, min(30, len(feat_names)), min(10, len(feat_names)))
        local_series = pd.Series(np.abs(sv_row), index=feat_names).sort_values(ascending=False)

        st.write("**선택 행 로컬 |SHAP| 상위**")
        st.dataframe(local_series.head(k_local).rename("local|SHAP|").to_frame(), use_container_width=True)

        fig_local, ax_local = plt.subplots(figsize=(6, 5))
        local_series.head(k_local)[::-1].plot(kind="barh", ax=ax_local)
        ax_local.set_title("Local Feature Importance (|SHAP|)")
        ax_local.set_xlabel("|SHAP|")
        ax_local.set_ylabel("Feature")
        st.pyplot(fig_local)

        # (선택) 방향성까지 보고 싶으면 원시 SHAP 값으로도 표시
        with st.expander("부호 포함 SHAP 값 보기 (선택)", expanded=False):
            signed_series = pd.Series(sv_row, index=feat_names).sort_values(key=lambda x: np.abs(x), ascending=False)
            st.dataframe(signed_series.head(k_local).rename("SHAP (signed)").to_frame(), use_container_width=True)

    except Exception as e:
        st.info(f"SHAP 계산 불가/로컬 중요도 에러: {e}")

# ---------------------------------
# 3-4)모델/중요도 탭 렌더러
# ---------------------------------
def render_region_tab(
    df: pd.DataFrame,
    target_col: str,
    geojson_path: str = "./streamlit/data/SGG_GEOJSON.geojson",
    df_code_col: str = "BJCD",
    df_name_col: str = "SGG_NAME",
    key_ns: str = "region", 
):
    # 키 헬퍼
    k = lambda name: f"{key_ns}:{name}"

    st.subheader("지역별 상세 · 지도 · 비교")

    # --- 점검/전처리 ---
    for c in [target_col, df_code_col, df_name_col]:
        if c not in df.columns:
            st.error(f"필수 컬럼 누락: `{c}`")
            st.stop()
    df = ensure_bjcd_string(df, col=df_code_col)

    # --- 컨트롤 ---
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_col]
    main_var = st.selectbox("지도에 표시할 변수", [target_col] + num_cols, index=0, key=k("main_var"))

    years = sorted(df["YEAR"].dropna().unique().tolist()) if "YEAR" in df.columns else None
    year = st.selectbox("연도 선택", years, index=len(years)-1, key=k("year")) if years else None
    dff = df if year is None else df[df["YEAR"] == year].copy()
    
    # --- 팝업에 넣을 컬럼 결정 ---
    # target_col을 우선('youth_move_rate' 계열), 나머지는 케이스 무시로 탐색

    youth_col = target_col or find_col_ci(dff, "youth_net_move_rate")
    cmr_col   = find_col_ci(dff, "cmr")
    ind_col   = find_col_ci(dff, "industrial_area", "industrial_Area")
    self_col  = find_col_ci(dff, "self_finance", "selfFinance")

    # 실제 존재하는 것만 사용
    popup_cols = [c for c in [youth_col, cmr_col, ind_col, self_col] if c]

    if dff.empty:
        st.warning("선택한 조건의 데이터가 비어 있습니다.")
        st.stop()

    # 지역 상세 카드 (여기서부터 key 제거)
    default_bjcd = st.session_state.get("selected_bjcd")
    if default_bjcd and default_bjcd in dff[df_code_col].values:
        default_region = dff.loc[dff[df_code_col] == default_bjcd, df_name_col].iloc[0]
    else:
        default_region = None

    region_opts = sorted(dff[df_name_col].dropna().astype(str).unique().tolist())
    idx_default = region_opts.index(default_region) if default_region in region_opts else 0
    sel_region = st.selectbox("지역 선택", region_opts, index=idx_default, key=k("region_select"))

    row = dff[dff[df_name_col].astype(str) == str(sel_region)].iloc[0]
    sel_bjcd = str(row[df_code_col])
    val = float(row[main_var]) if pd.notnull(row[main_var]) else np.nan
    pct = pct_rank(dff[main_var], val)

    c1, c2, c3 = st.columns(3)
    c1.metric("지역명", sel_region)  # ❌ key 제거
    c2.metric(f"{main_var}", f"{val:,.3f}" if pd.notnull(val) else "NA")
    c3.metric("분위(%)", f"{pct:.1f}%" if pd.notnull(pct) else "NA")

    # 박스플롯 렌더링 주석처리
    # fig_box, ax_box = plt.subplots(figsize=(4, 2.4))
    # ax_box.boxplot(dff[main_var].dropna(), vert=False, widths=0.5)
    # if pd.notnull(val): ax_box.axvline(val, linestyle="--", linewidth=2)
    # ax_box.set_title(f"{main_var} 분포와 선택 지역")
    # st.pyplot(fig_box)  # ❌ key 없음

    st.divider()
    
    # 지도
    try:
        with open(geojson_path, "r", encoding="utf-8") as f:
            gjson = json.load(f)
    except Exception as e:
        st.error(f"GeoJSON 로드 실패: {e}")
        st.stop()

    val_map = make_val_map(dff, df_code_col, popup_cols, df_name_col)
    label_map = {}
    if youth_col: label_map[youth_col] = "청년 순 이동률"
    if cmr_col:   label_map[cmr_col]   = "조혼인율"
    if ind_col:   label_map[ind_col]   = "공업지역 면적"
    if self_col:  label_map[self_col]  = "재정자립도"

    # --- (5) 지도 생성 + 채색 ---
    m = folium.Map(location=[36.5, 127.9], zoom_start=7, tiles="cartodbpositron")
    folium.Choropleth(
        geo_data=gjson,
        data=dff,
        columns=[df_code_col, main_var],
        key_on="feature.properties.BJCD",
        fill_color="YlOrRd",
        nan_fill_color="lightgray",
        fill_opacity=0.75,
        line_opacity=0.5,
        legend_name=f"{main_var}",
    ).add_to(m)

    # 지역 선택값을 세션에도 공유 (모델/정책 탭과 연동 원하면)
    # st.session_state["selected_bjcd"] = sel_bjcd

    # 선택 지역 자동 하이라이트 + 팝업 오픈 + 영역으로 줌
    if sel_bjcd:
        feat_sel = next(
            (f for f in gjson.get("features", []) if str(f.get("properties", {}).get("BJCD", "")) == str(sel_bjcd)),
            None
        )
        if feat_sel:
            folium.GeoJson(
                feat_sel,
                style_function=lambda x: {"fillOpacity": 0, "color": "#2a66ff", "weight": 3},
                highlight_function=lambda x: {"weight": 3, "color": "#2a66ff"},
                popup=folium.Popup(
                    popup_html_for_code(sel_bjcd, val_map, popup_cols, label_map, df_name_col),
                    max_width=300,
                    show=True  # ← 오픈 상태로
                ),
                name="selected-region"
            ).add_to(m)

            b = feature_bounds(feat_sel)
            if b:
                m.fit_bounds(b)
        else:
            st.caption("선택한 지역(BJCD)이 GeoJSON에 없습니다.")

    for feat in gjson.get("features", []):
        code_ = str(feat["properties"].get("BJCD", ""))
        rec = val_map.get(code_)
        if rec:
            lines = [f"<b>{rec.get(df_name_col, '')}</b>", f"CODE: {code_}"]
            for col in popup_cols:
                v = rec.get(col)
                label = label_map.get(col, col)
                if isinstance(v, (int, float, np.number)) and pd.notnull(v):
                    lines.append(f"{label}: {v:,.3f}")
                elif v is not None and str(v) != "nan":
                    lines.append(f"{label}: {v}")
            popup_html = "<br>".join(lines)
        else:
            popup_html = f"CODE: {code_}<br>(데이터 없음)"

        folium.GeoJson(
            feat,
            style_function=lambda x: {"fillOpacity": 0, "weight": 0},
            popup=folium.Popup(popup_html, max_width=280)
        ).add_to(m)

    # --- (7) 지도 렌더 ---
    out = st_folium(m, height=540, use_container_width=True)

    st.divider()

    # 비교 (4개 지표 고정: CMR, SELF_FINANCE, INDUSTRIAL_AREA, YOUTH_NET_MOVE_RATE)
    compare_targets = st.multiselect("비교할 지역 선택", region_opts, default=[sel_region], key=k("compare"))

    # 1) 컬럼 자동 매칭 (대소문자/케이스 무시)
    col_map = {
        "CMR":                 find_col_ci(dff, "CMR", "cmr"),
        "SELF_FINANCE":        find_col_ci(dff, "SELF_FINANCE", "self_finance", "selfFinance"),
        "INDUSTRIAL_AREA":     find_col_ci(dff, "INDUSTRIAL_AREA", "industrial_area", "industrial_Area"),
        "YOUTH_NET_MOVE_RATE": find_col_ci(dff, "YOUTH_NET_MOVE_RATE", "youth_net_move_rate", "youth_move_rate", target_col),
    }
    use_map = {k: v for k, v in col_map.items() if v}  # 실제 존재하는 컬럼만

    if len(use_map) < 1:
        st.warning("비교할 지표 컬럼을 데이터에서 찾지 못했습니다. 컬럼명을 확인해주세요.")
    else:
        # 2) 비교 테이블 구성
        cols_for_table = [df_name_col, df_code_col] + list(use_map.values())
        comp = dff[dff[df_name_col].astype(str).isin(compare_targets)][cols_for_table].copy()
        comp = comp.rename(columns={df_name_col: "REGION", df_code_col: "BJCD", **use_map})

        # 숫자형 보정
        for m in use_map.keys():
            comp[m] = pd.to_numeric(comp[m], errors="coerce")

        st.dataframe(comp.reset_index(drop=True), use_container_width=True)

        # 3) 그룹형 막대그래프 (지표 × 지역)
        if len(comp) >= 1:
            metrics = list(use_map.keys())  # ['CMR','SELF_FINANCE','INDUSTRIAL_AREA','YOUTH_NET_MOVE_RATE']
            x = np.arange(len(metrics))
            width = 0.8 / max(1, len(comp))  # 지역 수에 따라 막대 폭 조정

            fig, ax = plt.subplots(figsize=(8, 4))
            for i, (_, r) in enumerate(comp.iterrows()):
                y = [r.get(m) for m in metrics]
                ax.bar(x + i*width - (len(comp)-1)*width/2, y, width, label=str(r["REGION"]))

            ax.set_xticks(x)
            ax.set_xticklabels(metrics, rotation=20, ha="right")
            ax.set_title("지역 비교")
            ax.set_xlabel("지표")
            ax.set_ylabel("값")
            ax.legend()
            st.pyplot(fig)

# ---------------------------------
# 3-5)정책 시나리오 탭 렌더러
# ---------------------------------
def render_policy_tab(
    df: pd.DataFrame,
    target_col: str,
    model_paths: list = None,
    df_code_col: str = "BJCD",
    df_name_col: str = "SGG_NAME",
    key_ns: str = "policy",
):
    """
    정책 시나리오(What-if) · 핵심 변수 슬라이더 · 결과 비교
    """
    k = lambda name: f"{key_ns}:{name}"
    st.subheader("정책 시나리오 (What-if) — 변수 조정과 결과 비교")

    # 기본값: 두 모델 로컬 경로
    if model_paths is None:
        model_paths = [
            "/mnt/data/ElasticNet(GridSearch)_baseline.pkl",
            "/mnt/data/XGBoost_ensemble.pkl",
        ]

    # ------- 데이터 점검/준비 -------
    for c in [target_col, df_code_col, df_name_col]:
        if c not in df.columns:
            st.error(f"필수 컬럼 누락: `{c}`")
            st.stop()
    df = ensure_bjcd_string(df, col=df_code_col)

    years = sorted(df["YEAR"].dropna().unique().tolist()) if "YEAR" in df.columns else None
    year = st.selectbox("기준 연도 선택", years, index=len(years)-1 if years else 0, key=k("year")) if years else None
    dff = df if year is None else df[df["YEAR"] == year].copy()
    if dff.empty:
        st.warning("선택 연도의 데이터가 비어 있습니다.")
        st.stop()

    # 숫자형 피처(타깃 제외)
    EXCLUDE_COLS = {df_code_col, "YEAR"}
    num_feats = [
    c for c in dff.select_dtypes(include=[np.number]).columns
    if c not in EXCLUDE_COLS and c != target_col]

    # ------- 모델 로딩 & 선택 -------
    loaded, errors = load_models_from_paths(model_paths)
    if errors:
        st.warning("모델 로드 실패:\n" + "\n".join([f"- {e}" for e in errors]))
    if not loaded:
        st.stop()

    model_name_list = [name for name, _, _ in loaded]
    sel_model_name = st.selectbox("시나리오에 사용할 모델", model_name_list, key=k("model"))
    # 선택 모델 튜플
    disp_name, model, feat_from_model = next((t for t in loaded if t[0] == sel_model_name), loaded[0])

    # 피처 정합성
    X_all = dff[num_feats].copy()
    X_use, feat_names, missing = align_features(X_all, feat_from_model)
    if not feat_names:
        st.error("모델과 일치하는 피처가 없어 시나리오 예측을 할 수 없습니다.")
        st.stop()
    if missing:
        st.info(f"⚠️ 모델이 기대하는 누락 피처: {missing}")

    # ------- 지역 범위 선택 -------
    st.markdown("### 지역 범위")
    default_bjcd = st.session_state.get("selected_bjcd")
    mode = st.radio("적용 대상", ["전체", "지역 선택"], horizontal=True, key=k("scope"))

    if mode == "전체":
        idx_scope = X_use.index
    elif mode == "지역 선택":
        region_opts = dff[df_name_col].dropna().astype(str).unique().tolist()
        regions = st.multiselect("지역 선택", sorted(region_opts), key=k("regions"))
        if not regions:
            st.stop()
        idx_scope = dff[dff[df_name_col].astype(str).isin(regions)].index.intersection(X_use.index)
        if len(idx_scope) == 0:
            st.warning("선택 지역이 모델 입력 인덱스와 교집합이 없습니다.")
            st.stop()
    else:
        if not default_bjcd:
            st.info("지도 탭에서 지역을 클릭하면 BJCD가 저장됩니다. (st.session_state['selected_bjcd'])")
            st.stop()
        idx_scope = dff[dff[df_code_col].astype(str) == str(default_bjcd)].index.intersection(X_use.index)
        if len(idx_scope) == 0:
            st.warning("선택된 BJCD가 현재 데이터 인덱스와 일치하지 않습니다.")
            st.stop()

    X_base = X_use.loc[idx_scope].copy()
    names_scope = dff.loc[idx_scope, df_name_col].astype(str)

    # ------- 변수 조정 슬라이더 -------
    st.markdown("### 핵심 변수 조정")

    # 자동 컬럼 매칭(대/소문자 무시)
    col_CMR   = find_col_ci(dff, "CMR", "cmr")
    col_SELF  = find_col_ci(dff, "SELF_FINANCE", "self_finance", "selfFinance")
    col_IND   = find_col_ci(dff, "INDUSTRIAL_AREA", "industrial_area", "industrial_Area")

    # 슬라이더(±50%) — 존재하는 컬럼만 노출
    pct_changes = {}
    col_labels  = []
    for label, col in [("CMR", col_CMR), ("SELF_FINANCE", col_SELF), ("INDUSTRIAL_AREA", col_IND)]:
        if col and col in feat_names:
            pct = st.slider(f"{label} 증감(%)", -50, 50, 0, step=5, key=k(f"pct_{label}"))
            pct_changes[col] = pct / 100.0
            col_labels.append((label, col))
        elif col and col not in feat_names:
            st.caption(f"• {label} 컬럼은 데이터엔 있으나 선택 모델 피처에 포함되지 않아 조정이 적용되지 않습니다.")

    # 추가 변수(옵션)
    with st.expander("추가 변수 조정 (선택)", expanded=False):
        other_feats = [c for c in feat_names if c not in [c for _, c in col_labels]]
        add_cols = st.multiselect("추가로 조정할 변수 선택", other_feats, key=k("extra_feats"))
        for col in add_cols:
            pct = st.slider(f"{col} 증감(%)", -50, 50, 0, step=5, key=k(f"pct_{col}"))
            pct_changes[col] = pct / 100.0

    # 검증
    if len(pct_changes) == 0:
        st.info("조정할 변수를 선택하세요.")
        st.stop()

    # ------- 시나리오 적용 & 예측 -------
    X_scn = X_base.copy()
    # 원본 분포 과도 이탈 방지용 클리핑 범위
    q_low  = X_use.quantile(0.01)
    q_high = X_use.quantile(0.99)

    for col, pct in pct_changes.items():
        if col in X_scn.columns:
            X_scn[col] = X_scn[col] * (1.0 + pct)
            # 분포 밖 극단치 클리핑
            X_scn[col] = X_scn[col].clip(lower=q_low.get(col, None), upper=q_high.get(col, None))

    # 예측
    try:
        y_base = model.predict(X_base)
        y_scn  = model.predict(X_scn)
    except Exception as e:
        st.error(f"예측 실패: {e}")
        st.stop()

    st.divider()
    # ------- 결과 비교 -------
    res = pd.DataFrame({
        "REGION": names_scope.values,
        "BASELINE": y_base,
        "SCENARIO": y_scn,
    }, index=idx_scope)
    res["DELTA"] = res["SCENARIO"] - res["BASELINE"]
    res["DELTA_%"] = np.where(res["BASELINE"] != 0, res["DELTA"] / np.abs(res["BASELINE"]) * 100.0, np.nan)

    st.markdown("### 결과 비교")
    st.dataframe(
        res.reset_index(drop=True).sort_values(by="DELTA", ascending=False),
        use_container_width=True
    )

    # 요약 카드
    c1, c2, c3 = st.columns(3)
    c1.metric("평균 변화", f"{res['DELTA'].mean():,.3f}")
    c2.metric("상승 지역 수", int((res["DELTA"] > 0).sum()))
    c3.metric("하락 지역 수", int((res["DELTA"] < 0).sum()))

    # 그래프
    st.divider()
    # --- 변화량 막대그래프 (상위/하위) ---
    st.markdown("#### 변화량 막대그래프 (상위/하위)")

    n = len(res)
    if n <= 0:
        st.info("표시할 결과가 없습니다.")
    elif n == 1:
        # 슬라이더 없이 단일 막대만 표시
        st.caption("지역이 1개뿐이라 슬라이더 없이 표시합니다.")
        res_sorted = res.sort_values(by="DELTA", ascending=False)
        show = res_sorted
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(show["REGION"].astype(str), show["DELTA"])
        ax.set_title(f"예측 변화량 (모델: {disp_name})")
        ax.set_xlabel("지역")
        ax.set_ylabel("Δ (Scenario - Baseline)")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)
    else:
        # n >= 2인 경우에만 슬라이더 사용 (min < max 보장)
        maxN = min(30, n)              # 데이터 수 이내로 상한
        defaultN = min(10, maxN)       # 기본값은 최대 10
        topN = st.slider("표시할 상/하위 N",
                        min_value=1,
                        max_value=maxN,
                        value=defaultN,
                        step=1,
                        key=k("topN"))

        res_sorted = res.sort_values(by="DELTA", ascending=False)
        # 상위 N + 하위 N 결합 (중복 제거)
        show = pd.concat([res_sorted.head(topN), res_sorted.tail(topN)]).drop_duplicates()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(show["REGION"].astype(str), show["DELTA"])
        ax.set_title(f"예측 변화량 (모델: {disp_name})")
        ax.set_xlabel("지역")
        ax.set_ylabel("Δ (Scenario - Baseline)")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

    # 적용 변수 요약
    with st.expander("적용된 변수 조정 요약"):
        adj_tbl = pd.DataFrame(
            [(lab if lab else col, col, f"{pct*100:+.0f}%") for lab, col in col_labels] +
            [(None, c, f"{pct*100:+.0f}%") for c, pct in pct_changes.items() if c not in [c for _, c in col_labels]],
            columns=["Label", "Column", "Change"]
        )
        st.dataframe(adj_tbl, use_container_width=True)

# ---------------------------------
# 3-6)문서/링크 탭 렌더러
# ---------------------------------
def render_docs_tab(
    df: pd.DataFrame,
    target_col: str,
    code_col: str = "BJCD",
    name_col: str = "SGG_NAME",
    key_ns: str = "docs",
):
    k = lambda name: f"{key_ns}:{name}"

    st.subheader("🔗 문서/링크 모음")

    # 세션 상태에 링크 목록 준비 (자유롭게 라벨/URL 추가 가능)
    if "docs_links" not in st.session_state:
        st.session_state["docs_links"] = [
            {"label": "Git", "url": "https://github.com/outflow-project/population-outflow"},
            {"label": "보고서", "url": "https://drive.google.com/file/d/12XRVuKOucmzVNWLxbsJVoCGfMXDYNS31/view?usp=sharing"},
            {"label": "WBS", "url": "https://docs.google.com/spreadsheets/d/e/2PACX-1vRb9ohd88EGHGt8LsJhzMdrWoJuaOZsXOY50SoAreutAd5NDPouQTa1Y0wSdAIevUNgAa5AMlqC9Rm8/pubhtml?gid=1115838130&single=true"},
            {"label": "발표자료", "url": ""},
        ]

    mode = st.radio("모드 선택", ["보기"], horizontal=True, key=k("mode"))

    if mode == "편집":
        # 자유롭게 행 추가/삭제 가능한 에디터
        import pandas as pd
        links_df = pd.DataFrame(st.session_state["docs_links"])
        edited = st.data_editor(
            links_df,
            num_rows="dynamic",           # 행 추가/삭제 허용
            use_container_width=True,
            key=k("editor"),
        )
        col_save, col_reset = st.columns([1,1])
        with col_save:
            if st.button("저장", key=k("save")):
                st.session_state["docs_links"] = edited.to_dict("records")
                st.success("링크를 저장했습니다.")
        with col_reset:
            if st.button("기본값 복원", key=k("reset")):
                st.session_state["docs_links"] = [
                    {"label": "Git", "url": ""},
                    {"label": "보고서", "url": ""},
                    {"label": "WBS", "url": ""},
                    {"label": "발표자료", "url": ""},
                ]
                st.info("기본값으로 복원했습니다.")
    else:
        # 보기 모드
        links = st.session_state["docs_links"]
        if not links:
            st.info("등록된 링크가 없습니다. ‘편집’ 모드에서 추가하세요.")
        for item in links:
            label = item.get("label") or "링크"
            url = (item.get("url") or "").strip()
            if url:
                st.markdown(f"- **{label}**: [{url}]({url})")
            else:
                st.markdown(f"- **{label}**: _(미설정)_")


# ---------------------------------
# 4) 사이드바
# ---------------------------------
st.sidebar.title("📊 2조 - 남아주세유")
file = st.sidebar.file_uploader("시군구 연도별 데이터 CSV", type=["csv"])    
df = load_data(file)

# 연도 필터
years = sorted(df["YEAR"].dropna().unique().tolist()) if "YEAR" in df.columns else []
sel_years = st.sidebar.multiselect("연도 필터", years, default=years)
if sel_years:
    df = df[df["YEAR"].isin(sel_years)]

# 시도 필터
if "SGG_NAME" in df.columns:
    # 1) 시도명 추출
    df["SIDO_NAME"] = df["SGG_NAME"].apply(lambda x: str(x).split()[0] if pd.notna(x) else "미정")
    # 2) 사이드바 필터
    sidos = sorted(df["SIDO_NAME"].dropna().unique().tolist())
    sel_sidos = st.sidebar.multiselect("시도 필터", sidos, default=sidos)
    # 3) 필터 적용
    if sel_sidos:
        df = df[df["SIDO_NAME"].isin(sel_sidos)]

# ---------------------------------
# 5) 앱 탭 구성
# ---------------------------------
tab_overview, tab_eda, tab_model, tab_region, tab_policy, tab_docs = st.tabs([
    "개요", "EDA", "모델/중요도", "지역별 리포트", "정책 시나리오", "문서/링크"
])

with tab_overview: # 개요 탭
    render_overview_tab(df, PROJECT_META)

with tab_eda: # 데이터 EDA 탭
    render_eda_tab(df)

with tab_model:
    render_model_tab(df, target_col='YOUTH_NET_MOVE_RATE')
with tab_region:
    render_region_tab(
        df,
        target_col="YOUTH_NET_MOVE_RATE",
        geojson_path="./streamlit/data/SGG_GEOJSON.geojson",
        key_ns="region"   # 탭/페이지마다 다르게 주면 충돌 방지
    )

with tab_policy:
    render_policy_tab(
        df,
        target_col="YOUTH_NET_MOVE_RATE",   # 현재 타깃
        model_paths=[
            "./streamlit/model/ElasticNet(GridSearch)_baseline.pkl",
            "./streamlit/model/XGBoost_ensemble.pkl",
        ],
        df_code_col="BJCD",
        df_name_col="SGG_NAME",
        key_ns="policy"
    )

with tab_docs:
    render_docs_tab(
        df,
        target_col="YOUTH_NET_MOVE_RATE",
        code_col="BJCD",
        name_col="SGG_NAME",
        key_ns="docs"
    )