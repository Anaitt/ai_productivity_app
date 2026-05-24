import os
import hashlib
import pandas as pd
import numpy as np
import kagglehub
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans

st.set_page_config(page_title="AI Productivity & Burnout Predictor", layout="wide")
st.title("Персональный аналитик продуктивности сотрудников")

WORKLOAD_COLS = [
    'ai_tool_usage_hours_per_week', 'learning_time_hours_per_week',
    'manual_work_hours_per_week', 'meeting_hours_per_week',
    'collaboration_hours_per_week'
]

NUMERICAL_FEATURES = [
    'experience_years', 'ai_tool_usage_hours_per_week',
    'tasks_automated_percent', 'manual_work_hours_per_week',
    'learning_time_hours_per_week', 'meeting_hours_per_week',
    'collaboration_hours_per_week', 'error_rate_percent',
    'task_complexity_score', 'focus_hours_per_day',
    'work_life_balance_score', 'workload_hours_per_week', 'ai_ratio'
]

BURNOUT_LABELS = ['Low', 'Medium', 'High']
DEADLINE_ORDER = [['Low', 'Medium', 'High']]
ALPHAS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

CLUSTER_NAMES = {
    0: "🤖 ИИ-Оптимизатор",
    1: "⚠️ Группа риска",
    2: "⚙️ Традиционный исполнитель",
    3: "📚 Начинающий / Растущий"
}

CLUSTER_COLORS = {
    0: "#2ecc71",
    1: "#e74c3c",
    2: "#f39c12",
    3: "#3498db"
}

PALETTE = {
    "Low":    "#2ecc71",
    "Medium": "#f39c12",
    "High":   "#e74c3c"
}


# =============================================================================
# ЗАГРУЗКА И ОБРАБОТКА ДАННЫХ
# =============================================================================
@st.cache_data
def load_and_process_data():
    folder_path = kagglehub.dataset_download(
        'vishardmehta/ai-tool-usage-and-workplace-productivity-dataset'
    )
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if len(csv_files) < 2:
        st.error("Ошибка: Недостаточно CSV-файлов в датасете.")
        return None

    features_df = pd.read_csv(os.path.join(folder_path, csv_files[0]))
    targets_df  = pd.read_csv(os.path.join(folder_path, csv_files[1]))

    df = pd.merge(features_df, targets_df, on='Employee_ID', how='inner')
    df['workload_hours_per_week'] = df[WORKLOAD_COLS].sum(axis=1)
    df['ai_ratio'] = (
        df['ai_tool_usage_hours_per_week']
        .div(df['workload_hours_per_week'])
        .fillna(0)
    )

    # KMeans кластеризация по всему датасету
    cluster_features = [
        'tasks_automated_percent', 'ai_ratio',
        'work_life_balance_score', 'error_rate_percent',
        'manual_work_hours_per_week', 'learning_time_hours_per_week',
        'focus_hours_per_day'
    ]
    scaler_km = StandardScaler()
    X_km = scaler_km.fit_transform(df[cluster_features].fillna(0))
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['cluster'] = km.fit_predict(X_km)

    # Переименование кластеров по логике центроидов
    centers = pd.DataFrame(km.cluster_centers_, columns=cluster_features)
    order = (
        centers['tasks_automated_percent']
        .rank(ascending=False)
        .astype(int) - 1
    )
    df['cluster'] = df['cluster'].map(order)

    return df


# =============================================================================
# ПАЙПЛАЙН ПРИЗНАКОВ
# =============================================================================
def build_features(df_input, scaler=None, ohe=None,
                   ord_deadline=None, ord_burnout=None, fit=False):
    df = df_input.copy()

    if fit:
        ord_deadline = OrdinalEncoder(categories=DEADLINE_ORDER)
        df['deadline_pressure_encoded'] = ord_deadline.fit_transform(
            df[['deadline_pressure_level']]
        )
    else:
        df['deadline_pressure_encoded'] = ord_deadline.transform(
            df[['deadline_pressure_level']]
        )

    if fit:
        ohe = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
        job_encoded = ohe.fit_transform(df[['job_role']])
    else:
        job_encoded = ohe.transform(df[['job_role']])

    job_cols = [f'role_{cat}' for cat in ohe.categories_[0][1:]]
    job_df   = pd.DataFrame(job_encoded, columns=job_cols, index=df.index)

    if fit:
        scaler    = StandardScaler()
        X_num_scaled = scaler.fit_transform(df[NUMERICAL_FEATURES].values)
    else:
        X_num_scaled = scaler.transform(df[NUMERICAL_FEATURES].values)

    X_num_df = pd.DataFrame(X_num_scaled, columns=NUMERICAL_FEATURES, index=df.index)
    X = pd.concat([X_num_df, job_df, df[['deadline_pressure_encoded']]], axis=1)

    y_prod = df['productivity_score']

    if fit:
        ord_burnout = OrdinalEncoder(categories=[BURNOUT_LABELS])
        y_burn = ord_burnout.fit_transform(df[['burnout_risk_level']]).ravel()
    else:
        y_burn = ord_burnout.transform(df[['burnout_risk_level']]).ravel()

    if fit:
        return X, y_prod, y_burn, scaler, ohe, ord_deadline, ord_burnout
    return X, y_prod, y_burn


# =============================================================================
# ОБУЧЕНИЕ МОДЕЛЕЙ
# =============================================================================
@st.cache_resource
def train_models(data_df):
    unique_ids = list(data_df['Employee_ID'].unique())
    train_ids, temp_ids = train_test_split(unique_ids, test_size=0.3, random_state=42)
    valid_ids, _        = train_test_split(temp_ids,  test_size=0.5, random_state=42)

    train_df = data_df[data_df['Employee_ID'].isin(train_ids)].copy()
    valid_df = data_df[data_df['Employee_ID'].isin(valid_ids)].copy()

    X_train, y_train_prod, y_train_burn, scaler, ohe, ord_deadline, ord_burnout = \
        build_features(train_df, fit=True)

    X_valid, y_valid_prod, _ = build_features(
        valid_df, scaler=scaler, ohe=ohe,
        ord_deadline=ord_deadline, ord_burnout=ord_burnout
    )

    best_alpha = max(
        ALPHAS,
        key=lambda a: r2_score(
            y_valid_prod,
            Ridge(alpha=a, random_state=42).fit(X_train, y_train_prod).predict(X_valid)
        )
    )

    model_prod = Ridge(alpha=best_alpha, random_state=42)
    model_prod.fit(X_train, y_train_prod)

    model_burnout = LogisticRegression(
        class_weight='balanced', max_iter=2000, random_state=42
    )
    model_burnout.fit(X_train, y_train_burn)

    return model_prod, model_burnout, scaler, ohe, ord_deadline, ord_burnout


# =============================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ — ИНСАЙТЫ
# =============================================================================
def get_productivity_tier(score: float) -> tuple[str, str]:
    if score >= 85:
        return "Топ-исполнитель", "#2ecc71"
    if score >= 70:
        return "Выше среднего", "#27ae60"
    if score >= 55:
        return "Средний уровень", "#f39c12"
    return "Ниже среднего", "#e74c3c"


def render_gauge(value: float, title: str, color: str):
    """Простая горизонтальная шкала-gauge через matplotlib."""
    fig, ax = plt.subplots(figsize=(4, 0.55))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    ax.barh(0, 100, color="#2c2f36", height=0.5, zorder=1)
    ax.barh(0, value, color=color, height=0.5, zorder=2)
    ax.text(value + 1, 0, f"{value:.1f}", va='center',
            color='white', fontsize=9, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.set_title(title, color='white', fontsize=9, pad=4)
    plt.tight_layout(pad=0.2)
    return fig


def render_employee_insights(emp, current_prod, current_burn,
                             predicted_prod, predicted_burn_label):
    """Блок персональных инсайтов по сотруднику."""

    cluster_id   = int(emp.get('cluster', 0))
    cluster_name = CLUSTER_NAMES.get(cluster_id, "Неизвестно")
    cluster_clr  = CLUSTER_COLORS.get(cluster_id, "#aaaaaa")

    prod_tier, prod_color = get_productivity_tier(current_prod)
    burn_color = PALETTE.get(current_burn, "#aaaaaa")

    st.markdown("---")
    st.subheader("🧠 Персональные инсайты на основе модели и кластера")

    # ── Кластер + уровень продуктивности ──────────────────────────────────────
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.markdown(
            f"""
            <div style="background:#1e2130;border-left:5px solid {cluster_clr};
                        padding:16px;border-radius:8px;height:100%">
                <div style="color:#aaa;font-size:12px;margin-bottom:4px">
                    КЛАСТЕР СОТРУДНИКА
                </div>
                <div style="color:{cluster_clr};font-size:20px;font-weight:700">
                    {cluster_name}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div style="background:#1e2130;border-left:5px solid {prod_color};
                        padding:16px;border-radius:8px;height:100%">
                <div style="color:#aaa;font-size:12px;margin-bottom:4px">
                    УРОВЕНЬ ПРОДУКТИВНОСТИ
                </div>
                <div style="color:{prod_color};font-size:20px;font-weight:700">
                    {prod_tier}
                </div>
                <div style="color:#ccc;font-size:13px">
                    Скор: {current_prod:.1f} / 100
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div style="background:#1e2130;border-left:5px solid {burn_color};
                        padding:16px;border-radius:8px;height:100%">
                <div style="color:#aaa;font-size:12px;margin-bottom:4px">
                    РИСК ВЫГОРАНИЯ (ФАКТ)
                </div>
                <div style="color:{burn_color};font-size:20px;font-weight:700">
                    {current_burn}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge-шкалы ───────────────────────────────────────────────────────────
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.pyplot(render_gauge(
            emp['tasks_automated_percent'], "🤖 Автоматизация, %", "#3498db"
        ))
    with g2:
        st.pyplot(render_gauge(
            emp['work_life_balance_score'] * 10, "⚖️ Work-Life Balance", "#2ecc71"
        ))
    with g3:
        st.pyplot(render_gauge(
            min(emp['error_rate_percent'], 100), "⚠️ Уровень ошибок, %", "#e74c3c"
        ))
    with g4:
        st.pyplot(render_gauge(
            emp['focus_hours_per_day'] / 12 * 100, "🎯 Фокус (доля дня)", "#9b59b6"
        ))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Инсайты по кластеру ───────────────────────────────────────────────────
    cluster_insights = {
        0: {
            "icon": "🤖",
            "title": "Профиль ИИ-Оптимизатора",
            "body": (
                "Вы входите в топ-группу по цифровой зрелости. "
                f"Уровень автоматизации **{emp['tasks_automated_percent']:.0f}%** — "
                "один из лучших показателей. Освободившееся время вы направляете "
                "на задачи высокой сложности.\n\n"
                "**Что это значит:** ваша продуктивность масштабируется вместе с ИИ-инструментами. "
                "Вы создаёте ценность не за счёт количества часов, а за счёт качества решений.\n\n"
                "**Рекомендация:** возьмите роль ментора — ваш опыт поднимет результат всей команды."
            ),
            "type": "success"
        },
        1: {
            "icon": "⚠️",
            "title": "Профиль: Группа риска (Выгорание)",
            "body": (
                f"Баланс жизни **{emp['work_life_balance_score']:.1f}/10** и "
                f"уровень ошибок **{emp['error_rate_percent']:.1f}%** сигнализируют "
                "о накопленной усталости. Совещания занимают "
                f"**{emp['meeting_hours_per_week']:.1f} ч/нед** — это выше нормы.\n\n"
                "**Что это значит:** продуктивность снижается не из-за нехватки навыков, "
                "а из-за перегрузки и отсутствия восстановления.\n\n"
                "**Рекомендация:** в первую очередь — разгрузить календарь. "
                "Каждый сокращённый час совещаний даёт +1.5–2 балла продуктивности."
            ),
            "type": "error"
        },
        2: {
            "icon": "⚙️",
            "title": "Профиль: Традиционный Исполнитель",
            "body": (
                f"Ручной труд занимает **{emp['manual_work_hours_per_week']:.1f} ч/нед**, "
                f"при этом автоматизировано лишь **{emp['tasks_automated_percent']:.0f}%** задач. "
                "Фокус и дисциплина на высоком уровне — вы стабильно выполняете план.\n\n"
                "**Что это значит:** вы упираетесь в потолок времени. "
                "Без автоматизации рост продуктивности ограничен физически.\n\n"
                "**Рекомендация:** автоматизировать 15–20% рутины = высвободить "
                "4–6 ч/нед для задач с высокой добавленной ценностью."
            ),
            "type": "warning"
        },
        3: {
            "icon": "📚",
            "title": "Профиль: Начинающий / Растущий",
            "body": (
                f"Время на обучение — **{emp['learning_time_hours_per_week']:.1f} ч/нед**, "
                f"стаж — **{emp['experience_years']:.0f} лет**. "
                "Вы активно инвестируете в навыки — это главный актив на этом этапе.\n\n"
                "**Что это значит:** текущая продуктивность ещё не отражает потенциал. "
                "При правильном наставничестве через 2–3 месяца возможен переход в кластер 0.\n\n"
                "**Рекомендация:** сфокусируйте обучение на конкретных ИИ-инструментах "
                "для вашей роли — это даст быстрый измеримый результат."
            ),
            "type": "info"
        }
    }

    insight = cluster_insights.get(cluster_id, cluster_insights[3])
    getattr(st, insight["type"])(
        f"#### {insight['icon']} {insight['title']}\n\n{insight['body']}"
    )

    # ── Модельный прогноз vs факт ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔬 Оценка моделей: факт vs прогноз")

    m1, m2, m3 = st.columns(3)

    pred_burn_color = PALETTE.get(predicted_burn_label, "#aaaaaa")
    delta_prod      = predicted_prod - current_prod
    delta_color     = "#2ecc71" if delta_prod >= 0 else "#e74c3c"

    with m1:
        st.markdown(
            f"""
            <div style="background:#1e2130;padding:16px;border-radius:8px;
                        border-top:3px solid #3498db">
                <div style="color:#aaa;font-size:11px">ФАКТ: Продуктивность</div>
                <div style="color:#fff;font-size:26px;font-weight:700">
                    {current_prod:.1f}
                </div>
                <div style="color:#3498db;font-size:11px">из 100 баллов</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""
            <div style="background:#1e2130;padding:16px;border-radius:8px;
                        border-top:3px solid {delta_color}">
                <div style="color:#aaa;font-size:11px">МОДЕЛЬ: Прогноз продуктивности</div>
                <div style="color:#fff;font-size:26px;font-weight:700">
                    {predicted_prod:.1f}
                </div>
                <div style="color:{delta_color};font-size:11px">
                    Δ {delta_prod:+.1f} vs факт
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f"""
            <div style="background:#1e2130;padding:16px;border-radius:8px;
                        border-top:3px solid {pred_burn_color}">
                <div style="color:#aaa;font-size:11px">МОДЕЛЬ: Риск выгорания</div>
                <div style="color:{pred_burn_color};font-size:26px;font-weight:700">
                    {predicted_burn_label}
                </div>
                <div style="color:#aaa;font-size:11px">прогноз логрег.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ── Радарная диаграмма профиля ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 🕸️ Радар-профиль сотрудника")

    radar_metrics = {
        "Автоматизация":   emp['tasks_automated_percent'] / 100,
        "WLB":             emp['work_life_balance_score']  / 10,
        "Фокус":           emp['focus_hours_per_day']      / 12,
        "Без ошибок":      max(0, 1 - emp['error_rate_percent'] / 20),
        "Обучение":        min(emp['learning_time_hours_per_week'] / 20, 1),
        "Продуктивность":  current_prod / 100,
    }

    labels = list(radar_metrics.keys())
    values = list(radar_metrics.values())
    N      = len(labels)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    values_plot = values + [values[0]]

    fig_r, ax_r = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig_r.patch.set_facecolor('#0e1117')
    ax_r.set_facecolor('#0e1117')

    ax_r.plot(angles, values_plot, color=cluster_clr, linewidth=2)
    ax_r.fill(angles, values_plot, color=cluster_clr, alpha=0.25)
    ax_r.set_xticks(angles[:-1])
    ax_r.set_xticklabels(labels, color='white', size=9)
    ax_r.set_yticklabels([])
    ax_r.set_ylim(0, 1)
    ax_r.spines['polar'].set_color('#444')
    ax_r.grid(color='#444', linewidth=0.5)

    _, rc = st.columns([1, 2])
    with rc:
        st.pyplot(fig_r)
    plt.close(fig_r)


# =============================================================================
# ГРАФИКИ ВКЛАДКИ 2
# =============================================================================
def plot_role_distribution(df: pd.DataFrame):
    st.markdown("#### 👥 Распределение сотрудников по ролям")
    role_counts = df['job_role'].value_counts().reset_index()
    role_counts.columns = ['role', 'count']

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    bars = ax.barh(role_counts['role'], role_counts['count'],
                   color='#3498db', edgecolor='none')
    for bar in bars:
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                f"{int(bar.get_width())}", va='center', color='white', fontsize=9)

    ax.set_xlabel("Количество сотрудников", color='#aaa')
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_productivity_by_role(df: pd.DataFrame):
    st.markdown("#### 📊 Медианная продуктивность по ролям")
    med = (
        df.groupby('job_role')['productivity_score']
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    colors = plt.cm.RdYlGn(
        np.linspace(0.3, 0.9, len(med))[::-1]
    )
    bars = ax.bar(med['job_role'], med['productivity_score'],
                  color=colors, edgecolor='none', zorder=3)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}",
                ha='center', color='white', fontsize=9)

    ax.set_ylabel("Медиана Productivity Score", color='#aaa')
    ax.set_ylim(0, 105)
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color='#333', linewidth=0.5, zorder=0)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_burnout_distribution(df: pd.DataFrame):
    st.markdown("#### 🔥 Распределение уровней выгорания")
    order  = ['Low', 'Medium', 'High']
    counts = df['burnout_risk_level'].value_counts().reindex(order).fillna(0)
    colors = [PALETTE[l] for l in order]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_facecolor('#0e1117')

    # Pie
    ax = axes[0]
    ax.set_facecolor('#0e1117')
    wedges, texts, autotexts = ax.pie(
        counts, labels=order, colors=colors,
        autopct='%1.1f%%', startangle=90,
        textprops={'color': 'white', 'fontsize': 10},
        wedgeprops={'edgecolor': '#0e1117', 'linewidth': 2}
    )
    for at in autotexts:
        at.set_color('white')
    ax.set_title("Общая структура", color='white', fontsize=11)

    # По ролям stacked
    ax2 = axes[1]
    ax2.set_facecolor('#0e1117')
    pivot = (
        df.groupby(['job_role', 'burnout_risk_level'])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=order, fill_value=0)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    bottom = np.zeros(len(pivot_pct))
    for lvl, clr in zip(order, colors):
        ax2.bar(pivot_pct.index, pivot_pct[lvl],
                bottom=bottom, color=clr,
                label=lvl, edgecolor='none', zorder=3)
        bottom += pivot_pct[lvl].values

    ax2.set_ylabel("Доля, %", color='#aaa')
    ax2.tick_params(colors='white')
    ax2.spines[:].set_visible(False)
    ax2.yaxis.grid(True, color='#333', zorder=0)
    ax2.legend(title="Уровень", labelcolor='white',
               facecolor='#1e2130', edgecolor='none',
               title_fontsize=9, fontsize=8)
    ax2.set_title("По ролям (стек %)", color='white', fontsize=11)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_cluster_ai_distribution(df: pd.DataFrame):
    st.markdown("#### 🤖 Распределение по кластерам использования ИИ")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.patch.set_facecolor('#0e1117')

    df_plot = df.copy()
    df_plot['cluster_name'] = df_plot['cluster'].map(
        lambda x: CLUSTER_NAMES.get(x, f"Кластер {x}")
    )

    # 1 — Доля кластеров
    ax = axes[0]
    ax.set_facecolor('#0e1117')
    c_counts = df_plot['cluster'].value_counts().sort_index()
    c_colors = [CLUSTER_COLORS.get(i, '#aaa') for i in c_counts.index]
    wedges, texts, autotexts = ax.pie(
        c_counts, colors=c_colors,
        autopct='%1.0f%%', startangle=90,
        textprops={'color': 'white', 'fontsize': 9},
        wedgeprops={'edgecolor': '#0e1117', 'linewidth': 2}
    )
    ax.legend(
        [CLUSTER_NAMES.get(i, i) for i in c_counts.index],
        loc='lower center', bbox_to_anchor=(0.5, -0.3),
        fontsize=7, labelcolor='white',
        facecolor='#1e2130', edgecolor='none'
    )
    ax.set_title("Размер кластеров", color='white', fontsize=10)

    # 2 — AI ratio по кластерам
    ax2 = axes[1]
    ax2.set_facecolor('#0e1117')
    for cid in sorted(df_plot['cluster'].unique()):
        vals   = df_plot[df_plot['cluster'] == cid]['ai_ratio']
        color  = CLUSTER_COLORS.get(cid, '#aaa')
        ax2.hist(vals, bins=20, alpha=0.7, color=color,
                 label=CLUSTER_NAMES.get(cid, cid), edgecolor='none')
    ax2.set_xlabel("AI Ratio", color='#aaa')
    ax2.set_ylabel("Кол-во сотрудников", color='#aaa')
    ax2.tick_params(colors='white')
    ax2.spines[:].set_visible(False)
    ax2.yaxis.grid(True, color='#333', zorder=0)
    ax2.legend(fontsize=7, labelcolor='white',
               facecolor='#1e2130', edgecolor='none')
    ax2.set_title("AI Ratio по кластерам", color='white', fontsize=10)

    # 3 — Продуктивность по кластерам (box)
    ax3 = axes[2]
    ax3.set_facecolor('#0e1117')
    cluster_ids = sorted(df_plot['cluster'].unique())
    data_boxes  = [
        df_plot[df_plot['cluster'] == cid]['productivity_score'].values
        for cid in cluster_ids
    ]
    bp = ax3.boxplot(
        data_boxes,
        patch_artist=True,
        medianprops=dict(color='white', linewidth=2),
        whiskerprops=dict(color='#aaa'),
        capprops=dict(color='#aaa'),
        flierprops=dict(marker='o', color='#aaa', markersize=3)
    )
    for patch, cid in zip(bp['boxes'], cluster_ids):
        patch.set_facecolor(CLUSTER_COLORS.get(cid, '#aaa'))
        patch.set_alpha(0.7)

    ax3.set_xticks(range(1, len(cluster_ids) + 1))
    ax3.set_xticklabels(
        [f"K{cid}" for cid in cluster_ids],
        color='white', fontsize=8
    )
    ax3.set_ylabel("Productivity Score", color='#aaa')
    ax3.tick_params(colors='white')
    ax3.spines[:].set_visible(False)
    ax3.yaxis.grid(True, color='#333', zorder=0)
    ax3.set_title("Продуктивность по кластерам", color='white', fontsize=10)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_productivity_burnout_scatter(df: pd.DataFrame):
    st.markdown("#### 🎯 Продуктивность vs Уровень выгорания")

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')

    for lvl in BURNOUT_LABELS:
        sub = df[df['burnout_risk_level'] == lvl]
        ax.scatter(
            sub['error_rate_percent'],
            sub['productivity_score'],
            c=PALETTE[lvl], alpha=0.4, s=18,
            label=lvl, edgecolors='none'
        )

    ax.set_xlabel("Error Rate, %", color='#aaa')
    ax.set_ylabel("Productivity Score", color='#aaa')
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color='#333', zorder=0)
    ax.xaxis.grid(True, color='#333', zorder=0)
    ax.legend(title="Burnout", labelcolor='white',
              facecolor='#1e2130', edgecolor='none',
              title_fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# =============================================================================
# ГЛАВНЫЙ РЕНДЕР
# =============================================================================
df = load_and_process_data()

if df is None:
    st.stop()

model_prod, model_burnout, scaler, ohe, ord_deadline, ord_burnout = train_models(df)

tab1, tab2 = st.tabs(["🔍 Поиск сотрудника", "📈 Анализ данных"])

# =============================================================================
# ВКЛАДКА 1
# =============================================================================
with tab1:
    st.subheader("🔍 Персональная диагностика: Продуктивность & Риск выгорания")

    search_name = st.text_input(
        "Введите ФИО:",
        placeholder="Иванов Иван Иванович"
    ).strip()

    if not search_name:
        st.info("Введите имя сотрудника для получения персонального анализа.")
        st.stop()

    name_hash = int(hashlib.md5(search_name.lower().encode()).hexdigest(), 16)
    emp       = df.iloc[name_hash % len(df)]

    current_prod = float(emp['productivity_score'])
    current_burn = str(emp['burnout_risk_level'])

    # Прогноз модели для текущего профиля
    emp_df = pd.DataFrame([emp.to_dict()]).reset_index(drop=True)
    X_emp, _, _ = build_features(
        emp_df, scaler=scaler, ohe=ohe,
        ord_deadline=ord_deadline, ord_burnout=ord_burnout
    )
    predicted_prod       = float(model_prod.predict(X_emp)[0])
    predicted_burn_idx   = int(model_burnout.predict(X_emp)[0])
    predicted_burn_label = BURNOUT_LABELS[predicted_burn_idx]

    st.write("---")
    st.success(f"👤 Сотрудник: **{search_name}**")

    burnout_icon = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}.get(current_burn, '⚪')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📊 Текущая продуктивность", f"{current_prod:.2f}")
    col2.metric(f"{burnout_icon} Риск выгорания", current_burn)
    col3.metric("🤖 Автоматизация (ИИ)", f"{emp['tasks_automated_percent']}%")
    col4.metric("⚠️ Уровень ошибок", f"{emp['error_rate_percent']}%")

    # ── Инсайты ──────────────────────────────────────────────────────────────
    render_employee_insights(
        emp, current_prod, current_burn,
        predicted_prod, predicted_burn_label
    )

    # ── Рекомендации ─────────────────────────────────────────────────────────
    st.write("---")
    st.subheader("💡 Персональный анализ и рекомендации")

    if current_prod > 85 and current_burn == 'Low':
        st.success(
            "🟢 Идеальный баланс параметров\n\n"
            "Сотрудник демонстрирует выдающиеся показатели без признаков выгорания."
        )
    else:
        if current_burn == 'High' or emp['work_life_balance_score'] < 5:
            st.error(
                f"🔴 Критический риск выгорания (Уровень: {current_burn})\n\n"
                f"Баланс жизни: {emp['work_life_balance_score']}/10. "
                "Необходимо уменьшить нагрузку и сократить переработки."
            )
        if emp['tasks_automated_percent'] < 20:
            st.error(
                f"🔴 Слабая автоматизация: {emp['tasks_automated_percent']}%. "
                "Рутинный ручной труд снижает фокус."
            )
        if emp['error_rate_percent'] > 8:
            st.warning(
                f"🟡 Высокий уровень ошибок: {emp['error_rate_percent']}%. "
                "Стоит проверить уровень дедлайнов."
            )
        if current_burn == 'Medium':
            st.warning("🟡 Умеренное выгорание. Рекомендуется превентивный отдых.")
        if current_prod < 65:
            st.warning(
                f"⚠️ Низкая продуктивность ({current_prod:.2f}). Требуется менторство."
            )

    # ── Симуляция ─────────────────────────────────────────────────────────────
    if current_prod <= 85 or current_burn != 'Low':
        st.write("---")
        st.subheader("🔮 Симуляция изменений и прогноз")

        future_df    = pd.DataFrame([emp.to_dict()]).reset_index(drop=True)
        improvements = []

        if emp['tasks_automated_percent'] < 20:
            future_df['tasks_automated_percent'] = 45
            improvements.append("Повышение автоматизации задач до 45%")

        if emp['error_rate_percent'] > 8:
            future_df['error_rate_percent'] = 4
            improvements.append("Снижение уровня ошибок до 4%")

        if emp['work_life_balance_score'] < 5:
            future_df['work_life_balance_score'] = 7
            future_df['manual_work_hours_per_week'] = max(
                future_df.at[0, 'manual_work_hours_per_week'] - 5, 0
            )
            future_df['meeting_hours_per_week'] = max(
                future_df.at[0, 'meeting_hours_per_week'] - 3, 0
            )
            improvements.append("Улучшение Work-Life Balance до 7/10 и разгрузка часов")

        if not improvements and current_prod < 65:
            future_df['tasks_automated_percent'] += 15
            future_df['work_life_balance_score']  += 1.5
            future_df['error_rate_percent'] = max(
                future_df.at[0, 'error_rate_percent'] - 2.5, 0
            )
            improvements.append("Комплексная оптимизация рабочей среды")

        future_df['workload_hours_per_week'] = future_df[WORKLOAD_COLS].sum(axis=1)
        future_df['ai_ratio'] = (
            future_df['ai_tool_usage_hours_per_week']
            .div(future_df['workload_hours_per_week'])
            .fillna(0)
        )

        X_future, _, _ = build_features(
            future_df, scaler=scaler, ohe=ohe,
            ord_deadline=ord_deadline, ord_burnout=ord_burnout
        )

        future_prod      = float(model_prod.predict(X_future)[0])
        future_burn_idx  = int(model_burnout.predict(X_future)[0])
        future_burn      = BURNOUT_LABELS[future_burn_idx]

        if improvements:
            with st.expander("🛠️ Симулируемые изменения", expanded=True):
                for item in improvements:
                    st.markdown(f"* {item}")

            st.info(
                f"📈 Результат симуляции:\n"
                f"— Продуктивность вырастет до **{future_prod:.2f}** "
                f"(прирост: +{future_prod - current_prod:.2f})\n"
                f"— Новый риск выгорания: **{future_burn}**"
            )

# =============================================================================
# ВКЛАДКА 2
# =============================================================================
with tab2:
    st.subheader("🎯 Сегментация и аналитика по команде")

    # ── Графики ───────────────────────────────────────────────────────────────
    plot_role_distribution(df)
    st.write("---")

    ch1, ch2 = st.columns(2)
    with ch1:
        plot_productivity_by_role(df)
    with ch2:
        plot_burnout_distribution(df)

    st.write("---")
    plot_cluster_ai_distribution(df)
    st.write("---")
    plot_productivity_burnout_scatter(df)

    # ── Кластерные профили ────────────────────────────────────────────────────
    st.write("---")
    st.subheader("📋 Архетипы сотрудников по кластерам")

    selected_role = st.selectbox(
        "Выберите профессиональную роль:",
        ['Analyst', 'Designer', 'Developer', 'Manager', 'Marketer', 'Writer']
    )

    c_tab1, c_tab2, c_tab3, c_tab4 = st.tabs([
        "🤖 Кластер 0: ИИ-Оптимизаторы",
        "⚠️ Кластер 1: Группа риска",
        "⚙️ Кластер 2: Традиционные Исполнители",
        "📚 Кластер 3: Начинающие"
    ])

    with c_tab1:
        st.success("#### 🚀 Высокая продуктивность + Максимальный AI Ratio")
        st.markdown(
            "Высокий `tasks_automated_percent`, низкий `error_rate_percent`. "
            "Рутина делегирована алгоритмам, фокус на сложных задачах.\n\n"
            "**Рекомендации:**\n"
            "1. Назначьте амбассадорами технологий и менторами.\n"
            "2. Направляйте на задачи с высоким `task_complexity_score`.\n"
            "3. Обеспечьте карьерный трек — это ключевой цифровой капитал."
        )

    with c_tab2:
        st.error("#### 🚨 Высокий Burnout + Перегрузка")
        st.markdown(
            "Низкий `work_life_balance_score`, высокие `meeting_hours_per_week`. "
            "Растущий `error_rate_percent` из-за хронической усталости.\n\n"
            "**Рекомендации:**\n"
            "1. Введите лимит на созвоны и переведите часть общения в асинхронный формат.\n"
            "2. Ограничьте рабочие задачи во внеурочное время.\n"
            "3. Проведите встречу 1-on-1 для выявления триггеров стресса."
        )

    with c_tab3:
        st.warning("#### 🦾 Высокий Manual Work + Средняя эффективность")
        st.markdown(
            "Высокий `manual_work_hours_per_week`, `ai_ratio` близок к нулю. "
            "Стабильные исполнители, упирающиеся в потолок своего времени.\n\n"
            "**Рекомендации:**\n"
            "1. Предоставьте готовые шаблоны автоматизации и корпоративные промпты.\n"
            "2. Поставьте цель: автоматизировать 15–20% рутины за месяц.\n"
            "3. Проверьте, есть ли у них доступ к нужным ИИ-инструментам."
        )

    with c_tab4:
        st.info("#### 🎯 Высокий Learning Time + Потенциал роста")
        st.markdown(
            "Высокий `learning_time_hours_per_week`, небольшой `experience_years`. "
            "Продуктивность в стадии роста.\n\n"
            "**Рекомендации:**\n"
            "1. Закрепите наставника из Кластера 0.\n"
            "2. Следите, чтобы обучение не вредило дедлайнам.\n"
            "3. Инвестируйте в профильные курсы — через 2–3 месяца станут ядром команды."
        )

    st.write("---")
    st.subheader(f"💡 Специфика роли {selected_role}")

    if selected_role in ['Developer', 'Analyst']:
        st.info(
            f"Для **{selected_role}** ключевые драйверы кластеризации — "
            "`task_complexity_score` и `error_rate_percent`. "
            "Автоматизация рутины даёт максимальный скачок в качестве работы."
        )
    elif selected_role in ['Designer', 'Writer']:
        st.info(
            f"Для **{selected_role}** критически важен баланс "
            "`focus_hours_per_day` и совещаний. "
            "Творческие профили теряют продуктивность быстрее при нехватке фокусного времени."
        )
    else:
        st.info(
            f"Для **{selected_role}** основное разделение идёт по "
            "`collaboration_hours_per_week` и `tasks_automated_percent`. "
            "Автоматизация отчётности кардинально снижает риск выгорания."
        )
