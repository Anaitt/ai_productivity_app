import os
import pandas as pd
import kagglehub
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="AI Productivity Predictor", layout="wide")
st.title("📊 Персональный аналитик продуктивности сотрудников")

# === НАСТРОЙКА ПРИЗНАКОВ ДЛЯ МОДЕЛИ ===
# Используем базовые временные метрики, которые точно есть в датасете
MY_FEATURES = [
    'tasks_automated_percent',
    'experience_years',
    'focus_hours_per_day',
    'work_life_balance_score',
    'error_rate_percent'
]


@st.cache_data
def load_and_process_data():
    folder_path = kagglehub.dataset_download('vishardmehta/ai-tool-usage-and-workplace-productivity-dataset')
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

    if len(csv_files) < 2:
        st.error("Ошибка: Недостаточно CSV-файлов в датасете.")
        return None

    file1_path = os.path.join(folder_path, csv_files[0])
    file2_path = os.path.join(folder_path, csv_files[1])

    features_df = pd.read_csv(file1_path)
    targets_df = pd.read_csv(file2_path)
    df = pd.merge(features_df, targets_df, on='Employee_ID', how='inner')

    return df


@st.cache_resource
def train_model(df, features_list):
    target_col = [col for col in df.columns if 'productivity' in col.lower()][0]

    X = df[features_list]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    return model, target_col, score


df = load_and_process_data()

if df is not None:
    model, target_name, model_score = train_model(df, MY_FEATURES)

    # Создаем две вкладки
    tab1, tab2 = st.tabs(["🔍 Поиск сотрудника", "📈 Анализ данных"])

    # =========================================================================
    # ВКЛАДКА 1: ПОИСК СОТРУДНИКА, РЕКОМЕНДАЦИИ И ПРОГНОЗ (Ваш новый запрос)
    # =========================================================================
    # =========================================================================
    # ВКЛАДКА 1: ПОИСК СОТРУДНИКА, РЕКОМЕНДАЦИИ И ПРОГНОЗ (Широкий красивый вывод)
    # =========================================================================
    with tab1:
        st.subheader("🔍 Поиск сотрудника по ID")

        available_ids = df['Employee_ID'].astype(str).tolist()
        search_id = st.selectbox("Выберите или введите ID сотрудника:", available_ids[:50])

        if search_id:
            employee_data = df[df['Employee_ID'].astype(str) == search_id]

            if not employee_data.empty:
                emp = employee_data.iloc[0]
                current_productivity = emp[target_name]

                st.write("---")

                # 1. КАРТОЧКИ МЕТРИК (Верхний ряд)
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="📊 Текущая продуктивность", value=f"{current_productivity:.2f}")
                with col2:
                    st.metric(label="🤖 Процент автоматизации (ИИ)", value=f"{emp['tasks_automated_percent']}%")
                with col3:
                    st.metric(label="⚠️ Уровень ошибок", value=f"{emp['error_rate_percent']}%")

                st.write("---")

                # 2. РАЗДЕЛ РЕКОМЕНДАЦИЙ (На всю ширину экрана)
                st.subheader("💡 Персональный анализ и рекомендации")

                has_issues = False

                # Проверка 1: Защита отличников
                if current_productivity > 85:
                    st.success(
                        "**🟢 Идеальный баланс параметров**\n\n"
                        "У сотрудника выдающиеся показатели! Корректировки не требуются. "
                        "Текущий рабочий процесс выстроен максимально эффективно, сотрудник отлично совмещает задачи."
                    )
                else:
                    # Проверка 2: Слабая автоматизация
                    if emp['tasks_automated_percent'] < 20:
                        st.error(
                            "**🔴 Слабая автоматизация процессов**\n\n"
                            f"Сотрудник автоматизировал всего {emp['tasks_automated_percent']}% задач. "
                            "Рекомендуется провести аудит рутинных операций и передать ИИ-инструментам часть процессов "
                            "для снижения механической нагрузки."
                        )
                        has_issues = True

                    # Проверка 3: Высокий процент ошибок
                    if emp['error_rate_percent'] > 8:
                        st.shape = st.warning(
                            "**🟡 Высокий уровень ошибок**\n\n"
                            f"Процент брака/ошибок составляет {emp['error_rate_percent']}%, что выше нормы. "
                            "Рекомендуется временно снизить темп работы, проверить фокусные часы или "
                            "организовать дополнительную проверку результатов."
                        )
                        has_issues = True

                    # Проверка 4: Плохой Work-Life Balance
                    if emp['work_life_balance_score'] < 5:
                        st.info(
                            "**🔵 Риск выгорания (Низкий Work-Life Balance)**\n\n"
                            f"Оценка баланса работы и личной жизни критически низкая: {emp['work_life_balance_score']}/10. "
                            "Необходимо сократить количество переработок и оптимизировать нагрузку, иначе продуктивность продолжит падать."
                        )
                        has_issues = True

                    # Проверка 5: Скрытая низкая продуктивность
                    if not has_issues and current_productivity < 70:
                        st.warning(
                            "**⚠️ Скрытый спад эффективности**\n\n"
                            f"Формальные параметры в норме, "
                            f"очевидных перекосов нет, но общая продуктивность низкая ({current_productivity:.2f}).\n\n"
                            "**Рекомендуемые стратегические действия:**\n"
                            "— *Нехватка опыта/навыков:* Закрепите за сотрудником ментора или отправьте на профильное обучение.\n"
                            "— *Проблема с мотивацией:* Обсудите смену пула задач или пересмотрите KPI, текущая работа может казаться рутинной.\n"
                            "— *Внешние барьеры:* Возможно, работе мешают долгие согласования или размытая постановка задач руководством."
                        )
                        has_issues = True

                    # Проверка 6: Проблем нет, продуктивность средняя/хорошая
                    if not has_issues:
                        st.success(
                            "**🟢 Стабильные показатели**\n\n"
                            "Явных критических отклонений в параметрах сотрудника не обнаружено. "
                            "Продуктивность находится на стабильном среднем уровне."
                        )

                # 3. РАЗДЕЛ ДИНАМИЧЕСКОГО ПРОГНОЗА
                if current_productivity <= 85:
                    st.write("---")
                    st.subheader("🔮 Симуляция изменений и прогноз на будущее")
                    st.write("Что произойдет с продуктивностью, если точечно исправить выявленные проблемы?")

                    future_data = pd.DataFrame([emp[MY_FEATURES].to_dict()])
                    applied_improvements = []

                    # Моделируем точечные изменения
                    if emp['tasks_automated_percent'] < 20:
                        future_data.loc[0, 'tasks_automated_percent'] = 45
                        applied_improvements.append("Повышение автоматизации задач с уровня базы до 45%")

                    if emp['error_rate_percent'] > 8:
                        future_data.loc[0, 'error_rate_percent'] = 4
                        applied_improvements.append("Снижение уровня ошибок с критического до нормативных 4%")

                    if emp['work_life_balance_score'] < 5:
                        future_data.loc[0, 'work_life_balance_score'] = 7
                        applied_improvements.append("Восстановление баланса работы и личной жизни до 7/10")

                    # Моделируем комплексные изменения для скрытых причин
                    if not applied_improvements and current_productivity < 70:
                        future_data.loc[0, 'tasks_automated_percent'] = min(
                            future_data.loc[0, 'tasks_automated_percent'] + 15, 100)
                        future_data.loc[0, 'work_life_balance_score'] = min(
                            future_data.loc[0, 'work_life_balance_score'] + 1.5, 10)
                        future_data.loc[0, 'error_rate_percent'] = max(future_data.loc[0, 'error_rate_percent'] - 2.5,
                                                                       0)
                        future_data.loc[0, 'focus_hours_per_day'] = min(future_data.loc[0, 'focus_hours_per_day'] + 1.0,
                                                                        8)
                        applied_improvements.append(
                            "Комплексное оздоровление среды (оптимизация процессов, наставничество, снижение стресса)")

                    # Считаем предсказание
                    future_prediction = model.predict(future_data)[0]
                    diff = future_prediction - current_productivity

                    if applied_improvements:
                        # Создаем красивую рамку для отчета о прогнозе
                        with st.expander("🛠️ Посмотреть симулируемые изменения в профиле", expanded=True):
                            for imp_text in applied_improvements:
                                st.markdown(f"* {imp_text}")

                        if diff > 0:
                            st.info(
                                f"📈 **Результат симуляции:** Устранение проблем поднимет продуктивность сотрудника до **{future_prediction:.2f}** (Ожидаемый прирост: **+{diff:.2f}**).")
                        else:
                            st.info(
                                f"📊 **Результат симуляции:** Корректировка параметров стабилизирует рабочие процессы сотрудника на уровне **{future_prediction:.2f}**.")
            else:
                st.error("❌ Сотрудник с таким ID не найден в базе данных.")

    # =========================================================================
    # ВКЛАДКА 2: СТРАТЕГИЧЕСКИЙ АНАЛИЗ И КЛАСТЕРИЗАЦИЯ КОМАНДЫ (Обновлено)
    # =========================================================================
    with tab2:
        st.subheader("🎯 Сегментация сотрудников по ролям и профилям эффективности")
        st.write(
            "На основе многомерного кластерного анализа (KMeans, k=4) и снижения размерности UMAP, "
            "каждая профессия в компании была разделена на 4 отчетливых архетипа сотрудников. "
            "Выберите интересующую вас роль для просмотра профилей и рекомендаций:"
        )

        # Выбор роли/профессии сотрудника
        selected_role = st.selectbox(
            "Выберите профессиональную роль для анализа:",
            ['Analyst', 'Designer', 'Developer', 'Manager', 'Marketer', 'Writer']
        )

        st.write("---")
        st.markdown(f"### 📋 Профили кластеров для роли: **{selected_role}**")

        # Интерактивные вкладки под каждый из 4-х кластеров, полученных в ходе вашего анализа
        c_tab1, c_tab2, c_tab3, c_tab4 = st.tabs([
            "🤖 Кластер 0: ИИ-Оптимизаторы",
            "⚠️ Кластер 1: Группа риска (Выгорание)",
            "⚙️ Кластер 2: Традиционные Исполнители",
            "📚 Кластер 3: Начинающие / Стремящиеся"
        ])

        # Универсальные стратегические рекомендации, основанные на метриках профилирования вашего скрипта
        with c_tab1:
            st.success("#### 🚀 Характеристика: Высокая продуктивность + Максимальный AI Ratio")
            st.markdown(
                "**Описание профиля:** Сотрудники с высоким `tasks_automated_percent` и низким уровнем ошибок (`error_rate_percent`). "
                "Они эффективно делегируют рутину алгоритмам, высвобождая `focus_hours_per_day` для сложных задач.\n\n"
                "**🎯 Стратегические рекомендации для руководителя:**\n"
                "1. **Амбассадоры технологий:** Сделайте сотрудников этой группы внутренними менторами. Пусть они проведут мастер-классы для других кластеров по интеграции ИИ в рабочий процесс.\n"
                "2. **Делегирование R&D:** Направляйте их на самые сложные и нестандартные задачи (`task_complexity_score`), так как базовая рутина у них полностью автоматизирована.\n"
                "3. **Удержание:** Это ключевой цифровой капитал компании. Убедитесь, что их мотивация и карьерный трек соответствуют их высокой ценности."
            )

        with c_tab2:
            st.error("#### 🚨 Характеристика: Высокий Burnout Score + Перегрузка")
            st.markdown(
                "**Описание профиля:** Критически низкий `work_life_balance_score` и аномально высокие часы совещаний (`meeting_hours_per_week`). "
                "Продуктивность начинает падать, а `error_rate_percent` растет из-за хронической усталости.\n\n"
                "**🎯 Стратегические рекомендации для руководителя:**\n"
                "1. **Разгрузка расписания (Meeting Detox):** Введите жесткий лимит на созвоны. Переведите часть коммуникаций в асинхронный текстовый формат.\n"
                "2. **Принудительный Work-Life Balance:** Проверьте их переработки. На уровне менеджмента ограничьте отправку рабочих задач во внеурочное время.\n"
                "3. **Психологическая поддержка:** Проведите личную встречу 1-on-1, чтобы выявить триггеры стресса до того, как сотрудник примет решение об увольнении."
            )

        with c_tab3:
            st.warning("#### 🦾 Характеристика: Высокий Manual Work + Средняя эффективность")
            st.markdown(
                "**Описание профиля:** Сотрудники много работают руками (`manual_work_hours_per_week`), имеют хороший фокус, "
                "но игнорируют автоматизацию (`ai_ratio` близок к нулю). Выполняют задачи стабильно, но упираются в «потолок» своего времени.\n\n"
                "**🎯 Стратегические рекомендации для руководителя:**\n"
                "1. **Мягкое внедрение ИИ:** Выделите им готовые корпоративные промпты и шаблоны автоматизации, чтобы снизить страх перед новыми инструментами.\n"
                "2. **Снижение механического труда:** Поставьте задачу автоматизировать хотя бы 15-20% рутины за следующий месяц. Освободившееся время направить на обучение.\n"
                "3. **Оптимизация процессов:** Проверьте, почему они делают работу вручную — возможно, им просто не предоставили нужные доступы или лицензии к ИИ-инструментам."
            )

        with c_tab4:
            st.info("#### 🎯 Характеристика: Высокий Learning Time + Потенциал роста")
            st.markdown(
                "**Описание профиля:** Метрика `learning_time_hours_per_week` выше средней по компании. Обладают умеренным или небольшим стажем "
                "(`experience_years`), активно впитывают новые практики, но продуктивность пока находится в стадии роста.\n\n"
                "**🎯 Стратегические рекомендации для руководителя:**\n"
                "1. **Наставничество:** Закрепите за ними экспертов из Кластера 0. Это ускорит их переход от теории к высокоэффективной практике.\n"
                "2. **Контроль фокуса:** Следите, чтобы обучение не превращалось в прокрастинацию и не вредило текущим дедлайнам (`deadline_pressure_level`).\n"
                "3. **Инвестиции в развитие:** Поддерживайте их стремление к учебе, предоставляя профильные курсы. Через 2-3 месяца эта группа способна стать ядром продуктивности."
            )

        # Контекстные подсказки в зависимости от выбранной роли
        st.write("---")
        st.subheader(f"💡 Специфика роли {selected_role} в разрезе кластеризации")
        if selected_role in ['Developer', 'Analyst']:
            st.info(
                f"**Для {selected_role} ключевыми драйверами разделения на кластеры** чаще всего выступают `task_complexity_score` и `error_rate_percent`. "
                "Именно в этих ролях автоматизация рутины дает максимальный скачок в качестве кода и аналитических моделей."
            )
        elif selected_role in ['Designer', 'Writer']:
            st.info(
                f"**Для {selected_role} критически важен баланс** между `focus_hours_per_day` и совещаниями. "
                "Творческие профили в Кластере 1 (Выгорание) теряют продуктивность быстрее остальных ролей, если их лишают непрерывного фокусного времени."
            )
        else:
            st.info(
                f"**Для {selected_role} основное разделение** идет по линии коммуникаций (`collaboration_hours_per_week`) и `tasks_automated_percent`. "
                "Автоматизация отчетности и планирования в этих группах кардинально снижает риск выгорания."
            )

