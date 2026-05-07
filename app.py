import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="AI 极速预测 - TabPFN Demo", layout="wide")
st.title("⚡ AI 极速表格预测（TabPFN）")
st.caption("上传一份表格数据，选择你要预测的列，AI 几秒内给你结果。")

# --- 文件上传 ---
uploaded_file = st.file_uploader("上传 CSV 或 Excel 文件", type=["csv", "xlsx"])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"文件读取失败：{e}")
        st.stop()

    st.subheader("📊 数据预览")
    st.dataframe(df.head(10), use_container_width=True)
    st.write(f"共 {df.shape[0]} 行，{df.shape[1]} 列")

    # --- 选择目标列 ---
    target_col = st.selectbox("🎯 选择你要预测的列（必须为分类标签）", df.columns)

    if target_col:
        # 检查目标列是否为分类
        if df[target_col].dtype == 'object' or df[target_col].nunique() < 20:
            # 处理缺失值（简单演示版，直接删除）
            df = df.dropna(subset=[target_col])
            y = df[target_col].astype(str)
            X = df.drop(columns=[target_col])

            # 自动识别并编码非数值特征
            for col in X.columns:
                if X[col].dtype == 'object':
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))

            # 处理目标列的标签编码（TabPFN 需要 0~C-1 的整数）
            le_target = LabelEncoder()
            y_encoded = le_target.fit_transform(y)

            # 分割数据（简单的分层抽样）
            if st.button("🚀 开始训练并预测"):
                if len(X) < 5:
                    st.warning("数据行数太少，至少需要 5 行以上才能训练。")
                else:
                    with st.spinner("AI 正在极速训练中，请稍等几秒……"):
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
                            )
                        except ValueError:
                            # 分层失败则随机划分
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y_encoded, test_size=0.2, random_state=42
                            )

                        # 初始化 TabPFN 分类器（CPU 版，小数据下很快）
                        classifier = TabPFNClassifier(device='cpu')

                        # 训练
                        classifier.fit(X_train, y_train)

                        # 预测测试集
                        y_pred = classifier.predict(X_test)
                        y_pred_labels = le_target.inverse_transform(y_pred)

                        # 显示结果
                        st.success("✅ 预测完成！")
                        st.subheader("📋 测试集预测结果（前20条）")
                        result_df = X_test.copy()
                        result_df["真实标签"] = le_target.inverse_transform(y_test)
                        result_df["预测标签"] = y_pred_labels
                        result_df["是否正确"] = result_df["真实标签"] == result_df["预测标签"]
                        st.dataframe(result_df.head(20), use_container_width=True)

                        # 计算准确率
                        accuracy = (y_pred == y_test).mean()
                        st.metric("🎯 模型准确率（测试集）", f"{accuracy:.2%}")

                        # 提供下载
                        csv = result_df.to_csv(index=False).encode('utf-8-sig')
                        st.download_button(
                            label="📥 下载完整预测结果 (CSV)",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv",
                        )
        else:
            st.warning("你选择的列看起来不是分类标签（唯一值太多），请选择另一个列。")
