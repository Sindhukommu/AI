import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Predictive Maintenance AI", layout="wide")
st.title("🔧 Predictive Maintenance AI")
st.markdown("---")

# Get API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    try:
        from config import GROQ_API_KEY
    except:
        pass

# Initialize client
if GROQ_API_KEY:
    client = Groq(api_key=GROQ_API_KEY)
else:
    client = None
    st.error("⚠️ API Key not found. Create .env file with GROQ_API_KEY=your_key")

# Tabs
tabs = st.tabs(["📊 Input", "🤖 Predictions", "📈 Analytics", "🤔 AI Insights", "📋 Report"])

# Sample data
if 'df' not in st.session_state:
    np.random.seed(42)
    st.session_state.df = pd.DataFrame({
        'Machine_ID': [f'M{i:03d}' for i in range(1, 101)],
        'Temperature': np.random.normal(75, 15, 100).round(1),
        'Vibration': np.random.normal(0.5, 0.2, 100).round(3),
        'Pressure': np.random.normal(100, 20, 100).round(1),
        'RPM': np.random.normal(1500, 300, 100).round(0).astype(int),
        'Hourly_Output': np.random.normal(450, 50, 100).round(0).astype(int),
        'Maintenance_History': np.random.choice([0, 1, 2, 3], 100),
        'Failure_Status': np.random.choice([0, 1], 100, p=[0.9, 0.1])
    })

# Tab 1
with tabs[0]:
    st.header("Upload Sensor Data")
    uploaded = st.file_uploader("Choose file", type=['xlsx', 'xls', 'csv'])
    if uploaded:
        try:
            st.session_state.df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
            st.success("✅ File uploaded!")
        except Exception as e:
            st.error(f"Error: {e}")
    st.dataframe(st.session_state.df.head(10), use_container_width=True)

# Tab 2
with tabs[1]:
    st.header("Failure Predictions")
    df = st.session_state.df.copy()
    features = ['Temperature', 'Vibration', 'Pressure', 'RPM', 'Hourly_Output', 'Maintenance_History']
    features = [f for f in features if f in df.columns]
    
    if features:
        X = df[features].fillna(df[features].median())
        y = df['Failure_Status'] if 'Failure_Status' in df.columns else np.random.choice([0, 1], len(df), p=[0.9, 0.1])
        
        model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
        model.fit(X, y)
        
        df['Failure_Probability'] = model.predict_proba(X)[:, 1]
        df['Risk_Level'] = pd.cut(df['Failure_Probability'], bins=[0, 0.3, 0.7, 1], labels=['Low', 'Medium', 'High'])
        df['Days_To_Failure'] = (1 - df['Failure_Probability']) * 30
        df['Recommended'] = df['Failure_Probability'].apply(lambda x: '🚨 Immediate' if x > 0.7 else '📅 Within Week' if x > 0.3 else '✅ Monthly')
        
        st.session_state.results = df
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("🔴 High Risk", len(df[df.Risk_Level=='High']))
        c2.metric("🟡 Medium Risk", len(df[df.Risk_Level=='Medium']))
        c3.metric("🟢 Low Risk", len(df[df.Risk_Level=='Low']))
        c4.metric("💰 Savings", f"${len(df[df.Risk_Level=='High'])*1500:,}")
        
        st.dataframe(df[['Machine_ID', 'Failure_Probability', 'Risk_Level', 'Days_To_Failure', 'Recommended']].head(15))

# Tab 3
with tabs[2]:
    if 'results' in st.session_state:
        df = st.session_state.results
        col1, col2 = st.columns(2)
        with col1:
            counts = df['Risk_Level'].value_counts()
            fig = px.pie(values=counts.values, names=counts.index, title="Risk Distribution",
                        color_discrete_map={'Low':'green', 'Medium':'orange', 'High':'red'})
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(counts.reset_index(), x='Risk_Level', y='count', title="Risk Count",
                        color='Risk_Level', color_discrete_map={'Low':'green', 'Medium':'orange', 'High':'red'})
            st.plotly_chart(fig, use_container_width=True)
        col3, col4 = st.columns(2)
        with col3:
            fig = px.line(df.sort_values('Failure_Probability'), y='Failure_Probability', title="Failure Trend")
            st.plotly_chart(fig, use_container_width=True)
        with col4:
            fig = go.Figure(go.Indicator(mode="gauge+number", value=df['Failure_Probability'].mean()*100,
                title={'text': "Overall Risk"}, gauge={'axis': {'range': [0, 100]},
                'steps': [{'range': [0,30], 'color':'lightgreen'},{'range':[30,70],'color':'yellow'},{'range':[70,100],'color':'red'}]}))
            st.plotly_chart(fig, use_container_width=True)

# Tab 4
with tabs[3]:
    if 'results' in st.session_state and client:
        df = st.session_state.results
        analysis = st.selectbox("Analysis Type", ["System Health", "High-Risk Analysis", "Cost Strategy", "Schedule"])
        
        if st.button("Generate AI Insights"):
            with st.spinner("AI analyzing..."):
                try:
                    prompts = {
                        "System Health": f"Analyze {len(df)} machines: High:{len(df[df.Risk_Level=='High'])}, Medium:{len(df[df.Risk_Level=='Medium'])}, Low:{len(df[df.Risk_Level=='Low'])}. Avg temp:{df['Temperature'].mean():.1f}°F. Top risks:{df.nlargest(3,'Failure_Probability')['Machine_ID'].tolist()}. Provide recommendations.",
                        "High-Risk Analysis": f"Analyze high-risk machines:\n{df[df.Risk_Level=='High'][['Machine_ID','Temperature','Vibration','Failure_Probability']].to_string()}\nExplain risks and interventions.",
                        "Cost Strategy": f"High-risk:{len(df[df.Risk_Level=='High'])} (savings:${len(df[df.Risk_Level=='High'])*1500}), Medium:{len(df[df.Risk_Level=='Medium'])}. Provide cost-optimized strategy.",
                        "Schedule": f"Create 7-day schedule: Immediate:{len(df[df.Risk_Level=='High'])}, Weekly:{len(df[df.Risk_Level=='Medium'])}. Assume 3 teams."
                    }
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": "You are a predictive maintenance expert."},
                                 {"role": "user", "content": prompts[analysis]}],
                        temperature=0.7, max_tokens=500)
                    st.info(completion.choices[0].message.content)
                    st.session_state.ai_insights = completion.choices[0].message.content
                except Exception as e:
                    st.error(f"API Error: {e}")
    elif not client:
        st.warning("⚠️ Groq API not configured. Add API key to use AI insights.")

# Tab 5
with tabs[4]:
    if 'results' in st.session_state:
        df = st.session_state.results
        st.subheader("Executive Summary")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Machines", len(df))
        c2.metric("High Risk", len(df[df.Risk_Level=='High']))
        c3.metric("Savings", f"${len(df[df.Risk_Level=='High'])*1500:,}")
        
        schedule = df[df.Risk_Level!='Low'].sort_values('Failure_Probability', ascending=False)
        if len(schedule) > 0:
            schedule['Maintenance_Date'] = datetime.now() + pd.to_timedelta(schedule['Days_To_Failure'], unit='D')
            st.dataframe(schedule[['Machine_ID', 'Risk_Level', 'Failure_Probability', 'Maintenance_Date', 'Recommended']].head(20))
        
        if 'ai_insights' in st.session_state:
            st.subheader("AI Recommendations")
            st.info(st.session_state.ai_insights)
        
        st.download_button("📥 Download Report", data=df.to_csv(index=False).encode('utf-8'),
                          file_name=f"report_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")