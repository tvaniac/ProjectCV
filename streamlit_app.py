import streamlit as st 

st.set_page_config(layout="wide")

# Page Setup
about_me_page = st.Page(
    page = "views/about_me.py",
    title = "About Me",
    icon = "🥷",
    default = True
)

learn_pandas_page = st.Page(
    page = "views/learn_pandas.py",
    title = "Learn Pandas",
    icon = "🐼"
)

learn_plotly_page = st.Page(
    page = "views/learn_plotly.py",
    title = "Learn Plotly",
    icon = "📈"
)

dashboard_sales_page = st.Page(
    page = "views/dashboard_one.py",
    title = "Dashboard Sales",
    icon = "🖥️"
)

# Navigation
pg = st.navigation(
    pages = [about_me_page, learn_pandas_page, learn_plotly_page, dashboard_one_page],
)

st.logo("assets/ninja.png")

pg.run()