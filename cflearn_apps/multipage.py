import streamlit as st

from typing import Callable


class MultiPage:
    def __init__(self):
        self.pages = []

    def add_page(self, title: str, func: Callable[[], None]) -> None:
        self.pages.append({"title": title, "func": func})

    def run(self) -> None:
        st.markdown("---")
        page = st.sidebar.selectbox(
            "",
            self.pages,
            format_func=lambda p: p["title"],
        )
        st.sidebar.markdown("---")
        st.title(page["title"])
        page["func"]()
