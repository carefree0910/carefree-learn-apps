import streamlit as st

from typing import Callable


class MultiPage:
    def __init__(self):
        self.pages = []

    def add_page(self, title: str, func: Callable[[], None]) -> None:
        self.pages.append({"title": title, "func": func})

    def run(self) -> None:
        page = st.sidebar.selectbox(
            "App Navigation",
            self.pages,
            format_func=lambda p: p["title"],
        )
        page["func"]()
