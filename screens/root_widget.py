from kivy.uix.screenmanager import ScreenManager
from screens.auth_screen import AuthScreen
from screens.main_screen import MainScreen

class RootWidget(ScreenManager):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_widget(AuthScreen(name="auth"))
        self.add_widget(MainScreen(name="main"))