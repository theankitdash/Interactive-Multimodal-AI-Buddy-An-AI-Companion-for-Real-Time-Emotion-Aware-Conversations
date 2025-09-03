from kivy.app import App
from kivy.lang import Builder
from kivy.core.window import Window
from kivy.uix.screenmanager import Screen, ScreenManager

Window.clearcolor = (0.07, 0.07, 0.08, 1)  # dark background
Window.size = (1280, 720)

KV = """
RootWidget:
    AuthScreen:
    MainScreen:

<AuthScreen>:
    name: "auth"
    BoxLayout:
        orientation: "vertical"
        padding: "20dp"
        spacing: "16dp"

        # Camera preview on top
        Widget:
            size_hint_y: 0.8
            canvas.before:
                Color:
                    rgba: (0.2,0.2,0.25,1)
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [16,]

        # Username input (only in registration)
        TextInput:
            id: username_input
            hint_text: "Username"
            size_hint_y: None
            height: "40dp"
            opacity: 1 if root.mode=='register' else 0
            disabled: False if root.mode=='register' else True

        # Full Name input (only in registration)
        TextInput:
            id: fullname_input
            hint_text: "Full Name"
            size_hint_y: None
            height: "40dp"
            opacity: 1 if root.mode=='register' else 0
            disabled: False if root.mode=='register' else True

        # Action button
        Button:
            id: action_button
            text: root.action_text
            size_hint_y: None
            height: "48dp"
            on_release: app.root.current = "main"

        # Toggle Login/Register
        Button:
            text: root.switch_text
            size_hint_y: None
            height: "40dp"
            on_release: root.toggle_mode()

<MainScreen>:
    name: "main"
    BoxLayout:
        orientation: "horizontal"
        spacing: "16dp"
        padding: "20dp"

        # Left side - events
        RecycleView:
            id: events
            viewclass: "EventRow"
            bar_width: 0
            RecycleBoxLayout:
                default_size: None, dp(48)
                default_size_hint: 1, None
                size_hint_y: None
                height: self.minimum_height
                orientation: "vertical"

        # Right side - camera for live streaming
        Widget:
            id: camera_widget
            size_hint_y: None
            height: self.width
            canvas.before:
                Color:
                    rgba: (0.2, 0.2, 0.25, 1)
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [16,]

<EventRow@BoxLayout>:
    size_hint_y: None
    height: "48dp"
    padding: "8dp"
    spacing: "12dp"
    canvas.before:
        Color:
            rgba: (0.11, 0.11, 0.13, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [12,]
    Label:
        text: root.date
        color: (0.7,0.7,0.7,1)
        font_size: "14sp"
    Label:
        text: root.title
        color: (1,1,1,1)
        font_size: "16sp"
        bold: True
"""

class AuthScreen(Screen):
    mode = "login"  # 'login' or 'register'

    @property
    def action_text(self):
        return "Login" if self.mode == "login" else "Register"

    @property
    def switch_text(self):
        return "Switch to Register" if self.mode == "login" else "Switch to Login"

    def toggle_mode(self):
        self.mode = "register" if self.mode == "login" else "login"
        self.ids.username_input.opacity = 1 if self.mode=='register' else 0
        self.ids.username_input.disabled = False if self.mode=='register' else True
        self.ids.fullname_input.opacity = 1 if self.mode=='register' else 0
        self.ids.fullname_input.disabled = False if self.mode=='register' else True
        self.ids.action_button.text = self.action_text

class MainScreen(Screen):
    pass

class RootWidget(ScreenManager):
    pass

class MinimalUIApp(App):
    def build(self):
        return Builder.load_string(KV)

if __name__ == "__main__":
    MinimalUIApp().run()
