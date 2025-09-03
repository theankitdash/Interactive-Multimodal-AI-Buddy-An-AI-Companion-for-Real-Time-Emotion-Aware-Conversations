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
        padding: "32dp"
        spacing: "20dp"

        # Camera preview
        Widget:
            size_hint_y: 0.6
            canvas.before:
                Color:
                    rgba: (0.15,0.15,0.18,1)
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [20,]

        # Username input (only in registration)
        TextInput:
            id: username_input
            hint_text: "Username"
            size_hint_y: None
            height: "44dp"
            background_normal: ""
            background_color: (0.12,0.12,0.15,1)
            foreground_color: (1,1,1,1)
            padding: ["12dp","10dp"]
            radius: [12,]
            opacity: 1 if root.mode=='register' else 0
            disabled: False if root.mode=='register' else True

        # Full Name input (only in registration)
        TextInput:
            id: fullname_input
            hint_text: "Full Name"
            size_hint_y: None
            height: "44dp"
            background_normal: ""
            background_color: (0.12,0.12,0.15,1)
            foreground_color: (1,1,1,1)
            padding: ["12dp","10dp"]
            radius: [12,]
            opacity: 1 if root.mode=='register' else 0
            disabled: False if root.mode=='register' else True

        # Action button
        Button:
            id: action_button
            text: root.action_text
            size_hint_y: None
            height: "48dp"
            background_normal: ""
            background_color: (0.25,0.45,1,1)
            color: (1,1,1,1)
            font_size: "16sp"
            bold: True
            radius: [14,]
            on_release: app.root.current = "main"

        # Toggle Login/Register
        Button:
            text: root.switch_text
            size_hint_y: None
            height: "44dp"
            background_normal: ""
            background_color: (0.18,0.18,0.22,1)
            color: (0.8,0.8,0.85,1)
            font_size: "14sp"
            radius: [12,]
            on_release: root.toggle_mode()

<MainScreen>:
    name: "main"
    BoxLayout:
        orientation: "horizontal"
        spacing: "20dp"
        padding: "24dp"

        # Left side - events with modern scrollbar
        ScrollView:
            id: scroll_events
            bar_width: 6
            scroll_type: ['bars', 'content']
            effect_cls: "ScrollEffect"
            bar_color: (0.3,0.5,1,0.8)  # soft blue scrollbar
            bar_inactive_color: (0.3,0.5,1,0.3)  # faded when not moving
            canvas.before:
                Color:
                    rgba: (0,0,0,0)
                Rectangle:
                    pos: self.pos
                    size: self.size

            RecycleView:
                id: events
                viewclass: "EventRow"
                bar_width: 0
                RecycleBoxLayout:
                    default_size: None, dp(54)
                    default_size_hint: 1, None
                    size_hint_y: None
                    height: self.minimum_height
                    orientation: "vertical"
                    spacing: "8dp"

        # Right side - camera
        Widget:
            id: camera_widget
            size_hint_y: None
            height: self.width
            canvas.before:
                Color:
                    rgba: (0.15, 0.15, 0.18, 1)
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [20,]

<EventRow@BoxLayout>:
    size_hint_y: None
    height: "54dp"
    padding: "10dp"
    spacing: "16dp"
    canvas.before:
        Color:
            rgba: (0.12, 0.12, 0.15, 1)
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [14,]
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
