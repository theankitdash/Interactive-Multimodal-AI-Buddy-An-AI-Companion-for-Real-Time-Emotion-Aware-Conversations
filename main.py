from kivy.app import App
from kivy.lang import Builder
from screens.root_widget import RootWidget

class MultiModalBuddy(App):
    def build(self):
        Builder.load_file("ui/kv_layout.kv")
        return RootWidget()

if __name__ == "__main__":
    MultiModalBuddy().run()
