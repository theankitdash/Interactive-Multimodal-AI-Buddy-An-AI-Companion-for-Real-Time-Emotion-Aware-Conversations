from kivy.uix.screenmanager import Screen

class MainScreen(Screen):
    def talk_with_agent(self, message):
        if not message.strip():
            return
        print("🤖 Agent received:", message)
        # Here you can call your LLM / agent logic
        # Example: store in DB under user_knowledge