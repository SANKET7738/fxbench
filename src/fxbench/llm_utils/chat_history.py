from __future__ import annotations


class ChatHistory:
    def __init__(self, max_history_size: int = 1000):
        self.max_history_size = max_history_size
        self.messages = []

    def add_message(self, role: str, message: str) -> None:
        self.messages.append(
            {
                "role": role,
                "content": [
                    {
                        "type": "text",
                        "text": message,
                    }
                ],
            }
        )
        if len(self.messages) > self.max_history_size:
            self.messages = self.messages[-self.max_history_size :]

    def get_message_history(self) -> list[dict]:
        return self.messages

    def reset_chat_history(self) -> None:
        self.messages = []
