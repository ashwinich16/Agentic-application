from typing import Optional

def get_followup_question(user_text: Optional[str] = "") -> str:


    if not user_text or not user_text.strip():
        return "Can you provide more details about what you want me to do?"

    return (
        "I’m not sure what you want me to do with this input. "
        "Could you clarify your intended task?"
    )
