from agent import CustomerAgent
import uuid

def stream_response(agent, user_input, config, user_name=None):
    parsed = agent.parser.parse(user_input)
    context_query = agent._add_context(user_input, parsed)
    
    if user_name:
        context_query = f"Customer name: {user_name}\n{context_query}"
    
    for event in agent.agent.stream(
        {"messages": [{"role": "user", "content": context_query}]},
        stream_mode="values",
        config=config,
    ):
        if 'messages' in event:
            last_msg = event["messages"][-1]
            if hasattr(last_msg, 'content') and last_msg.content:
                if hasattr(last_msg, 'type') and last_msg.type == 'ai':
                    print("Assistant:", last_msg.content)
                elif not hasattr(last_msg, 'type') and last_msg.content:
                    print("Assistant:", last_msg.content)

def get_user_name():
    """Get user's name at the start of conversation"""
    print("Assistant: Merhaba! SBM müşteri hizmetlerine hoş geldiniz.")
    while True:
        name = input("Assistant: Öncelikle adınızı öğrenebilir miyim? ").strip()
        if name:
            return name
        print("Assistant: Lütfen adınızı girin.")

def main():
    agent = CustomerAgent()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    user_name = get_user_name()
    
    print(f"Assistant: Merhaba {user_name}! Size nasıl yardımcı olabilirim?")
    
    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q", "çıkış"]:
                print(f"Assistant: İyi günler {user_name}!")
                break
            stream_response(agent, user_input, config, user_name)
        except KeyboardInterrupt:
            print(f"\nAssistant: İyi günler {user_name}!")
            break
        except:
            print("Assistant: Bir hata oluştu, tekrar deneyin.")

if __name__ == "__main__":
    main()