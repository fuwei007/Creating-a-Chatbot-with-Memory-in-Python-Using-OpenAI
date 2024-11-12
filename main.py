import json
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Memory file path
memory_file_path = 'agent_memory.json'

# Load or initialize agent memory
if os.path.exists(memory_file_path):
    with open(memory_file_path, 'r') as file:
        agent_memory = json.load(file)
else:
    agent_memory = {"human": "", "agent": ""}

model = "gpt-4o-mini"

# Function to save memory into the JSON file
def core_memory_save(section: str, memory: str):
    agent_memory[section] += '\n' + memory
    with open(memory_file_path, 'w') as file:
        json.dump(agent_memory, file, indent=4)

# Tool description
core_memory_save_description = "Save important information about you," \
+ " the agent or the human you are chatting with."

# Arguments into the tool
core_memory_save_properties = {
    "section": {
        "type": "string",
        "enum": ["human", "agent"],
        "description": "Must be either 'human' (to save information about the human) or 'agent' (to save information about yourself)",
    },
    "memory": {
        "type": "string",
        "description": "Memory to save in the section",
    },
}

# Tool schema (passed to OpenAI)
core_memory_save_metadata = {
    "type": "function",
    "function": {
        "name": "core_memory_save",
        "description": core_memory_save_description,
        "parameters": {
            "type": "object",
            "properties": core_memory_save_properties,
            "required": ["section", "memory"],
        },
    }
}

system_prompt = "You are a chatbot. You have a section of your context called [MEMORY] that contains information relevant to your conversation."

system_prompt_os = system_prompt + "\nYou must either call a tool (core_memory_save) or write a response to the user. Do not take the same actions multiple times! When you learn new information, make sure to always call the core_memory_save tool."

def agent_step(user_message):
    # Prefix messages with system prompt and memory
    messages = [
        {"role": "system", "content": system_prompt_os},
        {"role": "system", "content": "[MEMORY]\n" + json.dumps(agent_memory)},
    ]

    # Append the most recent message
    messages.append({"role": "user", "content": user_message})

    # Agentic loop
    while True:
        chat_completion = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[core_memory_save_metadata]
        )
        response = chat_completion.choices[0]

        # Update the messages with the agent's response
        messages.append(response.message)

        # If not calling a tool, return the response
        if not response.message.tool_calls:
            return response.message.content

        # If calling a tool, execute the tool
        else:
            print("TOOL CALL:", response.message.tool_calls[0].function)

            # Parse the arguments from the LLM function call
            arguments = json.loads(response.message.tool_calls[0].function.arguments)

            # Run the function with the specified arguments
            core_memory_save(**arguments)

            # Add the tool call response to the message history
            messages.append({
                "role": "tool",
                "tool_call_id": response.message.tool_calls[0].id,
                "name": "core_memory_save",
                "content": f"Updated memory: {json.dumps(agent_memory)}"
            })

if __name__ == "__main__":
    # user_input = input("Enter your message: ")
    print('Start')
    while True:
        user_input = input("")
        response = agent_step(user_input)
        print("Agent Response:", response)