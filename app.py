import json

from dotenv import load_dotenv
import chainlit as cl
from movie_functions import *

load_dotenv()

# Note: If switching to LangSmith, uncomment the following, and replace @observe with @traceable
# from langsmith.wrappers import wrap_openai
# from langsmith import traceable
# client = wrap_openai(openai.AsyncClient())

from langfuse.decorators import observe
from langfuse.openai import AsyncOpenAI

client = AsyncOpenAI()

gen_kwargs = {
    "model": "gpt-4o",
    "temperature": 0.2,
    "max_tokens": 500
}

SYSTEM_PROMPT = """\
You are an assistant that can respond to user queries, particularly regarding movies. Detect when the user requests a list of currently playing movies, showtimes, or related tasks. You will respond in two steps:

Step 1 (Function Call): When appropriate, respond with the function call in JSON format without executing it. The structure of your response should be:

{
  "function_name": "function_name_here",
  "parameters": [list_of_arguments]
}

This step indicates that you will fetch the required data.

Step 2 (Data Response): Once the function call has been processed, respond again with the actual result of the function execution.

Here are the functions you can call:

get_now_playing_movies(): When the user asks for a list of currently playing movies.
get_showtimes(title, location): When the user asks for showtimes of a specific movie.
buy_ticket(theater, movie, showtime): When the user requests to buy a ticket.
get_reviews(movie_id): When the user asks for reviews of a specific movie.

After receiving the results of the function call, use the updated system message history to incorporate that 
information into your response to the user.

If the request does not require a function call, respond naturally to the user.
"""


@observe
@cl.on_chat_start
def on_chat_start():
    message_history = [{"role": "system", "content": SYSTEM_PROMPT}]
    cl.user_session.set("message_history", message_history)


@observe
async def generate_response(client, message_history, gen_kwargs):
    response_message = cl.Message(content="")
    await response_message.send()

    stream = await client.chat.completions.create(messages=message_history, stream=True, **gen_kwargs)
    async for part in stream:
        if token := part.choices[0].delta.content or "":
            await response_message.stream_token(token)

    await response_message.update()

    return response_message


# Function to safely parse and invoke
def parse_and_invoke(json_string):
    try:
        # Parse the JSON string
        parsed = json.loads(json_string)

        # Extract the function name and parameters
        function_name = parsed.get("function_name")
        parameters = parsed.get("parameters", [])

        # Dynamically get the function by name
        func = globals().get(function_name)

        # Check if the function exists and invoke it with the parameters
        if func and callable(func):
            return func(*parameters)
        else:
            print(f"Function '{function_name}' not found or is not callable.")
            return None

    except (json.JSONDecodeError, TypeError, ValueError) as e:
        # Handle any parsing or invocation errors
        print(f"Error parsing or invoking function: {e}")
        return None


@cl.on_message
@observe
async def on_message(message: cl.Message):
    message_history = cl.user_session.get("message_history", [])
    message_history.append({"role": "user", "content": message.content})

    response_message = await generate_response(client, message_history, gen_kwargs)
    print(f"Response: {response_message.content}")

    content = response_message.content
    while True:
        result = parse_and_invoke(content)
        if result is not None:
            print(f"Result: {result}")
            message_history.append({"role": "system", "content": f"Result of a function call: {result}"})
            response_message = await generate_response(client, message_history, gen_kwargs)
            content = response_message.content
            print(f"Response: {response_message.content}")
        else:
            break

    message_history.append({"role": "assistant", "content": response_message.content})
    cl.user_session.set("message_history", message_history)


if __name__ == "__main__":
    cl.main()
