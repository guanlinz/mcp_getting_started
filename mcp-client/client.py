import asyncio
import logging
from typing import Optional
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from openai import OpenAI
from dotenv import load_dotenv, dotenv_values
import json
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

config = dotenv_values()  # load environment variables from .env
MODEL="google/gemini-2.0-flash-001"
# MODEL="openai/gpt-4o"
# MODEL="anthropic/claude-3-7-sonnet"

logging.info(f"Loaded config: {config}")

def convert_tool_format(tool):
    converted_tool = {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": tool.inputSchema["properties"],
                "required": tool.inputSchema["required"]
            }
        }
    }
    return converted_tool

class MCPClient:
    def __init__(self):
        # Initialize session and client objects
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key= config['OPENROUTER_API_KEY']
        )
    # methods will go here


    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py or .js)
        """
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None
        )

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        logging.info(f"Connected to server with tools: {[tool.name for tool in tools]}")
        self.messages = []

    async def process_query(self, query: str) -> str:
        self.messages.append({
            "role": "user",
            "content": query
        })
        response = await self.session.list_tools()
        available_tools = [convert_tool_format(tool) for tool in response.tools]
        response = self.openai.chat.completions.create(
            model=MODEL,
            tools=available_tools,
            messages=self.messages
        )
        logging.info(f"OpenAI response (initial): {response}")
        self.messages.append(response.choices[0].message.model_dump())
        final_text = []
        content = response.choices[0].message
        logging.info(f"Message content: {content}")
        if content.tool_calls is not None:
            tool_name = content.tool_calls[0].function.name
            tool_args = content.tool_calls[0].function.arguments
            tool_args = json.loads(tool_args) if tool_args else {}
            logging.info(f"content tool id: {content.tool_calls[0].id}")
            # Execute tool call
            try:
                result = await self.session.call_tool(tool_name, tool_args)
                final_text.append(f"[Calling tool {tool_name} with args {tool_args}]")
            except Exception as e:
                logging.error(f"Error calling tool {tool_name}:", exc_info=True)
                result = None
            
            self.messages.append({
                "role": "tool",
                "tool_call_id": content.tool_calls[0].id,
                "name": tool_name,
                # "content": result.content[0].text
                "content": result.content[0].model_dump_json()
            })
            logging.info(f"self.message: {self.messages}")

            response = self.openai.chat.completions.create(
                model=MODEL,
                messages=self.messages,
                tools=available_tools
            )
            logging.info(f"OpenAI response (after tool call): {response}")
            final_text.append(response.choices[0].message.content)
        else:
            final_text.append(content.content)
        return "\n".join(final_text)
    async def chat_loop(self):
        """Run an interactive chat loop"""
        logging.info("MCP Client Started!")
        logging.info("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == 'quit':
                    break

                response = await self.process_query(query)
                logging.info(f"\n{response}")

            except Exception as e:
                logging.error("Error occurred during chat loop:", exc_info=True)


    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        logging.error("Usage: python client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()

if __name__ == "__main__":
    import sys
    asyncio.run(main())
