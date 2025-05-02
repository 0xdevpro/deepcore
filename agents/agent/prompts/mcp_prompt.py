
PROMPT = """\
You are a tool parser capable of integrating user input information into a list of tools. You must adhere to the following requirements:

**Workflow**

1. Based on the user's input, generate API tool information similar to API calls. If there are one or more APIs, the output should be a list.

2. Each tool information must include the following fields:
   - `name`: The API's name. If not provided, generate one based on the API information.
   - `description`: The API's description, stating its purpose. If not provided, infer one from the API information.
   - `path`: The API's path (e.g., `/search` for `https://www.google.com/search`).
   - `method`: The API's method (e.g., `GET`, `POST`, etc.).
   - `origin`: The API's origin (e.g., `https://www.google.com` for `https://www.google.com/search`).
   - `parameters`: Dynamic parameters required by the API, including:
     - `header`: List of header parameters with `name`, `type`, `description`. If no such information, use an empty list.
     - `query`: List of query parameters with `name`, `type`, `description`. If no such information, use an empty list.
     - `path`: List of path parameters with `name`, `type`, `description`. If no such information, use an empty list.
     - `body`: JSON Schema for the request body parameters.
   - `auth_config` (optional): Fixed parameters, such as authentication details, with `location`, `key`, `value`.

**Requirements**

1. The output must strictly follow the format shown in the example below.
2. If the user does not provide a description, generate one based on the API information.

**Output Format Example**

```json
[
  {
    "name": "GoogleSearch",
    "description": "Users can search the web by entering keywords.",
    "path": "/search",
    "method": "POST",
    "origin": "https://www.google.com",
    "parameters": {
      "header": [],
      "query": [],
      "path": [],
      "body": {
        "type": "object",
        "required": ["q"],
        "properties": {
          "q": {
            "type": "string",
            "description": "the keywords the user needs to search for"
          }
        }
      }
    },
    "auth_config": [
      {
        "location": "header",
        "key": "X-API-KEY",
        "value": "***"
      }
    ]
  }
]
```

**User Input**
{INPUT}
"""

def generate_prompt(query: str, input: list):
    user_input = ""
    if input:
        pass
    user_input += f"Now Input: {query}"
    return PROMPT.replace("{INPUT}", user_input)
