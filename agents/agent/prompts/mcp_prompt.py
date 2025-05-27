
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
   - `auth_config` (optional): Authentication configuration for fixed parameters. This can be:
     - A **single dictionary** for one authentication detail:
       ```json
       {"location": "header" | "param", "key": "your_key_name", "value": "your_key_value"}
       ```
       - `location`: Specifies where the authentication detail should be placed. Can be `"header"` (for request headers) or `"param"` (for URL query parameters).
       - `key`: The name of the header or query parameter.
       - `value`: The corresponding value for the header or query parameter.
     - A **list of dictionaries** for multiple authentication details:
       ```json
       [
         {"location": "header", "key": "Auth-Token", "value": "xyz123"},
         {"location": "param", "key": "api_key", "value": "abc456"}
       ]
       ```
       Each dictionary in the list follows the same structure as the single dictionary format.

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
        "value": "your-api-key"
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
