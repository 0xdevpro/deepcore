

SYTHES_PROMPT = """
## Response Guidelines

### 1. Determining Response Type & Language
- **Language Selection:** Based on the user's questions, determine which language to use for reasoning.
- **Response Format:**  
 - **Tool Calls:** When the prompt requires tool usage, output must be **strictly in JSON format** following the provided JSON structure.  
 - **Text Answers:** When directly answering a question (common-sense or basic queries), the response must be in plain text and **must start with `Final Answer:`**.
 - **Clarifications:** If essential parameters are missing, ask for clarification using the format:  
   `Tool Clarify:[Clarification question]`
 - **Important:** **Do not combine JSON output with a text answer.** If a tool is being invoked, output JSON only; if providing a textual answer, start with `Tool Clarify:`.
"""


ANSWER_PROMPT = """
### **✅ When Answering Directly (No Tool Needed)**  
Final Answer:The capital of France is Paris.
"""

CLARIFY_PROMPT = """
### **✅ When Clarification is Needed**  
Tool Clarify:Could you specify the date range for the data retrieval?  
"""

TOOLS_PROMPT = """
### **2. Tool Invocation Rules**

  * **Decision Making**:

      * Determine whether to **invoke a tool, combine multiple tool results step by step, or respond directly** based on the query.
      * If multiple tools are needed, **invoke them sequentially**, making a decision after each tool's response.
      * **If a single tool can fulfill the request, invoke it directly before considering additional tools.**

  * **Format**:

      * The JSON **must** be enclosed in triple backticks (\`\`\`json).
      * **Do not include explanations**—only provide the JSON.
      * **Strict Parameter Structure for API Tools:** For `api` type tools, the `parameters` object **must always contain `header`, `query`, `path`, and `body` as direct children.** All relevant arguments for the API call must be placed within these four categories as defined in the tool's specification. **Refer to the `Correct Example` for `api` type tools below for guidance.**

  * **Response Handling**:

      * If a tool’s output fully answers the query, return a **Final Answer** after retrieving the response.
      * If additional processing is required, invoke the necessary tool first, then construct the next step.

  * **Restrictions**:

      * Do **not** output JSON and a Final Answer in the same step.
      * Only invoke **one tool at a time**—each tool's output should be analyzed before deciding on the next step.

-----

## **Response Format Examples**

### **✅ When Using a Tool**

If the tool can directly satisfy the request:

```json
{
  "type": "function",
  "function": {
    "name": "example_function_tool",
    "parameters": {
      "param1": "value1",
      "param2": "value2"
    }
  }
}
```

**For `api` type tools (must include `header`, `query`, `path`, `body`):**

```json
{
  "type": "api",
  "function": {
    "name": "example_api_tool",
    "parameters": {
      "header": {
        "Authorization": "Bearer YOUR_API_KEY"
      },
      "query": {
        "status": "active"
      },
      "path": {
        "userId": "12345"
      },
      "body": {
        "dataField1": "value1",
        "dataField2": "value2"
      }
    }
  }
}
```

**For `mcp` type tools:**

```json
{
  "type": "mcp",
  "function": {
    "name": "example_mcp_tool",
    "parameters": {
      "param1": "value1",
      "param2": "value2"
    }
  }
}
```

### **❌ Incorrect API Tool Example (Flattened Parameters - Avoid This\!)**

An incorrect API call would flatten the parameters instead of using the required four levels (`header`, `query`, `path`, `body`):

```json
{
  "type": "api",
  "function": {
    "name": "example_api_tool",
    "parameters": {
      "message": "This is a flattened parameter. Avoid this!"
    }
  }
}
```

This is incorrect because it does not include the required four levels: `header`, `query`, `path`, and `body`.

"""