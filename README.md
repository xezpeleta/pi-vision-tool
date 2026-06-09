# Pi Vision Tool

A [Pi Agent](https://github.com/earendil-works/pi-coding-agent) extension that adds a `describe_image` tool, letting **non-multimodal models** (like DeepSeek V4 Pro, GPT-5 Codex without image support, etc.) delegate image analysis to a vision-capable model.

## How it works

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  DeepSeek Pro    │────▶│  describe_image  │────▶│  Qwen VL / any   │
│  (no vision)     │     │  (this tool)     │     │  vision model    │
│                  │◀────│                  │◀────│                  │
│  "that's red"    │     │  text response   │     │  "it's red"      │
└──────────────────┘     └──────────────────┘     └──────────────────┘
```

1. The calling model decides it needs to understand an image
2. It calls `describe_image` with an image path and a specific prompt
3. The tool sends the image + prompt to your vision model
4. The vision model's text response is returned to the calling model as a tool result
5. The calling model integrates the result into its reasoning

## Installation

### Via Git (recommended)

```bash
pi install git:github.com/xezpeleta/pi-vision-tool
```

### Via local path

```bash
pi install /path/to/pi-vision-tool
```

### Quick test (no install)

```bash
pi -e /path/to/pi-vision-tool
```

## Configuration

### 1. Add a vision model to `~/.pi/agent/models.json`

```json
{
  "providers": {
    "my-vision-provider": {
      "baseUrl": "https://your-llm-server/v1",
      "apiKey": "$VISION_API_KEY",
      "api": "openai-completions",
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": false
      },
      "models": [
        {
          "id": "my-vision-model",
          "reasoning": true,
          "input": ["text", "image"]
        }
      ]
    }
  }
}
```

The `input: ["text", "image"]` field is required — it tells Pi the model supports images.

### 2. Set the API key in `~/.pi/agent/auth.json`

```json
{
  "my-vision-provider": {
    "type": "api_key",
    "key": "sk-your-key-here"
  }
}
```

### 3. Configure environment variables

```bash
export PI_VISION_PROVIDER=my-vision-provider
export PI_VISION_MODEL=my-vision-model
```

These tell the tool which provider and model to use for image analysis.

## Usage

Once installed, any model in your session will see the `describe_image` tool. Just reference an image in your prompt and the model will call it automatically.

### Example prompts

| What you need | How to ask |
|---|---|
| **Description** | "Describe everything visible in this screenshot" |
| **Pixel coordinates** | "Give [x,y,w,h] bounding boxes for all buttons" |
| **Text extraction** | "Read all visible text, preserving structure" |
| **Error analysis** | "What error is shown in this terminal screenshot?" |
| **UI inspection** | "List all interactive elements and their states" |
| **Color values** | "What hex color is the header bar?" |
| **Layout analysis** | "Describe the page layout: sidebar, main content, etc." |
| **Comparison** | "Compare these two screenshots — what changed?" |

### Image formats

- **File path**: `/tmp/screenshot.png`, `~/Desktop/photo.jpg`
- **Data URL**: `data:image/png;base64,iVBORw0KGgo...`
- **Raw base64**: A base64-encoded string over 100 characters

Supported formats: PNG, JPEG, GIF, WebP, BMP.

## How it works (technical)

The tool:

1. Resolves the vision model from Pi's model registry using `ctx.modelRegistry.find()`
2. Resolves the API key via `ctx.modelRegistry.getApiKeyAndHeaders()`
3. Decodes the image (file path, data URL, or raw base64)
4. Makes a direct OpenAI-compatible `/chat/completions` call to the vision model's base URL
5. Returns the vision model's text response as the tool result

## License

MIT
