# Pi Vision Tool

A [Pi Agent](https://github.com/earendil-works/pi-coding-agent) extension that adds a `describe_image` tool, letting **non-multimodal models** (like DeepSeek V4 Pro, GPT-5 Codex without image support, etc.) delegate image analysis to a vision-capable model.

## Features

The calling model has **full control** over every call, deciding what matters for each image:

| Feature | Parameter | What the model controls |
|---|---|---|
| **Compression** | `compress` | `true` for faster/general use, `false` for pixel-perfect accuracy |
| **Reasoning depth** | `reasoning` | `"off"` for instant answers, `"high"`/`"xhigh"` for complex analysis |
| **Prompt** | `prompt` | Free-text instruction: "describe", "extract text", "find the bug", ... |
| **Image source** | `image_path` | File path, data URL, or raw base64 |

This means the model itself decides the cost/quality tradeoff per call — no pre-configuration needed. Just like a developer chooses between a quick `cat` and a deep `git bisect`, the model picks the right tool settings for the job.

### How it works

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

### Reasoning / extended thinking

For vision models with `reasoning: true`, the calling model can choose the reasoning effort per call via the `reasoning` parameter:

| Level | When to use |
|---|---|
| `off` | Simple queries: "what color is this?" |
| `minimal` | Quick checks: "is there an error on this screenshot?" |
| `low` | Basic descriptions, text extraction |
| `medium` | UI analysis, layout descriptions |
| `high` | Architecture diagrams, complex screenshots |
| `xhigh` | Bug hunting, multi-step visual reasoning |

When omitted, the tool uses the configured default (off by default). The calling model should decide based on task complexity — similar to how it picks `compress: true/false`. Read the [models.md](https://github.com/earendil-works/pi-coding-agent/blob/main/docs/models.md#thinking-level-map) thinking level map section for per-model tuning.

Set the default reasoning level via:
```bash
/vision config reasoning-effort medium
# or via env var:
export PI_VISION_REASONING_EFFORT=medium
```

## Installation

### Via npm (recommended)

```bash
pi install npm:pi-vision-tool
```

This is the primary installation method and the way it's listed in the [Pi package gallery](https://pi.dev/packages).

### Via git

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

### 3. Configure the vision model

**Recommended: Use the /vision command (persistent)**

In any Pi session with the extension loaded:

```
/vision config provider my-vision-provider
/vision config model my-vision-model
```

Settings are saved to `~/.pi/agent/vision-tool.json` and persist across all sessions. Changes take effect immediately — no `/reload` or restart needed.

Run `/vision` with no arguments to see current configuration.

**Legacy: Environment variables**

```bash
export PI_VISION_PROVIDER=my-vision-provider
export PI_VISION_MODEL=my-vision-model
```

Env vars work but must be set before starting Pi and don't persist between sessions. When a config file exists, it takes priority over env vars.

### 4. (Optional) Install sharp for image compression

```bash
npm install sharp
```

If `sharp` is available, images are automatically compressed before sending:
- Downscaled to 1568px max dimension (screenshots, high-res photos)
- Alpha channel stripped (RGBA → RGB)
- Lossless PNG converted to JPEG (quality 85)

This reduces payload size ~4x and speeds up responses significantly.
Without `sharp`, images are sent as raw bytes.

### Compression controls

| Env var | Default | Description |
|---|---|---|
| `PI_VISION_MAX_DIM` | `1568` | Max width/height in pixels before downscaling |
| `PI_VISION_JPEG_QUALITY` | `85` | JPEG quality (1-100) for converted images |

The calling model controls per-call compression via the `compress` parameter. Set `compress: false` when pixel-perfect accuracy is needed (e.g., reading coordinates or detecting small UI elements).

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

For complex analysis, the calling model can set `reasoning: "high"`:

```json
{
  "image_path": "/tmp/architecture.png",
  "prompt": "Analyze this system architecture diagram in detail",
  "compress": true,
  "reasoning": "high"
}
```

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
4. Optionally compresses the image (resize, strip alpha, convert to JPEG) via `sharp`
5. Makes a direct OpenAI-compatible `/chat/completions` call to the vision model's base URL
6. Returns the vision model's text response as the tool result

## License

MIT
