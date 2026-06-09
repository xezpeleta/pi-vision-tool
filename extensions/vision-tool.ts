/**
 * Vision Tool — delegates image analysis to a vision-capable model.
 *
 * Non-multimodal models (like DeepSeek Pro, GPT-5 Codex without image support, etc.)
 * can call this tool whenever they need to understand an image.
 *
 * The tool sends the image to a configurable vision model,
 * collects the full text response, and returns it to the calling model as a tool result.
 *
 * ## Configuration
 *
 * The vision model is resolved from Pi's model registry (models.json).
 *
 * **Recommended: /vision command (persistent)**
 *   Use `/vision config provider my-provider` and `/vision config model my-vision-model`
 *   to set the vision model. Settings are saved to `~/.pi/agent/vision-tool.json`
 *   and persist across all sessions. Run `/vision` with no arguments to see
 *   current configuration. Changes take effect immediately — no /reload needed.
 *
 * **Legacy: environment variables**
 *   PI_VISION_PROVIDER=my-provider  PI_VISION_MODEL=my-vision-model
 *   Env vars are read at session start as a fallback when no config file exists.
 *
 * **Priority:** /vision config settings > env vars > built-in defaults
 *
 * Make sure the provider and model are defined in ~/.pi/agent/models.json
 * with `input: ["text", "image"]`.
 *
 * ## Compression
 *
 * Images are automatically preprocessed to reduce payload size and token count:
 * - Downscaled to 1568px max dimension (configurable via PI_VISION_MAX_DIM)
 * - Alpha channel stripped (RGBA → RGB)
 * - Lossless PNG converted to JPEG (quality 85, configurable via PI_VISION_JPEG_QUALITY)
 *
 * Set PI_VISION_COMPRESS=false to disable all preprocessing (send raw bytes).
 * Requires `sharp` for image processing. Falls back to raw bytes if not installed.
 *
 * ## Usage
 *
 * The `prompt` parameter is a free-text instruction, so the calling model can ask
 * for exactly what it needs:
 *
 * - Description: "Describe everything visible in this image"
 * - Coordinates: "Give pixel coordinates [x,y,w,h] of the red button"
 * - Text: "Extract all visible text, preserving structure"
 * - Analysis: "Is there a compiler error shown? What does it say?"
 * - UI: "List all interactive elements and their states"
 * - Colors: "What hex color is the header bar?"
 * - Comparison: "Compare these two screenshots — what changed?"
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { readFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { getAgentDir } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import type { Model } from "@earendil-works/pi-ai";
import type { Api } from "@earendil-works/pi-ai";

// ---------------------------------------------------------------------------
// Config file path
// ---------------------------------------------------------------------------
const CONFIG_PATH = join(getAgentDir(), "vision-tool.json");

// ---------------------------------------------------------------------------
// Runtime config (mutable, populated on session_start)
// ---------------------------------------------------------------------------
interface VisionConfig {
  provider?: string;
  model?: string;
  maxDimension: number;
  jpegQuality: number;
}

let config: VisionConfig = {
  maxDimension: parseInt(process.env.PI_VISION_MAX_DIM ?? "1568", 10),
  jpegQuality: parseInt(process.env.PI_VISION_JPEG_QUALITY ?? "85", 10),
};

const VISION_SYSTEM_PROMPT = [
  "You are an expert vision analysis assistant.",
  "Examine the provided image and respond to the user's request precisely.",
  "",
  "Guidelines:",
  "- If asked for a description, describe everything you see thoroughly.",
  "- If asked for pixel coordinates of elements, provide them in [x, y, width, height] format.",
  "- If asked to read text, extract all visible text verbatim.",
  "- If asked about UI elements, describe their appearance, position, and state.",
  "- Be precise and factual. Do not invent details that are not in the image.",
  "- Structure your response clearly with markdown formatting when appropriate.",
].join("\n");

// ---------------------------------------------------------------------------
// Config persistence helpers
// ---------------------------------------------------------------------------

/** Load config from the JSON file. Returns null if file doesn't exist. */
function loadConfigFile(): VisionConfig | null {
  try {
    if (!existsSync(CONFIG_PATH)) return null;
    const raw = JSON.parse(readFileSync(CONFIG_PATH, "utf-8"));
    return {
      provider: raw.provider || undefined,
      model: raw.model || undefined,
      maxDimension: raw.maxDimension ?? config.maxDimension,
      jpegQuality: raw.jpegQuality ?? config.jpegQuality,
    };
  } catch {
    return null;
  }
}

/** Save current config to the JSON file. */
function saveConfigFile() {
  try {
    mkdirSync(dirname(CONFIG_PATH), { recursive: true });
  } catch {
    // directory already exists or no perms — ignore
  }
  writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2) + "\n");
}

/**
 * Resolve config with priority:
 *   1. Config file (~/.pi/agent/vision-tool.json)
 *   2. Environment variables (PI_VISION_PROVIDER, PI_VISION_MODEL, etc.)
 *   3. Built-in defaults
 *
 * The file wins over env vars so that /vision config changes are sticky.
 */
function resolveConfig(): VisionConfig {
  const fileCfg = loadConfigFile();
  return {
    provider: fileCfg?.provider || process.env.PI_VISION_PROVIDER || undefined,
    model: fileCfg?.model || process.env.PI_VISION_MODEL || undefined,
    maxDimension:
      fileCfg?.maxDimension ??
      parseInt(process.env.PI_VISION_MAX_DIM ?? "1568", 10),
    jpegQuality:
      fileCfg?.jpegQuality ??
      parseInt(process.env.PI_VISION_JPEG_QUALITY ?? "85", 10),
  };
}

/**
 * Build a human-readable config summary for the /vision command.
 */
function configSummary(): string {
  const src = loadConfigFile() ? "config file" : process.env.PI_VISION_PROVIDER ? "env vars" : "none";
  const provider = config.provider ?? "(not set)";
  const model = config.model ?? "(not set)";
  return [
    `Vision tool configuration (source: ${src})`,
    `  Provider:     ${provider}`,
    `  Model:        ${model}`,
    `  Max dim:      ${config.maxDimension}px`,
    `  JPEG quality: ${config.jpegQuality}`,
    ``,
    `Config file: ${CONFIG_PATH}`,
    ``,
    "Use /vision config provider <name> or /vision config model <name> to set.",
  ].join("\n");
}

// ---------------------------------------------------------------------------
// Image processing
// ---------------------------------------------------------------------------

async function imageToBase64(
  pathOrData: string,
  compress: boolean,
): Promise<{ mimeType: string; data: string }> {
  // If it looks like a base64 data URL, parse it
  if (pathOrData.startsWith("data:")) {
    const match = /^data:([^;]+);base64,(.+)$/.exec(pathOrData);
    if (match) {
      const buffer = Buffer.from(match[2], "base64");
      return compress
        ? await optimizeImage(buffer, match[1])
        : { mimeType: match[1], data: match[2] };
    }
  }

  // If it's raw base64 without a data URL prefix, try to detect
  if (/^[A-Za-z0-9+/=]+$/.test(pathOrData) && pathOrData.length > 100) {
    const buffer = Buffer.from(pathOrData, "base64");
    return compress
      ? await optimizeImage(buffer, "image/png")
      : { mimeType: "image/png", data: pathOrData };
  }

  // Otherwise treat as a file path
  const resolvedPath = resolve(pathOrData);
  const buffer = await readFile(resolvedPath);
  const ext = resolvedPath.split(".").pop()?.toLowerCase() ?? "png";
  const mimeMap: Record<string, string> = {
    png: "image/png",
    jpg: "image/jpeg",
    jpeg: "image/jpeg",
    gif: "image/gif",
    webp: "image/webp",
    bmp: "image/bmp",
  };
  const mimeType = mimeMap[ext] ?? "image/png";
  return compress
    ? await optimizeImage(buffer, mimeType)
    : { mimeType, data: buffer.toString("base64") };
}

/**
 * Optimize an image before sending to the vision model.
 * - Downscales if larger than config.maxDimension on either axis
 * - Strips alpha channel (RGBA → RGB)
 * - Converts lossless PNG to JPEG for smaller payload
 * Falls back to raw bytes if sharp is not available.
 */
async function optimizeImage(
  buffer: Buffer,
  originalMime: string,
): Promise<{ mimeType: string; data: string }> {
  if (buffer.length === 0) {
    return { mimeType: originalMime, data: "" };
  }

  try {
    // Dynamic import — users who don't have sharp installed get raw bytes
    const sharp = (await import("sharp")).default;
    let pipeline = sharp(buffer);
    const metadata = await pipeline.metadata();
    const width = metadata.width ?? 0;
    const height = metadata.height ?? 0;

    // Downscale if needed
    if (width > config.maxDimension || height > config.maxDimension) {
      pipeline = pipeline.resize(config.maxDimension, config.maxDimension, {
        fit: "inside",
        withoutEnlargement: true,
      });
    }

    // Strip alpha — vision models often don't need it and it wastes tokens
    if (metadata.hasAlpha || metadata.channels === 4) {
      pipeline = pipeline.removeAlpha();
    }

    // Convert to JPEG for smaller payload (except GIF)
    if (originalMime !== "image/gif") {
      const optimized = await pipeline.jpeg({ quality: config.jpegQuality }).toBuffer();
      return { mimeType: "image/jpeg", data: optimized.toString("base64") };
    }

    // Keep GIF as-is (sharp can't re-encode animated GIF well)
    const optimized = await pipeline.toBuffer();
    return { mimeType: originalMime, data: optimized.toString("base64") };
  } catch {
    // sharp not available or decode failed — send raw bytes
    return { mimeType: originalMime, data: buffer.toString("base64") };
  }
}

async function callVisionModel(
  visionModel: Model<Api>,
  apiKey: string | undefined,
  imageBase64: string,
  mimeType: string,
  prompt: string,
  signal: AbortSignal | undefined,
): Promise<string> {
  const baseUrl = visionModel.baseUrl.replace(/\/+$/, "");

  const messages = [
    {
      role: "system",
      content: VISION_SYSTEM_PROMPT,
    },
    {
      role: "user",
      content: [
        {
          type: "image_url",
          image_url: {
            url: `data:${mimeType};base64,${imageBase64}`,
          },
        },
        {
          type: "text",
          text: prompt,
        },
      ],
    },
  ];

  const response = await fetch(`${baseUrl}/chat/completions`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
    },
    body: JSON.stringify({
      model: visionModel.id,
      messages,
      max_tokens: 4096,
      temperature: 0,
    }),
    signal,
  });

  if (!response.ok) {
    const body = await response.text().catch(() => "");
    throw new Error(`Vision model returned ${response.status}: ${body.slice(0, 500)}`);
  }

  const json = (await response.json()) as {
    choices?: Array<{ message?: { content?: string } }>;
  };

  return json.choices?.[0]?.message?.content ?? "(no response from vision model)";
}

// ---------------------------------------------------------------------------
// Extension entry point
// ---------------------------------------------------------------------------

export default function visionToolExtension(pi: ExtensionAPI) {
  // -----------------------------------------------------------------------
  // Session lifecycle: load & persist config
  // -----------------------------------------------------------------------

  pi.on("session_start", async (_event, ctx) => {
    // Resolve config from file/env, then check for session-persisted overrides
    config = resolveConfig();

    // Restore any mid-session config changes from session entries
    const entries = ctx.sessionManager.getEntries();
    for (const entry of entries) {
      if (entry.type === "custom" && entry.customType === "vision-config") {
        const data = entry.data as Partial<VisionConfig> | undefined;
        if (data?.provider !== undefined) config.provider = data.provider || undefined;
        if (data?.model !== undefined) config.model = data.model || undefined;
        if (data?.maxDimension !== undefined) config.maxDimension = data.maxDimension;
        if (data?.jpegQuality !== undefined) config.jpegQuality = data.jpegQuality;
      }
    }

    updateStatus(ctx);
  });

  /**
   * Persist the current config into the session file.
   * Called whenever config changes via /vision config.
   */
  function persistConfig() {
    pi.appendEntry("vision-config", { ...config });
  }

  // -----------------------------------------------------------------------
  // /vision command
  // -----------------------------------------------------------------------

  pi.registerCommand("vision", {
    description: "Vision tool settings (config, show, clear)",
    handler: async (args, ctx) => {
      const trimmed = args?.trim() ?? "";

      // No args: show current config
      if (!trimmed) {
        ctx.ui.notify(configSummary(), "info");
        return;
      }

      // Parse subcommand
      const parts = trimmed.split(/\s+/);
      const subcommand = parts[0].toLowerCase();
      const rest = parts.slice(1).join(" ");

      // /vision show — show current config
      if (subcommand === "show" || subcommand === "status") {
        ctx.ui.notify(configSummary(), "info");
        return;
      }

      // /vision clear — reset to defaults
      if (subcommand === "clear" || subcommand === "reset") {
        config.provider = undefined;
        config.model = undefined;
        config.maxDimension = parseInt(process.env.PI_VISION_MAX_DIM ?? "1568", 10);
        config.jpegQuality = parseInt(process.env.PI_VISION_JPEG_QUALITY ?? "85", 10);
        saveConfigFile();
        persistConfig();
        updateStatus(ctx);
        ctx.ui.notify("Vision config reset to defaults", "info");
        return;
      }

      // /vision config <setting> [value]
      if (subcommand === "config" || subcommand === "cfg") {
        const settingParts = rest.split(/\s+/);
        const setting = settingParts[0]?.toLowerCase();
        const value = settingParts.slice(1).join(" ");

        if (!setting) {
          ctx.ui.notify(configSummary(), "info");
          return;
        }

        if (setting === "provider") {
          if (!value) {
            ctx.ui.notify(`Current provider: ${config.provider ?? "(not set)"}`, "info");
            return;
          }
          config.provider = value || undefined;
          saveConfigFile();
          persistConfig();
          updateStatus(ctx);
          ctx.ui.notify(`Vision provider set to "${config.provider}" (saved to ${CONFIG_PATH})`, "info");
          return;
        }

        if (setting === "model") {
          if (!value) {
            ctx.ui.notify(`Current model: ${config.model ?? "(not set)"}`, "info");
            return;
          }
          config.model = value || undefined;
          saveConfigFile();
          persistConfig();
          updateStatus(ctx);
          ctx.ui.notify(`Vision model set to "${config.model}" (saved to ${CONFIG_PATH})`, "info");
          return;
        }

        if (setting === "max-dim" || setting === "maxdim") {
          if (!value) {
            ctx.ui.notify(`Current max dimension: ${config.maxDimension}px`, "info");
            return;
          }
          const dim = parseInt(value, 10);
          if (isNaN(dim) || dim < 1) {
            ctx.ui.notify(`Invalid dimension: "${value}". Must be a positive number.`, "error");
            return;
          }
          config.maxDimension = dim;
          saveConfigFile();
          persistConfig();
          ctx.ui.notify(`Max image dimension set to ${config.maxDimension}px`, "info");
          return;
        }

        if (setting === "quality" || setting === "jpeg-quality") {
          if (!value) {
            ctx.ui.notify(`Current JPEG quality: ${config.jpegQuality}`, "info");
            return;
          }
          const q = parseInt(value, 10);
          if (isNaN(q) || q < 1 || q > 100) {
            ctx.ui.notify(`Invalid quality: "${value}". Must be 1-100.`, "error");
            return;
          }
          config.jpegQuality = q;
          saveConfigFile();
          persistConfig();
          ctx.ui.notify(`JPEG quality set to ${config.jpegQuality}`, "info");
          return;
        }

        ctx.ui.notify(
          `Unknown config setting: "${setting}". Use: provider, model, max-dim, or quality`,
          "error",
        );
        return;
      }

      // Shorthand: /vision provider <name> or /vision model <name>
      if (subcommand === "provider") {
        if (!rest) {
          ctx.ui.notify(`Current provider: ${config.provider ?? "(not set)"}`, "info");
          return;
        }
        config.provider = rest || undefined;
        saveConfigFile();
        persistConfig();
        updateStatus(ctx);
        ctx.ui.notify(`Vision provider set to "${config.provider}" (saved to ${CONFIG_PATH})`, "info");
        return;
      }

      if (subcommand === "model") {
        if (!rest) {
          ctx.ui.notify(`Current model: ${config.model ?? "(not set)"}`, "info");
          return;
        }
        config.model = rest || undefined;
        saveConfigFile();
        persistConfig();
        updateStatus(ctx);
        ctx.ui.notify(`Vision model set to "${config.model}" (saved to ${CONFIG_PATH})`, "info");
        return;
      }

      ctx.ui.notify(
        `Unknown subcommand: "${subcommand}". Use: config, show, clear (or provider/model)`,
        "error",
      );
    },
  });

  /**
   * Update the footer status bar to show current vision config.
   */
  function updateStatus(ctx: { ui: { setStatus: (id: string, text: string | undefined) => void } }) {
    if (config.provider && config.model) {
      ctx.ui.setStatus("vision", `👁 ${config.provider}/${config.model}`);
    } else {
      ctx.ui.setStatus("vision", undefined);
    }
  }

  // -----------------------------------------------------------------------
  // describe_image tool
  // -----------------------------------------------------------------------

  pi.registerTool({
    name: "describe_image",
    label: "Describe Image",
    description: [
      "Analyze an image using a vision-capable model.",
      "Use this when you need to understand the content of an image:",
      "screenshots, diagrams, photos, UI mockups, error dialogs, charts, etc.",
      "",
      "The `image_path` can be:",
      "- A file path (e.g., /tmp/screenshot.png)",
      "- A data URL (e.g., data:image/png;base64,...)",
      "- A raw base64-encoded image string",
      "",
      "Set `prompt` to exactly what you need:",
      '- Description: "Describe everything visible in this image"',
      '- Coordinates: "Give pixel coordinates [x,y,w,h] of the red button"',
      '- Text: "Extract all visible text, preserving structure"',
      '- Analysis: "Is there a compiler error shown? What does it say?"',
      "",
      "Set `compress` to control image optimization:",
      "- `true`: Resize large images, strip alpha, convert to JPEG (~4x faster, fewer tokens).",
      "  Use for general descriptions, text extraction, UI analysis.",
      "- `false`: Send raw pixels unchanged.",
      "  Use when you need pixel-perfect analysis: exact coordinates, fine text, color accuracy.",
      "IMPORTANT: Always decide between true/false based on what the user needs.",
    ].join("\n"),
    promptSnippet: "Analyze the provided image and respond to the prompt",
    promptGuidelines: [
      "Use describe_image when you need to understand the visual content of any image (screenshot, diagram, photo, etc.). Provide a specific prompt describing exactly what information you need from the image.",
      "For most tasks (descriptions, text extraction, general analysis), use compress: true.",
      "Only set compress: false for pixel-perfect accuracy (exact coordinates or fine-detail inspection).",
    ],
    parameters: Type.Object({
      image_path: Type.String({
        description:
          "Path to image file, data URL (data:image/...;base64,...), or raw base64-encoded image data",
      }),
      prompt: Type.String({
        description:
          "What to analyze or extract from the image. Be specific: 'Describe all UI elements and their positions', 'Read all text in this screenshot', 'What error is shown?', 'Give coordinates of the submit button', etc.",
      }),
      compress: Type.Boolean({
        description:
          "Whether to compress the image before sending. Use true for most tasks (faster, fewer tokens). Use false when pixel-perfect accuracy is needed (exact coordinates, fine text, color precision).",
      }),
    }),
    async execute(_toolCallId, params, signal, onUpdate, ctx) {
      // Validate configuration
      if (!config.provider || !config.model) {
        return {
          content: [
            {
              type: "text",
              text: [
                "Vision tool is not configured.",
                "",
                "Use /vision to set the vision provider and model:",
                "  /vision config provider my-provider",
                "  /vision config model my-vision-model",
                "",
                "Or set environment variables (legacy):",
                "  export PI_VISION_PROVIDER=my-provider",
                "  export PI_VISION_MODEL=my-vision-model",
              ].join("\n"),
            },
          ],
          details: { error: "not_configured" },
          isError: true,
        };
      }

      // Resolve the vision model from the registry
      const visionModel = ctx.modelRegistry.find(config.provider!, config.model!);
      if (!visionModel) {
        return {
          content: [
            {
              type: "text",
              text: [
                `Vision tool error: model "${config.provider}/${config.model}" not found in model registry.`,
                "",
                "Make sure:",
                "1. The provider and model are defined in ~/.pi/agent/models.json",
                '2. The model has `input: ["text", "image"]`',
                "3. Use /vision show to check or /vision config to update the configuration",
              ].join("\n"),
            },
          ],
          details: { error: "model_not_found" },
          isError: true,
        };
      }

      // Resolve API key
      const auth = await ctx.modelRegistry.getApiKeyAndHeaders(visionModel);
      if (!auth.ok) {
        return {
          content: [
            {
              type: "text",
              text: `Vision tool error: unable to resolve API key for "${config.provider}". ${auth.error}`,
            },
          ],
          details: { error: "auth_error", authError: auth.error },
          isError: true,
        };
      }

      // Decode the image
      const compress = params.compress;
      let imageData: { mimeType: string; data: string };
      try {
        imageData = await imageToBase64(params.image_path, compress);
      } catch (err) {
        return {
          content: [
            {
              type: "text",
              text: `Vision tool error: could not read image "${params.image_path}": ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
          details: { error: "image_read_error" },
          isError: true,
        };
      }

      const compressLabel = compress
        ? `compressed (${(imageData.data.length / 1024).toFixed(0)}KB base64)`
        : `raw (${(imageData.data.length / 1024).toFixed(0)}KB base64)`;

      // Notify UI
      onUpdate?.({
        content: [
          {
            type: "text",
            text: `Analyzing image with ${config.model} (${compressLabel})...`,
          },
        ],
      });

      // Call the vision model
      try {
        const result = await callVisionModel(
          visionModel,
          auth.apiKey,
          imageData.data,
          imageData.mimeType,
          params.prompt,
          signal,
        );

        return {
          content: [{ type: "text", text: result }],
          details: {
            model: `${config.provider}/${config.model}`,
            image_path: params.image_path,
            prompt: params.prompt,
            compressed: compress,
          },
        };
      } catch (err) {
        return {
          content: [
            {
              type: "text",
              text: `Vision tool error: ${err instanceof Error ? err.message : String(err)}`,
            },
          ],
          details: { error: "vision_call_error" },
          isError: true,
        };
      }
    },
  });
}
