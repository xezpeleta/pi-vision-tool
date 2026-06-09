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
 * Configure the provider and model via environment variables:
 *   PI_VISION_PROVIDER=my-provider
 *   PI_VISION_MODEL=my-vision-model
 *
 * Make sure the provider and model are defined in ~/.pi/agent/models.json
 * with `input: ["text", "image"]`.
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

import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Type } from "typebox";
import type { Model } from "@earendil-works/pi-ai";
import type { Api } from "@earendil-works/pi-ai";

const VISION_PROVIDER = process.env.PI_VISION_PROVIDER;
const VISION_MODEL_ID = process.env.PI_VISION_MODEL;

if (!VISION_PROVIDER || !VISION_MODEL_ID) {
  // Defer error to tool execution time with a helpful message
}

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

async function imageToBase64(pathOrData: string): Promise<{ mimeType: string; data: string }> {
  // If it looks like a base64 data URL, parse it
  if (pathOrData.startsWith("data:")) {
    const match = /^data:([^;]+);base64,(.+)$/.exec(pathOrData);
    if (match) return { mimeType: match[1], data: match[2] };
  }

  // If it's raw base64 without a data URL prefix, try to detect
  if (/^[A-Za-z0-9+/=]+$/.test(pathOrData) && pathOrData.length > 100) {
    return { mimeType: "image/png", data: pathOrData };
  }

  // Otherwise treat as a file path
  const resolvedPath = resolve(pathOrData);
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
  const buffer = await readFile(resolvedPath);
  return { mimeType, data: buffer.toString("base64") };
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
      model: VISION_MODEL_ID,
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

export default function visionToolExtension(pi: ExtensionAPI) {
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
    ].join("\n"),
    promptSnippet: "Analyze the provided image and respond to the prompt",
    promptGuidelines: [
      "Use describe_image when you need to understand the visual content of any image (screenshot, diagram, photo, etc.). Provide a specific prompt describing exactly what information you need from the image.",
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
    }),
    async execute(_toolCallId, params, signal, onUpdate, ctx) {
      // Validate configuration
      if (!VISION_PROVIDER || !VISION_MODEL_ID) {
        return {
          content: [{
            type: "text",
            text: [
              "Vision tool is not configured.",
              "",
              "Set the PI_VISION_PROVIDER and PI_VISION_MODEL environment variables",
              "to point to a vision-capable model in your models.json.",
              "",
              "Example:",
              "  export PI_VISION_PROVIDER=my-provider",
              "  export PI_VISION_MODEL=my-vision-model",
            ].join("\n"),
          }],
          details: { error: "not_configured" },
          isError: true,
        };
      }

      // Resolve the vision model from the registry
      const visionModel = ctx.modelRegistry.find(VISION_PROVIDER!, VISION_MODEL_ID!);
      if (!visionModel) {
        return {
          content: [
            {
              type: "text",
              text: [
                `Vision tool error: model "${VISION_PROVIDER}/${VISION_MODEL_ID}" not found in model registry.`,
                "",
                "Make sure:",
                "1. The provider and model are defined in ~/.pi/agent/models.json",
                "2. The model has `input: [\"text\", \"image\"]`",
                "3. PI_VISION_PROVIDER and PI_VISION_MODEL env vars are set correctly",
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
              text: `Vision tool error: unable to resolve API key for "${VISION_PROVIDER}". ${auth.error}`,
            },
          ],
          details: { error: "auth_error", authError: auth.error },
          isError: true,
        };
      }

      // Decode the image
      let imageData: { mimeType: string; data: string };
      try {
        imageData = await imageToBase64(params.image_path);
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

      // Notify UI
      onUpdate?.({
        content: [{ type: "text", text: `Analyzing image with ${VISION_MODEL_ID}...` }],
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
            model: `${VISION_PROVIDER}/${VISION_MODEL_ID}`,
            image_path: params.image_path,
            prompt: params.prompt,
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
