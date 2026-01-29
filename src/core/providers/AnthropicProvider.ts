import Anthropic from "@anthropic-ai/sdk";
import { ILLMProvider, LLMResponse } from "./ILLMProvider";
import { AnthropicConfig } from "../types";
import { Stream } from "@anthropic-ai/sdk/core/streaming";
import { 
    RawMessageStreamEvent, 
    Message,
    MessageCreateParamsNonStreaming,
    MessageCreateParamsStreaming 
} from "@anthropic-ai/sdk/resources/messages";

/**
 * Anthropic provider implementation
 */
export class AnthropicProvider implements ILLMProvider {
    private client: Anthropic;
    readonly provider = "anthropic";

    constructor(apiKey?: string) {
        this.client = new Anthropic({
            apiKey: apiKey || process.env.ANTHROPIC_API_KEY,
        });
    }

    async invoke(
        prompt: string,
        config: AnthropicConfig
    ): Promise<LLMResponse> {
        const {
            model,
            temperature,
            maxTokens,
            topK,
            topP,
            thinking,
            webSearch,
            webFetch,
            providerOptions,
            stream: stream,
        } = config;

        if (!maxTokens) {
            throw new Error("maxTokens is required for Anthropic models");
        }

        // Build tools array (can have both web search and web fetch)
        const tools: any[] = [];
        if (webSearch?.enabled) {
            tools.push({
                type: "web_search_20250305" as const,
                name: "web_search" as const,
                ...(webSearch.maxUses !== undefined && { max_uses: webSearch.maxUses }),
                ...(webSearch.allowedDomains && { allowed_domains: webSearch.allowedDomains }),
                ...(webSearch.userLocation && { user_location: webSearch.userLocation }),
            });
        }
        if (webFetch?.enabled) {
            tools.push({
                type: "web_fetch_20250910" as const,
                name: "web_fetch" as const,
                ...(webFetch.maxUses !== undefined && { max_uses: webFetch.maxUses }),
                ...(webFetch.allowedDomains && { allowed_domains: webFetch.allowedDomains }),
                ...(webFetch.citations && { citations: webFetch.citations }),
            });
        }

        const baseParams = {
            model,
            max_tokens: maxTokens,
            messages: [{ role: "user" as const, content: prompt }],
            ...(temperature !== undefined && { temperature }),
            ...(topK !== undefined && { top_k: topK }),
            ...(topP !== undefined && { top_p: topP }),
            ...(providerOptions?.systemPrompt && { system: providerOptions.systemPrompt }),
            ...(thinking && { thinking }),
            ...(tools.length > 0 && { tools }),
        };

        // Add beta header if web fetch is enabled
        const requestOptions = webFetch?.enabled
            ? { headers: { "anthropic-beta": "web-fetch-2025-09-10" } }
            : undefined;

        const response = stream
            ? await this.client.messages.create({
                ...baseParams,
                stream: true,
            } as MessageCreateParamsStreaming, requestOptions)
            : await this.client.messages.create({
                ...baseParams,
                stream: false,
            } as MessageCreateParamsNonStreaming, requestOptions);

        // Extract content and thinking
        let content = "";
        let thinkingContent = "";
        let usage: Message["usage"] | undefined;

        // Stream response handling
        if (stream) {
            const streamResponse = response as Stream<RawMessageStreamEvent>;
            
            for await (const event of streamResponse) {
                switch (event.type) {
                    case "message_start":
                        usage = event.message.usage;
                        break;
                    case "message_delta":
                        // Update usage with delta
                        if (event.usage) {
                            usage = {
                                ...usage,
                                input_tokens: (usage?.input_tokens || 0) + (event.usage.input_tokens || 0),
                                output_tokens: (usage?.output_tokens || 0) + (event.usage.output_tokens || 0),
                            } as Message["usage"];
                        }
                        break;
                    case "content_block_start":
                        if (event.content_block.type === "text") {
                            content += event.content_block.text;
                        } else if (event.content_block.type === "thinking") {
                            thinkingContent += event.content_block.thinking;
                        }
                        break;
                    case "content_block_delta":
                        if (event.delta.type === "text_delta") {
                            content += event.delta.text;
                        } else if (event.delta.type === "thinking_delta") {
                            thinkingContent += event.delta.thinking;
                        }
                        break;
                }
            }
        } else {
            const messageResponse = response as Message;
            usage = messageResponse.usage;
            
            for (const block of messageResponse.content) {
                if (block.type === "text") {
                    content += block.text;
                } else if ((block as any).type === "thinking") {
                    thinkingContent += (block as any).thinking;
                }
            }
        }

        // Calculate thinking tokens (rough estimate if not provided)
        // Anthropic includes thinking tokens in output tokens
        let thinkingTokens = 0;
        if (thinking && usage) {
            // Rough estimate: thinking tokens = total output - content length/4
            // This is an approximation since we don't get exact thinking token count
            const contentTokenEstimate = Math.ceil(content.length / 4);
            thinkingTokens = Math.max(
                0,
                (usage.output_tokens || 0) - contentTokenEstimate
            );
        }

        return {
            content,
            thinking: thinkingContent || undefined,
            usage: {
                inputTokens: usage?.input_tokens || 0,
                outputTokens: usage?.output_tokens || 0,
                thinkingTokens: thinkingTokens,
                searchCount: (usage as any)?.search_count,
                fetchCount: (usage as any)?.fetch_count,
            },
            raw: response,
        };
    }
}
