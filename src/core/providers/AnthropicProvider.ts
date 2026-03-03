import Anthropic from "@anthropic-ai/sdk";
import { ILLMProvider, LLMResponse, ProviderBatchRequest, ProviderBatchResponse, ProviderBatchItemResult } from "./ILLMProvider";
import { AnthropicConfig, BatchMetadata, BatchStatus, LLMConfig, StreamChunk, TokenUsage } from "../types";
import { Stream } from "@anthropic-ai/sdk/core/streaming";
import {
    RawMessageStreamEvent,
    Message,
    MessageCreateParamsNonStreaming,
    MessageCreateParamsStreaming,
    MessageDeltaUsage
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

    private buildWebTools(config: AnthropicConfig): any[] {
        const tools: any[] = [];
        if (config.webSearch?.enabled) {
            tools.push({
                type: "web_search_20250305" as const,
                name: "web_search" as const,
                ...(config.webSearch.maxUses !== undefined && { max_uses: config.webSearch.maxUses }),
                ...(config.webSearch.allowedDomains && { allowed_domains: config.webSearch.allowedDomains }),
                ...(config.webSearch.userLocation && { user_location: config.webSearch.userLocation }),
            });
        }
        if (config.webFetch?.enabled) {
            tools.push({
                type: "web_fetch_20250910" as const,
                name: "web_fetch" as const,
                ...(config.webFetch.maxUses !== undefined && { max_uses: config.webFetch.maxUses }),
                ...(config.webFetch.allowedDomains && { allowed_domains: config.webFetch.allowedDomains }),
                ...(config.webFetch.citations && { citations: config.webFetch.citations }),
            });
        }
        return tools;
    }

    private buildRequestOptions(config: AnthropicConfig): object | undefined {
        return config.webFetch?.enabled
            ? { headers: { "anthropic-beta": "web-fetch-2025-09-10" } }
            : undefined;
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
            providerOptions,
            stream: stream,
        } = config;

        if (!maxTokens) {
            throw new Error("maxTokens is required for Anthropic models");
        }

        const tools = this.buildWebTools(config);

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

        const requestOptions = this.buildRequestOptions(config);

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
        let usage: Message["usage"] | MessageDeltaUsage | undefined;

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
                            usage = event.usage;
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

    async *invokeStream(
        prompt: string,
        config: AnthropicConfig
    ): AsyncGenerator<StreamChunk> {
        const {
            model,
            maxTokens,
            temperature,
            topK,
            topP,
            thinking,
            providerOptions,
        } = config;

        if (!maxTokens) {
            throw new Error("maxTokens is required for Anthropic models");
        }

        const tools = this.buildWebTools(config);

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

        const requestOptions = this.buildRequestOptions(config);

        const stream = await this.client.messages.create({
            ...baseParams,
            stream: true,
        } as MessageCreateParamsStreaming, requestOptions);

        const tokenUsage: TokenUsage = { inputTokens: 0, outputTokens: 0 };

        for await (const event of stream as Stream<RawMessageStreamEvent>) {
            switch (event.type) {
                case "message_start":
                    tokenUsage.inputTokens = event.message.usage?.input_tokens || 0;
                    break;
                case "content_block_delta":
                    if (event.delta.type === "text_delta") {
                        yield { text: event.delta.text };
                    }
                    break;
                case "message_delta":
                    if (event.usage) {
                        tokenUsage.outputTokens = event.usage.output_tokens || 0;
                    }
                    break;
            }
        }

        yield { text: "", tokenUsage };
    }

    supportsBatch(): boolean {
        return true;
    }

    async createBatch(
        requests: ProviderBatchRequest[],
        config: LLMConfig
    ): Promise<BatchMetadata> {
        const anthropicConfig = config as AnthropicConfig;
        const { model, maxTokens, temperature, topK, topP, thinking, providerOptions } = anthropicConfig;

        if (!maxTokens) {
            throw new Error("maxTokens is required for Anthropic batch requests");
        }

        const tools = this.buildWebTools(anthropicConfig);
        const requestOptions = this.buildRequestOptions(anthropicConfig);

        const batchRequests = requests.map((req) => ({
            custom_id: req.customId,
            params: {
                model,
                max_tokens: maxTokens,
                messages: [{ role: "user" as const, content: req.prompt }],
                ...(temperature !== undefined && { temperature }),
                ...(topK !== undefined && { top_k: topK }),
                ...(topP !== undefined && { top_p: topP }),
                ...(providerOptions?.systemPrompt && { system: providerOptions.systemPrompt }),
                ...(thinking && { thinking }),
                ...(tools.length > 0 && { tools }),
            },
        }));

        const batch = await this.client.messages.batches.create({
            requests: batchRequests,
        }, requestOptions);

        return {
            batchId: batch.id,
            provider: "anthropic",
            model,
            requestCount: requests.length,
            createdAt: new Date().toISOString(),
        };
    }

    async retrieveBatch(
        metadata: BatchMetadata,
        config: LLMConfig
    ): Promise<ProviderBatchResponse> {
        const batch = await this.client.messages.batches.retrieve(metadata.batchId);

        // Map Anthropic processing_status to our BatchStatus
        let status: BatchStatus;
        switch (batch.processing_status) {
            case "in_progress":
                status = "in_progress";
                break;
            case "canceling":
                status = "cancelling";
                break;
            case "ended":
                status = "completed";
                break;
            default:
                status = "in_progress";
        }

        const requestCounts = {
            total: batch.request_counts.processing +
                batch.request_counts.succeeded +
                batch.request_counts.errored +
                batch.request_counts.canceled +
                batch.request_counts.expired,
            completed: batch.request_counts.succeeded,
            failed: batch.request_counts.errored,
            expired: batch.request_counts.expired,
            cancelled: batch.request_counts.canceled,
        };

        // If not ended, return status only
        if (batch.processing_status !== "ended") {
            return { status, requestCounts };
        }

        // Batch is ended — stream results
        const results: ProviderBatchItemResult[] = [];

        for await (const result of await this.client.messages.batches.results(metadata.batchId)) {
            const itemResult: ProviderBatchItemResult = {
                customId: result.custom_id,
                status: 'failed',
            };

            switch (result.result.type) {
                case "succeeded": {
                    const message = result.result.message;
                    let content = "";
                    for (const block of message.content) {
                        if (block.type === "text") {
                            content += block.text;
                        }
                    }
                    itemResult.status = "success";
                    itemResult.content = content;
                    itemResult.tokenUsage = {
                        inputTokens: message.usage?.input_tokens || 0,
                        outputTokens: message.usage?.output_tokens || 0,
                        searchCount: (message.usage as any)?.server_tool_use?.web_search_requests,
                        fetchCount: (message.usage as any)?.server_tool_use?.web_fetch_requests,
                    };
                    break;
                }
                case "errored": {
                    itemResult.status = "failed";
                    itemResult.error = result.result.error?.error?.message || "Unknown error";
                    break;
                }
                case "canceled": {
                    itemResult.status = "cancelled";
                    break;
                }
                case "expired": {
                    itemResult.status = "expired";
                    break;
                }
            }

            results.push(itemResult);
        }

        return { status, results, requestCounts };
    }
}
