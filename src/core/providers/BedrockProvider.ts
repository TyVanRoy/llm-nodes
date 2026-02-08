import { AnthropicBedrock } from "@anthropic-ai/bedrock-sdk";
import { ILLMProvider, LLMResponse } from "./ILLMProvider";
import { BedrockConfig, StreamChunk, TokenUsage } from "../types";
import { Stream } from "@anthropic-ai/sdk/core/streaming";
import {
    RawMessageStreamEvent,
    Message,
    MessageCreateParamsNonStreaming,
    MessageCreateParamsStreaming
} from "@anthropic-ai/sdk/resources/messages";

/**
 * AWS Bedrock provider implementation using Anthropic models
 */
export class BedrockProvider implements ILLMProvider {
    private client: AnthropicBedrock;
    readonly provider = "bedrock";

    constructor(config?: {
        awsRegion?: string;
        awsAccessKeyId?: string;
        awsSecretAccessKey?: string;
        awsSessionToken?: string;
    }) {
        this.client = new AnthropicBedrock({
            awsRegion: config?.awsRegion || process.env.AWS_REGION,
            ...(config?.awsAccessKeyId && { awsAccessKey: config.awsAccessKeyId }),
            ...(config?.awsSecretAccessKey && { awsSecretKey: config.awsSecretAccessKey }),
            ...(config?.awsSessionToken && { awsSessionToken: config.awsSessionToken }),
        });
    }

    async invoke(
        prompt: string,
        config: BedrockConfig
    ): Promise<LLMResponse> {
        const {
            model,
            temperature,
            maxTokens,
            topK,
            topP,
            thinking,
            providerOptions,
            stream,
        } = config;

        if (!maxTokens) {
            throw new Error("maxTokens is required for Bedrock models");
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
        };

        const response = stream
            ? await this.client.messages.create({
                ...baseParams,
                stream: true,
            } as MessageCreateParamsStreaming)
            : await this.client.messages.create({
                ...baseParams,
                stream: false,
            } as MessageCreateParamsNonStreaming);

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
            },
            raw: response,
        };
    }

    async *invokeStream(
        prompt: string,
        config: BedrockConfig
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
            throw new Error("maxTokens is required for Bedrock models");
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
        };

        const stream = await this.client.messages.create({
            ...baseParams,
            stream: true,
        } as MessageCreateParamsStreaming);

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
        return false;
    }
}
