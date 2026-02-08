import { GoogleGenAI } from "@google/genai";
import { ILLMProvider, LLMResponse } from "./ILLMProvider";
import { GoogleGenAIProviderConfig, StreamChunk, TokenUsage } from "../types";

export class GoogleGenAIProvider implements ILLMProvider {
    readonly provider = "genai";
    private client: GoogleGenAI;
    private model: string;

    constructor(config: GoogleGenAIProviderConfig) {
        this.model = config.model;
        this.client = new GoogleGenAI({ apiKey: config.apiKey });
    }

    async invoke(
        prompt: string,
        config: GoogleGenAIProviderConfig
    ): Promise<LLMResponse> {
        const model = config.model ?? this.model;
        const maxTokens = config.maxTokens ?? 3000;
        const temperature = config.temperature;
        const systemInstruction = config.providerOptions?.systemPrompt;

        const thinkingBudget =
            config.thinking?.type === "enabled"
                ? config.thinking.budget_tokens ?? 0
                : 0;

        const contents = [{ parts: [{ text: prompt }] }];

        const response = await this.client.models.generateContent({
            model,
            contents,
            config: {
                maxOutputTokens: config.maxTokens ?? 3000,
                thinkingConfig: { thinkingBudget },
                ...(config.topK !== undefined && { topK: config.topK }),
                ...(config.topP !== undefined && { topP: config.topP }),
                ...(temperature !== undefined && { temperature }),
                ...(systemInstruction && { systemInstruction }),
            },
        });

        return {
            content: response.text ?? "",
            thinking: config.thinking as any,
            usage: {
                inputTokens: response.usageMetadata?.promptTokenCount ?? 0,
                outputTokens: response.usageMetadata?.candidatesTokenCount ?? 0,
                thinkingTokens: thinkingBudget,
                searchCount: 0,
            },
            raw: response,
        };
    }

    async *invokeStream(
        prompt: string,
        config: GoogleGenAIProviderConfig
    ): AsyncGenerator<StreamChunk> {
        const model = config.model ?? this.model;
        const temperature = config.temperature;
        const systemInstruction = config.providerOptions?.systemPrompt;

        const thinkingBudget =
            config.thinking?.type === "enabled"
                ? config.thinking.budget_tokens ?? 0
                : 0;

        const contents = [{ parts: [{ text: prompt }] }];

        const stream = await this.client.models.generateContentStream({
            model,
            contents,
            config: {
                maxOutputTokens: config.maxTokens ?? 3000,
                thinkingConfig: { thinkingBudget },
                ...(config.topK !== undefined && { topK: config.topK }),
                ...(config.topP !== undefined && { topP: config.topP }),
                ...(temperature !== undefined && { temperature }),
                ...(systemInstruction && { systemInstruction }),
            },
        });

        const tokenUsage: TokenUsage = { inputTokens: 0, outputTokens: 0 };

        for await (const chunk of stream) {
            const text = chunk.candidates?.[0]?.content?.parts?.[0]?.text;
            if (text) {
                yield { text };
            }
            // usageMetadata is cumulative; last one is the total
            if (chunk.usageMetadata) {
                tokenUsage.inputTokens = chunk.usageMetadata.promptTokenCount ?? 0;
                tokenUsage.outputTokens = chunk.usageMetadata.candidatesTokenCount ?? 0;
            }
        }

        yield { text: "", tokenUsage };
    }

    supportsBatch(): boolean {
        return false;
    }
}
