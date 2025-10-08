import { GoogleGenAI } from "@google/genai";
import { ILLMProvider, LLMResponse } from "./ILLMProvider";
import { GoogleGenAIProviderConfig } from "../types";

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
}
